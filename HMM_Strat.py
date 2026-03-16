"""
Regime-Adaptive Trading System with Multi-Output Neural Networks
"""

import sys
import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from ta import add_all_ta_features
from hmmlearn import hmm
from statsmodels.tsa.stattools import adfuller
from scipy.special import logsumexp
import tensorflow as tf
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from keras import layers, Model, regularizers, callbacks
from sklearn.preprocessing import RobustScaler

tf.get_logger().setLevel('ERROR')

# GLOBAL CONFIGURATION
CONFIG = {
    # Data parameters
    'TICKER':         'AAPL',
    'START_DATE':     '1993-01-01',
    'END_DATE':       '2025-12-31',
    'NUM_LEAD':       1,
    'HORIZON':        5,

    # Walk-forward parameters
    'WINDOW_SIZE':           4 * 252,   # 4-year rolling training window
    'TRAIN_TEST_SPLIT_DAYS': 90,        # DEPRECATED — no longer used
    'MIN_SAMPLES_HMM': 30,
    'MIN_SAMPLES_NN':  150,
    'RETRAIN_INTERVAL': 20,

    # Neural network architecture
    'SHARED_UNITS':   [8, 4],
    'HEAD_UNITS':     2,
    'DROPOUT_RATE':   0.6,
    'L2_REG':         0.1,
    'L1_REG':         0.005,
    'GAUSSIAN_NOISE': 0.05,
    'USE_BATCH_NORM': True,

    # Training hyperparameters
    'LEARNING_RATE': 0.001,
    'EPOCHS':        100,
    'BATCH_SIZE':    128,
    'PATIENCE':      15,
    'LOSS_WEIGHTS': {
        'sharpe':     0.2,
        'big_move':   0.4,
        'multiclass': 0.4,
    },

    # Signal thresholds
    'SIGNAL_THRESHOLDS': {
        'sharpe':     0.3,
        'big_move':   0.3,
        'multiclass': 0.3,
    },

    # Ensemble seeds
    'ENSEMBLE_SEEDS': [42, 123, 999],

    # Feature Set
    'FIXED_FEATURES': [
        'momentum_rsi',
        'trend_macd',
        'trend_macd_signal',
        'trend_macd_diff',
        'trend_adx',
        'trend_sma_fast',
        'trend_sma_slow',
        'trend_ema_fast',
        'trend_ema_slow',
        'volatility_atr',
        'volatility_bbw',
        'volatility_bbp',
        'volume_obv',
        'volume_vpt',
        'ret_lag1', 'ret_lag2', 'ret_lag3',
        'volume_ratio_20',
        'sma_ratio',
        'ema_ratio',
    ],

    # Short selling flag
    # True  -> signals can be -1 (short), 0 (flat), or +1 (long)
    # False -> signals are 0 (flat) or +1 (long) only  [use for SPY]
    # NOTE: big_move_only is ALWAYS long-or-flat regardless of this flag
    #       because only target_big_up was trained; no big-down head exists.
    'ALLOW_SHORT': True,

    # MA crossover benchmark windows
    'MA_FAST': 50,
    'MA_SLOW': 200,

    # Non-overlapping OOS windows for walk-forward evaluation.
    # Stationarity decisions are made on data strictly before the earliest window start (2018-01-01) so no OOS info leaks into feature engineering.
    'OOS_PERIODS': [
        {'start': '2018-01-01', 'end': '2019-12-31', 'label': '2018-2019'},
        {'start': '2020-01-01', 'end': '2022-12-31', 'label': '2020-2022'},
        {'start': '2023-01-01', 'end': '2024-12-31', 'label': '2023-2024'},
        {'start': '2025-01-01', 'end': '2025-12-31', 'label': '2025'},
    ],

    # Strategies to highlight in the walk-forward summary table and chart.
    'SUMMARY_STRATEGIES': [
        'Big-move only',
        'MA Cross (50/200)',
        'Combined (>=2/3)',
        'Buy & Hold',
    ],
}


# Data Download

def get_data(ticker, start_date, end_date):
    """Downloads and prepares OHLCV data. Returns None on failure."""
    try:
        data = yf.download(
            ticker, start=start_date, end=end_date,
            auto_adjust=True, progress=False
        )
        if data.empty:
            print(f"Warning: no data downloaded for {ticker}.")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data['returns'] = data['Close'].pct_change()
        return data
    except Exception as exc:
        print(f"Error downloading {ticker}: {exc}")
        return None



# Feature Engineering with Multiple Targets

def engineer_features(data_df, num_lead, backtest_start):
    """
    Engineers technical features, ensures stationarity, and creates three
    5-day targets.
    """
    df_ta = pd.DataFrame(index=data_df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_ta[col] = data_df[col].astype(float)

    features_df = add_all_ta_features(
        df_ta,
        open='Open', high='High', low='Low',
        close='Close', volume='Volume',
        fillna=True
    )
    features_df = features_df.drop(
        columns=['Open', 'High', 'Low', 'Close', 'Volume'], errors='ignore'
    )

    # Stationarity: decisions from in-sample only
    to_diff, to_drop = [], []
    in_sample = features_df[features_df.index < pd.to_datetime(backtest_start)]

    for col in in_sample.columns:
        series = in_sample[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 20:
            to_drop.append(col)
            continue
        try:
            pvalue = adfuller(series, regression='c', autolag='AIC')[1]
        except Exception:
            to_drop.append(col)
            continue
        if np.isnan(pvalue):
            to_drop.append(col)
        elif pvalue > 0.05:
            to_diff.append(col)

    if to_diff:
        features_df[to_diff] = features_df[to_diff].pct_change()
    if to_drop:
        features_df.drop(columns=to_drop, inplace=True, errors='ignore')

    df = pd.concat([data_df, features_df], axis=1)

    # Manually computed features
    df['ret_lag1'] = df['returns'].shift(1)
    df['ret_lag2'] = df['returns'].shift(2)
    df['ret_lag3'] = df['returns'].shift(3)
    df['volume_ratio_20'] = df['Volume'] / df['Volume'].rolling(20).mean()
    sma_20 = df['Close'].rolling(20).mean()
    ema_20 = df['Close'].ewm(span=20).mean()
    df['sma_ratio'] = df['Close'] / sma_20 - 1
    df['ema_ratio'] = df['Close'] / ema_20 - 1

    # Targets (5-day forward)
    horizon        = CONFIG['HORIZON']
    forward_ret    = df['Close'].shift(-horizon) / df['Close'] - 1
    fwd_daily_rets = df['returns'].shift(-1)
    forward_vol    = fwd_daily_rets.rolling(horizon).std().shift(-(horizon - 1))

    df['target_sharpe']     = forward_ret / (forward_vol + 1e-9)
    df['target_big_up']     = (forward_ret > 0.03).astype(int)
    df['target_multiclass'] = np.select(
        [forward_ret < -0.02, forward_ret.abs() <= 0.02, forward_ret > 0.02],
        [0, 1, 2], default=1
    )
    df['y_signal'] = np.where(df['returns'].shift(-num_lead) > 0, 1, 0)

    final_features = [c for c in CONFIG['FIXED_FEATURES'] if c in df.columns]
    return df, final_features



# HMM Forward Algorithm + Canonical Labeling

def _forward_filter(hmm_model, X):
    """
    Version-independent causal forward filtering algorithm.
    Uses only public hmmlearn attributes: startprob_, transmat_,
    _compute_log_likelihood. Returns fwd_probs: shape (T, K) — P(z_t=k | x_{1:t}), strictly causal.
    """
    framelogprob = hmm_model._compute_log_likelihood(X)
    T, K         = framelogprob.shape
    log_transmat = np.log(hmm_model.transmat_ + 1e-300)
    log_alpha    = np.empty((T, K))
    log_alpha[0] = np.log(hmm_model.startprob_ + 1e-300) + framelogprob[0]
    for t in range(1, T):
        log_alpha[t] = (
            logsumexp(log_alpha[t - 1, :, None] + log_transmat, axis=0)
            + framelogprob[t]
        )
    log_norm  = logsumexp(log_alpha, axis=1, keepdims=True)
    fwd_probs = np.exp(log_alpha - log_norm)
    return fwd_probs


def fit_hmm_causal(hmm_model, hmm_scaled_vals, returns_vals):
    """
    Causal regime labels + canonical ordering:
        regime 0 = bear (lower mean return in the window)
        regime 1 = bull (higher mean return in the window)
    Returns: regimes array, permutation dict, last forward probs (raw label space).
    """
    fwd_probs          = _forward_filter(hmm_model, hmm_scaled_vals)
    regimes_raw        = fwd_probs.argmax(axis=1)
    last_fwd_probs_raw = fwd_probs[-1]

    mean_ret = {
        r: float(returns_vals[regimes_raw == r].mean())
        if (regimes_raw == r).sum() > 0 else 0.0
        for r in [0, 1]
    }
    perm    = {0: 1, 1: 0} if mean_ret[0] > mean_ret[1] else {0: 0, 1: 1}
    regimes = np.vectorize(perm.get)(regimes_raw)
    return regimes, perm, last_fwd_probs_raw



# Multi-Output Neural Network

def create_multi_output_model(input_dim, config):
    """Builds a multi-output network with L1+L2 regularisation."""
    inputs = layers.Input(shape=(input_dim,), name='input')
    x      = layers.GaussianNoise(config['GAUSSIAN_NOISE'])(inputs)

    for units in config['SHARED_UNITS']:
        x = layers.Dense(
            units, activation='relu',
            kernel_regularizer=regularizers.l1_l2(
                l1=config['L1_REG'], l2=config['L2_REG'])
        )(x)
        if config['USE_BATCH_NORM']:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(config['DROPOUT_RATE'])(x)

    head_units = config.get('HEAD_UNITS', 0)

    def _head(shared, prefix):
        h = layers.Dense(
            head_units, activation='relu',
            kernel_regularizer=regularizers.l1_l2(
                l1=config['L1_REG'], l2=config['L2_REG']),
            name=f'{prefix}_hidden'
        )(shared)
        if config['USE_BATCH_NORM']:
            h = layers.BatchNormalization(name=f'{prefix}_bn')(h)
        return h

    if head_units > 0:
        sharpe_out     = layers.Lambda(lambda z: z * 5.0, name='sharpe')(
            layers.Dense(1, activation='tanh')(_head(x, 'sharpe')))
        bigmove_out    = layers.Dense(
            1, activation='sigmoid', name='big_move')(_head(x, 'big_move'))
        multiclass_out = layers.Dense(
            3, activation='softmax', name='multiclass')(_head(x, 'multiclass'))
    else:
        sharpe_out     = layers.Lambda(lambda z: z * 5.0, name='sharpe')(
            layers.Dense(1, activation='tanh')(x))
        bigmove_out    = layers.Dense(1, activation='sigmoid', name='big_move')(x)
        multiclass_out = layers.Dense(3, activation='softmax', name='multiclass')(x)

    return Model(inputs=inputs, outputs=[sharpe_out, bigmove_out, multiclass_out])


# Signal Combination

def combine_predictions(sharpe_pred, big_prob, multiclass_probs, thresholds,
                        mode='combined', allow_short=False):
    """
    Convert raw head outputs into a trading signal: +1 (long), 0 (flat), -1 (short).
    """
    if mode == 'sharpe_only':
        if not np.isnan(sharpe_pred):
            if sharpe_pred > thresholds['sharpe']:
                return 1
            if allow_short and sharpe_pred < -thresholds['sharpe']:
                return -1
        return 0

    if mode == 'big_move_only':
        return 1 if (not np.isnan(big_prob)
                     and big_prob > thresholds['big_move']) else 0

    if mode == 'multiclass_only':
        if isinstance(multiclass_probs, np.ndarray) and len(multiclass_probs) == 3:
            down, flat, up = multiclass_probs
            if up > thresholds['multiclass'] and up > flat and up > down:
                return 1
            if (allow_short and down > thresholds['multiclass']
                    and down > flat and down > up):
                return -1
        return 0

    # combined: majority consensus >= 2 of 3 heads
    score = 0
    if not np.isnan(sharpe_pred):
        if sharpe_pred > thresholds['sharpe']:
            score += 1
        elif sharpe_pred < -thresholds['sharpe']:
            score -= 1
    if not np.isnan(big_prob) and big_prob > thresholds['big_move']:
        score += 1   # big_move can only vote bullish
    if isinstance(multiclass_probs, np.ndarray) and len(multiclass_probs) == 3:
        down, flat, up = multiclass_probs
        if up > thresholds['multiclass'] and up > flat and up > down:
            score += 1
        elif down > thresholds['multiclass'] and down > flat and down > up:
            score -= 1

    if score >= 2:
        return 1
    if allow_short and score <= -2:
        return -1
    return 0


def apply_signal_mode(results_df, mode, thresholds, allow_short=False):
    """
    Replay a signal mode over stored raw prediction columns.
    Returns a copy with 'signal' column added/replaced.
    """
    df = results_df.copy()

    def _row(row):
        multi = (np.nan if pd.isna(row['pred_multi_down'])
                 else np.array([row['pred_multi_down'],
                                row['pred_multi_flat'],
                                row['pred_multi_up']]))
        return float(combine_predictions(
            row['pred_sharpe'], row['pred_big_prob'], multi,
            thresholds, mode=mode, allow_short=allow_short
        ))

    df['signal'] = df.apply(_row, axis=1)
    return df


# MA Crossover Benchmark

def compute_ma_crossover_signal(price_series, fast=50, slow=200, allow_short=False):
    """
    Causal SMA golden/death-cross signal.

    Signal at day t is computed from MA values at day t-1 (shifted by 1),
    matching the NUM_LEAD=1 convention used throughout the backtest so the
    MA strategy and NN strategies are on an equal causal footing.
    """
    fast_ma = price_series.rolling(fast).mean()
    slow_ma = price_series.rolling(slow).mean()
    raw     = (fast_ma > slow_ma).astype(int)
    if allow_short:
        raw = raw.replace(0, -1)
    # Shift 1: yesterday's MA comparison signals today's position
    return raw.shift(1).fillna(0)



# Walk-Forward Backtest

def run_backtest(data_df, feature_list, backtest_signal_start_date_str,
                 num_lead, config):
    """
    Event-driven walk-forward backtest with per-regime multi-output NNs.

    Stores raw head outputs per day (pred_sharpe, pred_big_prob,
    pred_multi_down/flat/up, pred_regime). No 'signal' column is written —
    all signal modes are applied post-hoc via apply_signal_mode() so every
    mode can be evaluated from a single training run.
    """
    window_size      = config['WINDOW_SIZE']
    min_samples_hmm  = config['MIN_SAMPLES_HMM']
    min_samples_nn   = config['MIN_SAMPLES_NN']
    retrain_interval = config['RETRAIN_INTERVAL']
    horizon          = config['HORIZON']

    data_df = data_df.copy()
    for col in ['pred_sharpe', 'pred_big_prob',
                'pred_multi_down', 'pred_multi_flat', 'pred_multi_up']:
        data_df[col] = np.nan
    data_df['pred_regime'] = -1
    overfitting_log        = []

    # Loop boundaries
    target_start     = pd.to_datetime(backtest_signal_start_date_str)
    dates_after      = data_df.index[data_df.index >= target_start]
    if dates_after.empty:
        print(f"  Start date {backtest_signal_start_date_str} after all data. Aborting.")
        return data_df, overfitting_log
    first_target_idx = data_df.index.get_loc(dates_after[0])
    loop_start       = max(window_size, first_target_idx)

    if config.get('BACKTEST_END') is not None:
        backtest_end_dt = pd.to_datetime(config['BACKTEST_END'])
        dates_before    = data_df.index[data_df.index <= backtest_end_dt]
        loop_end = (data_df.index.get_loc(dates_before[-1]) + 1
                    if not dates_before.empty else len(data_df))
    else:
        loop_end = len(data_df)

    print(f"  Backtest: {data_df.index[loop_start].strftime('%Y-%m-%d')} -> "
          f"{data_df.index[loop_end-1].strftime('%Y-%m-%d')}  "
          f"(retrain every {retrain_interval} days)")

    # Cache
    cached_models       = {0: [], 1: []}
    cached_scalers      = {0: None, 1: None}
    cached_hmm_model    = None
    cached_hmm_perm     = {0: 0, 1: 1}
    cached_last_fwd_raw = np.array([0.5, 0.5])
    last_train_idx      = -retrain_interval

    for t in range(loop_start, loop_end):
        current_date = data_df.index[t]
        data_sample  = data_df.iloc[t - window_size : t].copy()
        if len(data_sample) < window_size:
            continue

        train_data   = data_sample.copy()
        top_features = feature_list

        if len(train_data) < min_samples_hmm * 2:
            continue

        # HMM (every iteration — cheap)
        hmm_features = train_data[['returns']].dropna()
        if len(hmm_features) < min_samples_hmm:
            continue

        hmm_mean        = hmm_features['returns'].median()
        hmm_std         = max(float(hmm_features['returns'].std()), 1e-8)
        hmm_scaled_vals = ((hmm_features - hmm_mean) / hmm_std).values

        hmm_model_obj = hmm.GaussianHMM(
            n_components=2, covariance_type='full',
            n_iter=200, tol=1e-2, min_covar=1e-3, random_state=100
        )
        hmm_model_obj.fit(hmm_scaled_vals)

        regimes, perm, last_fwd_probs_raw = fit_hmm_causal(
            hmm_model_obj, hmm_scaled_vals, hmm_features['returns'].values
        )
        train_data = train_data.copy()
        train_data.loc[hmm_features.index, 'regime'] = regimes

        cached_hmm_model    = hmm_model_obj
        cached_hmm_perm     = perm
        cached_last_fwd_raw = last_fwd_probs_raw

        # NN retrain
        if t - last_train_idx >= retrain_interval:
            regime_models  = {0: [], 1: []}
            regime_scalers = {0: None, 1: None}
            iter_log = {
                'date': current_date,
                'regime_0_trained': False,
                'regime_1_trained': False,
            }

            for regime in [0, 1]:
                if 'regime' not in train_data.columns:
                    continue
                regime_data = train_data[train_data['regime'] == regime].copy()
                if len(regime_data) < min_samples_nn:
                    continue

                X        = regime_data[top_features].iloc[:-horizon]
                y_sharpe = regime_data['target_sharpe'].iloc[:-horizon]
                y_big    = regime_data['target_big_up'].iloc[:-horizon]
                y_multi  = regime_data['target_multiclass'].iloc[:-horizon]

                X        = X.replace([np.inf, -np.inf], np.nan)
                y_sharpe = y_sharpe.replace([np.inf, -np.inf], np.nan)
                y_big    = y_big.replace([np.inf, -np.inf], np.nan)
                y_multi  = y_multi.replace([np.inf, -np.inf], np.nan)

                mask     = ~(X.isnull().any(axis=1) | y_sharpe.isnull()
                             | y_big.isnull() | y_multi.isnull())
                X        = X[mask]
                y_sharpe = y_sharpe[mask].clip(-5, 5)
                y_big    = y_big[mask]
                y_multi  = y_multi[mask]

                if len(X) < min_samples_nn:
                    continue

                scaler   = RobustScaler(quantile_range=(5.0, 95.0))
                X_scaled = scaler.fit_transform(X)
                split    = int(len(X) * 0.7)
                X_tr, X_val = X_scaled[:split], X_scaled[split:]
                y_tr  = {'sharpe':     y_sharpe.values[:split],
                         'big_move':   y_big.values[:split],
                         'multiclass': y_multi.values[:split]}
                y_val = {'sharpe':     y_sharpe.values[split:],
                         'big_move':   y_big.values[split:],
                         'multiclass': y_multi.values[split:]}

                train_losses, val_losses = [], []

                for seed in config['ENSEMBLE_SEEDS']:
                    tf.random.set_seed(seed)
                    np.random.seed(seed)

                    model = create_multi_output_model(X.shape[1], config)
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(
                            learning_rate=config['LEARNING_RATE'], clipnorm=1.0),
                        loss={'sharpe':     'mse',
                              'big_move':   'binary_crossentropy',
                              'multiclass': 'sparse_categorical_crossentropy'},
                        loss_weights=config['LOSS_WEIGHTS'],
                        metrics={'sharpe':     ['mae'],
                                 'big_move':   ['accuracy'],
                                 'multiclass': ['accuracy']}
                    )
                    early_stop = callbacks.EarlyStopping(
                        monitor='val_loss', patience=config['PATIENCE'],
                        restore_best_weights=True)
                    reduce_lr  = callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.5,
                        patience=5, min_lr=1e-6, verbose=0)
                    history = model.fit(
                        X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=config['EPOCHS'], batch_size=config['BATCH_SIZE'],
                        callbacks=[early_stop, reduce_lr], verbose=0)
                    regime_models[regime].append(model)
                    train_losses.append(history.history['loss'][-1])
                    val_losses.append(history.history['val_loss'][-1])

                regime_scalers[regime] = scaler
                iter_log[f'regime_{regime}_trained']    = True
                iter_log[f'regime_{regime}_train_loss'] = float(np.mean(train_losses))
                iter_log[f'regime_{regime}_val_loss']   = float(np.mean(val_losses))
                iter_log[f'regime_{regime}_best_epoch'] = early_stop.best_epoch

            cached_models  = regime_models
            cached_scalers = regime_scalers
            last_train_idx = t
            overfitting_log.append(iter_log)

        # Next-day regime prediction
        next_probs_raw    = cached_last_fwd_raw @ cached_hmm_model.transmat_
        perm_inv          = {v: k for k, v in cached_hmm_perm.items()}
        next_regime_probs = np.array([next_probs_raw[perm_inv[0]],
                                      next_probs_raw[perm_inv[1]]])
        chosen_regime     = 0 if next_regime_probs[0] > next_regime_probs[1] else 1
        model_list        = cached_models[chosen_regime]
        scaler            = cached_scalers[chosen_regime]

        sharpe_pred, big_prob, multiclass_probs = np.nan, np.nan, np.nan
        features_today = (data_sample[top_features].iloc[-num_lead:].copy()
                          if top_features else pd.DataFrame())

        if (model_list and scaler is not None
                and not features_today.empty
                and not features_today.isnull().values.any()
                and not np.isinf(features_today.values).any()):
            X_today    = scaler.transform(features_today)
            all_s, all_b, all_m = [], [], []
            for m in model_list:
                preds = m.predict(X_today, verbose=0)
                all_s.append(preds[0][0, 0])
                all_b.append(preds[1][0, 0])
                all_m.append(preds[2][0])
            sharpe_pred      = float(np.mean(all_s))
            big_prob         = float(np.mean(all_b))
            multiclass_probs = np.mean(all_m, axis=0)

        data_df.loc[current_date, 'pred_sharpe']   = sharpe_pred
        data_df.loc[current_date, 'pred_big_prob'] = big_prob
        data_df.loc[current_date, 'pred_regime']   = chosen_regime
        if isinstance(multiclass_probs, np.ndarray) and len(multiclass_probs) == 3:
            data_df.loc[current_date, 'pred_multi_down'] = float(multiclass_probs[0])
            data_df.loc[current_date, 'pred_multi_flat'] = float(multiclass_probs[1])
            data_df.loc[current_date, 'pred_multi_up']   = float(multiclass_probs[2])

        if t % 100 == 0 or t == loop_start:
            print(f"    {current_date.strftime('%Y-%m-%d')}  "
                  f"sharpe={sharpe_pred:.3f}  big={big_prob:.3f}  "
                  f"regime={chosen_regime}")

    return data_df, overfitting_log


# Overfitting Diagnostics

def analyze_overfitting_log(overfitting_log):
    """Prints per-regime train/val loss summary and overfitting % per period."""
    if not overfitting_log:
        print("  No overfitting log data.")
        return
    log_df = pd.DataFrame(overfitting_log)
    for regime in [0, 1]:
        tc = f'regime_{regime}_train_loss'
        vc = f'regime_{regime}_val_loss'
        if tc not in log_df.columns or vc not in log_df.columns:
            continue
        valid = log_df[[tc, vc]].dropna()
        if valid.empty:
            print(f"  Regime {regime}: no completed windows.")
            continue
        gap         = valid[tc] - valid[vc]
        pct_overfit = (gap < 0).sum() / len(gap) * 100
        print(f"  Regime {regime} ({len(valid)} windows)  "
              f"train={valid[tc].mean():.4f}  val={valid[vc].mean():.4f}  "
              f"gap={gap.mean():.4f}  overfit%={pct_overfit:.1f}%")


# Performance Helpers

def _perf_stats(rets):
    """Return dict of annualised performance statistics for a return series."""
    rets = rets.dropna()
    if rets.empty:
        return {k: np.nan for k in ['ann_ret', 'cum_ret', 'ann_vol',
                                     'sharpe', 'calmar', 'max_dd', 'sortino']}
    years   = len(rets) / 252
    cum     = (1 + rets).prod() - 1
    ann_ret = (1 + cum) ** (1 / years) - 1 if years > 0 else 0.0
    ann_vol = float(rets.std() * np.sqrt(252))
    sharpe  = ann_ret / ann_vol if ann_vol else np.nan
    cum_s   = (1 + rets).cumprod()
    peak    = cum_s.expanding().max()
    max_dd  = float(((cum_s - peak) / peak).min())
    calmar  = ann_ret / abs(max_dd) if max_dd else np.nan
    down    = rets[rets < 0]
    sortino = (ann_ret / (down.std() * np.sqrt(252))
               if len(down) > 0 and down.std() > 0 else np.nan)
    return dict(ann_ret=ann_ret, cum_ret=cum, ann_vol=ann_vol,
                sharpe=sharpe, calmar=calmar, max_dd=max_dd, sortino=sortino)


def _signal_stats(results_df, signal_col='signal'):
    """Return (pct_long, pct_short) tuple."""
    sig = results_df[signal_col].dropna()
    n   = len(sig)
    return (float((sig ==  1).sum() / n) if n else np.nan,
            float((sig == -1).sum() / n) if n else np.nan)


# Compare_signal_modes

def compare_signal_modes(results_df, raw_returns, thresholds, ticker_label,
                         allow_short=False, ma_signal_series=None):
    """
    Apply all NN modes + optional MA crossover benchmark to stored predictions,
    print a side-by-side comparison table and a cumulative return chart.
    """
    MODES = [
        ('sharpe_only',     'Sharpe only',     'steelblue'),
        ('big_move_only',   'Big-move only',   'darkorange'),
        ('multiclass_only', 'Multiclass only', 'green'),
        ('combined',        'Combined (>=2/3)', 'crimson'),
    ]
    MA_LABEL  = 'MA Cross (50/200)'
    BAH_LABEL = 'Buy & Hold'

    all_stats = {}
    all_sigs  = {}
    all_rets  = {}

    # NN modes
    for mode_key, mode_label, _ in MODES:
        df_mode    = apply_signal_mode(results_df, mode_key, thresholds,
                                       allow_short=allow_short)
        df_mode    = df_mode.dropna(subset=['returns'])
        strat_rets = (df_mode['returns'] * df_mode['signal']).dropna()
        stats      = _perf_stats(strat_rets)
        pl, ps     = _signal_stats(df_mode)
        stats['pct_long']  = pl
        stats['pct_short'] = ps
        all_stats[mode_label] = stats
        all_sigs[mode_label]  = df_mode['signal']
        all_rets[mode_label]  = strat_rets

    # MA crossover benchmark
    if ma_signal_series is not None:
        ma_aligned = ma_signal_series.reindex(results_df.index).fillna(0)
        ma_rets    = (results_df['returns'] * ma_aligned).dropna()
        ma_stats   = _perf_stats(ma_rets)
        ma_stats['pct_long']  = float((ma_aligned ==  1).sum() / len(ma_aligned))
        ma_stats['pct_short'] = float((ma_aligned == -1).sum() / len(ma_aligned))
        all_stats[MA_LABEL] = ma_stats
        all_sigs[MA_LABEL]  = ma_aligned
        all_rets[MA_LABEL]  = ma_rets

    # Buy-and-hold
    bah_rets  = raw_returns.dropna()
    bah_stats = _perf_stats(bah_rets)
    bah_stats['pct_long']  = 1.0
    bah_stats['pct_short'] = 0.0
    all_stats[BAH_LABEL] = bah_stats
    all_rets[BAH_LABEL]  = bah_rets

    # Column order: NN modes | MA (if present) | B&H
    col_order = [lbl for _, lbl, _ in MODES]
    if ma_signal_series is not None:
        col_order.append(MA_LABEL)
    col_order.append(BAH_LABEL)

    # Print table
    METRICS = [
        ('ann_ret',   'Annual return',  True),
        ('cum_ret',   'Cumulative ret', True),
        ('ann_vol',   'Annual vol',     True),
        ('sharpe',    'Sharpe ratio',   False),
        ('calmar',    'Calmar ratio',   False),
        ('max_dd',    'Max drawdown',   True),
        ('sortino',   'Sortino ratio',  False),
        ('pct_long',  '% days long',    True),
        ('pct_short', '% days short',   True),
    ]
    col_w   = 16
    label_w = 18
    total_w = label_w + col_w * len(col_order)

    print(f"SIGNAL MODE COMPARISON -- {ticker_label}")
    print(f"{'Metric':<{label_w}}" + ''.join(f"{c:>{col_w}}" for c in col_order))
    for key, mlabel, is_pct in METRICS:
        row = f"{mlabel:<{label_w}}"
        for col in col_order:
            v = all_stats[col].get(key, np.nan)
            if np.isnan(v):
                row += f"{'N/A':>{col_w}}"
            elif is_pct:
                row += f"{v*100:>{col_w-1}.2f}%"
            else:
                row += f"{v:>{col_w}.3f}"
        print(row)

    # Signal distribution
    short_note = "  [big_move_only: long-or-flat only]" if allow_short else ""
    print(f"\nSignal distribution (days):{short_note}")
    hdr = (f"{'Strategy':<22}  {'Long':>8}  {'Short':>8}  "
           f"{'Flat':>8}  {'% Long':>8}  {'% Short':>8}")
    print(hdr)
    for lbl, sig in all_sigs.items():
        n  = len(sig)
        nl = int((sig ==  1).sum())
        ns = int((sig == -1).sum())
        nf = int((sig ==  0).sum())
        print(f"{lbl:<22}  {nl:>8}  {ns:>8}  {nf:>8}  "
              f"{nl/n*100:>7.1f}%  {ns/n*100:>7.1f}%")

    # Cumulative return chart
    color_map = {lbl: c for _, lbl, c in MODES}
    color_map[MA_LABEL]  = 'purple'
    color_map[BAH_LABEL] = 'black'

    fig, ax = plt.subplots(figsize=(15, 7))
    for col in col_order:
        rets = all_rets[col].dropna()
        cum  = (1 + rets).cumprod()
        ls   = ('--' if col == BAH_LABEL else (':' if col == MA_LABEL else '-'))
        lw   = 1.2 if col in (BAH_LABEL, MA_LABEL) else 1.8
        ax.plot(cum, label=col, color=color_map.get(col, 'grey'),
                linewidth=lw, linestyle=ls, alpha=0.9)

    subtitle = " (shorting enabled)" if allow_short else " (long-only)"
    ax.set_title(f"{ticker_label} -- Strategy Comparison{subtitle}", fontsize=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return (1 = start)", fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()

    # Return stats dict for walk-forward aggregation
    return all_stats


# Walk-Forward Summary

def print_walk_forward_summary(period_stats_list, focus_strategies=None):
    """
    Average metrics across all OOS periods, print mean +/- std table,
    print a per-period Sharpe table, and plot a Sharpe persistence chart.
    The Sharpe persistence chart is the key diagnostic: a strategy with
    genuine edge should show positive Sharpe across ALL periods, not just one.
    """
    if not period_stats_list:
        print("No period stats to summarise.")
        return

    period_labels = [lbl for lbl, _ in period_stats_list]
    strategies    = focus_strategies or list(period_stats_list[0][1].keys())

    METRICS = [
        ('ann_ret',  'Ann. Return',  True),
        ('ann_vol',  'Ann. Vol',     True),
        ('sharpe',   'Sharpe',       False),
        ('max_dd',   'Max DD',       True),
        ('calmar',   'Calmar',       False),
        ('pct_long', '% Long',       True),
    ]

    # Collect per-period values for each (strategy, metric) pair
    collected = {s: {k: [] for k, _, _ in METRICS} for s in strategies}
    for _, stats_dict in period_stats_list:
        for s in strategies:
            sd = stats_dict.get(s, {})
            for k, _, _ in METRICS:
                collected[s][k].append(sd.get(k, np.nan))

    col_w   = 22
    label_w = 16
    n_s     = len(strategies)
    total_w = label_w + col_w * n_s

    print(f"WALK-FORWARD SUMMARY  --  {len(period_labels)} periods: "
          f"{period_labels[0]} -> {period_labels[-1]}")
    print(f"Values shown as  mean +/- std  across periods")
    print(f"{'Metric':<{label_w}}" + ''.join(f"{s:>{col_w}}" for s in strategies))

    for key, mlabel, is_pct in METRICS:
        row = f"{mlabel:<{label_w}}"
        for s in strategies:
            vals  = [v for v in collected[s][key] if not np.isnan(v)]
            if not vals:
                row += f"{'N/A':>{col_w}}"
                continue
            mean_ = np.mean(vals)
            std_  = np.std(vals)
            cell  = (f"{mean_*100:.1f}+/-{std_*100:.1f}%"
                     if is_pct else f"{mean_:.2f}+/-{std_:.2f}")
            row += f"{cell:>{col_w}}"
        print(row)

    # Per-period Sharpe table — key diagnostic for persistence
    print(f"\nSharpe ratio per period:")
    print(f"{'Period':<20}" + ''.join(f"{s:>{col_w}}" for s in strategies))
    for plabel, stats_dict in period_stats_list:
        row = f"{plabel:<20}"
        for s in strategies:
            v = stats_dict.get(s, {}).get('sharpe', np.nan)
            row += (f"{v:>{col_w}.3f}" if not np.isnan(v)
                    else f"{'N/A':>{col_w}}")
        print(row)
    print("NOTE: positive Sharpe in ALL periods = genuine edge; "
          "positive in only 1-2 periods = noise.")

    # Sharpe persistence chart
    COLOR_MAP = {
        'Big-move only':    'darkorange',
        'MA Cross (50/200)':'purple',
        'Buy & Hold':       'black',
        'Combined (>=2/3)': 'crimson',
        'Sharpe only':      'steelblue',
        'Multiclass only':  'green',
    }
    x   = np.arange(len(period_labels))
    fig, ax = plt.subplots(figsize=(12, 5))
    for s in strategies:
        sharpes = [period_stats_list[i][1].get(s, {}).get('sharpe', np.nan)
                   for i in range(len(period_labels))]
        ls  = '--' if s == 'Buy & Hold' else '-'
        col = COLOR_MAP.get(s, 'grey')
        ax.plot(x, sharpes, marker='o', label=s, color=col,
                linewidth=1.8, linestyle=ls, markersize=7)

    ax.axhline(0, color='red', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(period_labels, fontsize=10)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_title("Sharpe Ratio Persistence Across OOS Periods", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()


# Overfitting Persistence Across Periods

def plot_overfitting_persistence(all_oof_logs, period_labels):
    """
    For each OOS period plot:
      Left  panel: % training windows where train_loss < val_loss per regime.
                   Values consistently above 50% indicate structural overfitting.
      Right panel: mean (train_loss - val_loss) gap per regime.
                   Negative = overfitting; positive = underfitting.
    Also prints a numeric summary table.
    """
    r0_pct, r1_pct = [], []
    r0_gap, r1_gap = [], []

    for oof_log in all_oof_logs:
        if not oof_log:
            for lst in [r0_pct, r1_pct, r0_gap, r1_gap]:
                lst.append(np.nan)
            continue
        log_df = pd.DataFrame(oof_log)

        def _stats(regime):
            tc = f'regime_{regime}_train_loss'
            vc = f'regime_{regime}_val_loss'
            if tc not in log_df.columns or vc not in log_df.columns:
                return np.nan, np.nan
            valid = log_df[[tc, vc]].dropna()
            if valid.empty:
                return np.nan, np.nan
            gap = valid[tc] - valid[vc]
            return float((gap < 0).sum() / len(gap) * 100), float(gap.mean())

        p0, g0 = _stats(0);  r0_pct.append(p0);  r0_gap.append(g0)
        p1, g1 = _stats(1);  r1_pct.append(p1);  r1_gap.append(g1)

    x   = np.arange(len(period_labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overfitting frequency
    ax = axes[0]
    ax.plot(x, r0_pct, marker='o', label='Regime 0 (bear)',
            color='steelblue', linewidth=1.8)
    ax.plot(x, r1_pct, marker='s', label='Regime 1 (bull)',
            color='darkorange', linewidth=1.8)
    ax.axhline(50, color='red', linewidth=0.8, linestyle='--', alpha=0.6,
               label='50% reference')
    ax.set_xticks(x)
    ax.set_xticklabels(period_labels, fontsize=9)
    ax.set_ylabel("% windows: train_loss < val_loss", fontsize=11)
    ax.set_title("Overfitting Frequency by Period", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)

    # Right: mean train-val gap
    ax = axes[1]
    ax.plot(x, r0_gap, marker='o', label='Regime 0 (bear)',
            color='steelblue', linewidth=1.8)
    ax.plot(x, r1_gap, marker='s', label='Regime 1 (bull)',
            color='darkorange', linewidth=1.8)
    ax.axhline(0, color='red', linewidth=0.8, linestyle='--', alpha=0.6,
               label='0 = balanced')
    ax.set_xticks(x)
    ax.set_xticklabels(period_labels, fontsize=9)
    ax.set_ylabel("Mean gap (train loss - val loss)", fontsize=11)
    ax.set_title("Train-Val Gap by Period  [negative = overfitting]", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)

    plt.suptitle("Regime Overfitting Persistence Across OOS Periods",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()

    # Numeric table
    print(f"\nOverfitting persistence table:")
    print(f"{'Period':<20}  {'R0 %OVF':>10}  {'R0 Gap':>10}  "
          f"{'R1 %OVF':>10}  {'R1 Gap':>10}")
    for i, lbl in enumerate(period_labels):
        def _f(v):
            return f"{v:>10.1f}" if not np.isnan(v) else f"{'N/A':>10}"
        print(f"{lbl:<20}  {_f(r0_pct[i])}  {_f(r0_gap[i])}  "
              f"{_f(r1_pct[i])}  {_f(r1_gap[i])}")


# Main Execution

if __name__ == '__main__':
    cfg = CONFIG

    # Download full price history once
    raw = get_data(cfg['TICKER'], cfg['START_DATE'], cfg['END_DATE'])
    if raw is None or raw.empty:
        print(f"Failed to download data for {cfg['TICKER']}. Exiting.")
        sys.exit(1)

    # Engineer features once.
    EARLIEST_OOS = cfg['OOS_PERIODS'][0]['start']
    data, features = engineer_features(raw.copy(), cfg['NUM_LEAD'], EARLIEST_OOS)
    print(f"Features   : {len(features)}")
    print(f"Data shape : {data.shape}")
    print(f"Date range : {data.index[0].date()} -> {data.index[-1].date()}")

    # Compute MA crossover signal on the FULL price history.
    ma_signal_full = compute_ma_crossover_signal(
        raw['Close'],
        fast=cfg['MA_FAST'],
        slow=cfg['MA_SLOW'],
        allow_short=cfg['ALLOW_SHORT'],
    )

    # 4. Walk-forward loop over three non-overlapping OOS windows
    all_period_stats = []   # list of (label, stats_dict) — one per period
    all_oof_logs     = []   # list of overfitting_log lists — one per period
    all_results_oos = []          # list of DataFrames, one per period
    all_ma_oos = []

    for period in cfg['OOS_PERIODS']:
        print(f"OOS PERIOD: {period['label']}  "
              f"({period['start']} -> {period['end']})")

        period_cfg = cfg.copy()
        period_cfg['BACKTEST_START'] = period['start']
        period_cfg['BACKTEST_END']   = period['end']

        # Train ONCE per period — all signal modes evaluated from one run
        results, oof_log = run_backtest(
            data.copy(), features,
            period['start'], cfg['NUM_LEAD'], period_cfg
        )

        results_oos = results[
            (results.index >= pd.to_datetime(period['start'])) &
            (results.index <= pd.to_datetime(period['end']))
        ].copy()
        all_results_oos.append(results_oos)

        if results_oos.empty:
            print("  No results — check data availability.")
            all_period_stats.append((period['label'], {}))
            all_oof_logs.append([])
            continue

        # Slice MA signal to this OOS window (already causal from full history)
        ma_oos = ma_signal_full.reindex(results_oos.index).fillna(0)
        all_ma_oos.append(ma_oos)

        # Evaluate all signal modes — zero retraining needed
        period_stats = compare_signal_modes(
            results_oos,
            results_oos['returns'].dropna(),
            cfg['SIGNAL_THRESHOLDS'],
            f"{cfg['TICKER']} -- {period['label']}",
            allow_short=cfg['ALLOW_SHORT'],
            ma_signal_series=ma_oos,
        )

        # Overfitting diagnostics for this period
        print(f"\nOVERFITTING ANALYSIS -- {period['label']}")
        analyze_overfitting_log(oof_log)

        all_period_stats.append((period['label'], period_stats))
        all_oof_logs.append(oof_log)
        
    # Begin inserted code for full-period cumulative plot
    if all_results_oos:
        full_results = pd.concat(all_results_oos).sort_index()
        full_ma      = pd.concat(all_ma_oos).sort_index()

        # Define which strategies to plot
        plot_modes = {
            'Combined (>=2/3)': 'combined',
            'Sharpe only': 'sharpe_only',
            'Big-move only': 'big_move_only',
            'MA Cross (50/200)': None,
            'Buy & Hold': None,
        }

        cum_curves = {}
        for label, mode_key in plot_modes.items():
            if mode_key is not None:
                df_mode = apply_signal_mode(
                    full_results, mode_key,
                    cfg['SIGNAL_THRESHOLDS'],
                    allow_short=cfg['ALLOW_SHORT']
                )
                strat_rets = df_mode['returns'] * df_mode['signal']
                cum_curves[label] = (1 + strat_rets).cumprod()
            elif label == 'Buy & Hold':
                cum_curves[label] = (1 + full_results['returns']).cumprod()
            elif label == 'MA Cross (50/200)':
                ma_rets = full_results['returns'] * full_ma
                cum_curves[label] = (1 + ma_rets).cumprod()

        # Plot
        plt.figure(figsize=(16, 8))
        for label, cum in cum_curves.items():
            plt.plot(cum.index, cum, label=label, linewidth=1.8)

        plt.title(f"{cfg['TICKER']} – Full OOS Performance (2018–2025)", fontsize=15)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return (1 = start)")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    # End inserted code

    # Walk-forward aggregated summary across all periods
    if len(all_period_stats) > 1:
        print("WALK-FORWARD AGGREGATED RESULTS")
        print_walk_forward_summary(
            all_period_stats,
            focus_strategies=cfg['SUMMARY_STRATEGIES'],
        )

        # Overfitting persistence chart + table
        print("\nOVERFITTING PERSISTENCE ACROSS PERIODS")
        plot_overfitting_persistence(
            all_oof_logs,
            [lbl for lbl, _ in all_period_stats],
        )

    print("\nAll done.")
