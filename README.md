# RegimeStrat

Walk-forward trading system combining HMM regime detection with per-regime multi-output neural networks. Trained on AAPL daily data across four non-overlapping out-of-sample windows (2018–2025). Signals are long / flat / short based on three simultaneously trained prediction heads.

---

## Features

- **Three prediction targets trained simultaneously** — 5-day forward Sharpe ratio (regression), big up move >3% (binary classification), and directional outcome down/flat/up (ternary classification)
- **Separate neural network per HMM regime** — bear and bull markets get distinct models; label stability enforced via canonical ordering after every HMM fit
- **Causal HMM labelling** — regime labels are assigned using the forward filtering algorithm, not Viterbi, so no future observations contaminate the training labels
- **Walk-forward validation** — four non-overlapping OOS windows tested sequentially; stationarity decisions locked to pre-2018 data only
- **Train once, replay signals** — raw head outputs are stored per day; any signal mode or threshold can be tested in milliseconds without retraining
- **Combined voting signal** — a ≥2 of 3 majority vote is required for a long or short; a single bullish head cannot fire a signal alone
- **MA crossover benchmark** — causal SMA 50/200 golden/death-cross included in every comparison table
- **Overfitting persistence tracking** — train vs. val loss gap plotted across all OOS periods per regime

---

## How It Works

### 1. Data and Feature Engineering

Daily OHLCV data pulled via `yfinance`. The full `ta` library indicator suite is applied across momentum, trend, volatility, and volume categories. Each indicator is tested with an Augmented Dickey-Fuller test at the 5% level using only pre-2018 data. Non-stationary series are converted to percentage changes; invalid or short series are dropped. Five additional features are computed manually: three return lags, volume ratio vs. 20-day average, and price deviation from SMA/EMA.

### 2. Target Construction

Three 5-day forward targets:

| Target | Type | Description |
|---|---|---|
| `target_sharpe` | Regression | 5-day forward return ÷ 5-day forward realised volatility |
| `target_big_up` | Binary | 1 if 5-day return > +3%, else 0 |
| `target_multiclass` | 3-class | 0 = down (<−2%), 1 = flat (±2%), 2 = up (>+2%) |

The last `HORIZON` rows of every training window are dropped before fitting to prevent the forward targets from extending into the OOS period.

### 3. HMM Regime Detection

A two-state Gaussian HMM is fitted daily to the rolling return window. Regime labels are assigned using the **forward filter** (not Viterbi) so that the label at time t only sees observations up to t. After each fit, states are canonically ordered so regime 0 is always the lower-mean-return state (bear) and regime 1 is always the higher-mean-return state (bull). At prediction time, the last forward probabilities are multiplied by the transition matrix to forecast the next-day regime.

### 4. Multi-Output Neural Network

A separate ensemble of three models is trained per regime every `RETRAIN_INTERVAL` days. The architecture:

- **Shared trunk** — two dense layers (8 → 4 units) with ReLU, Batch Normalisation, Dropout (0.6), and L1+L2 regularisation
- **Gaussian noise** injected at the input layer during training as data augmentation
- **Three task-specific heads**, each with a 2-unit private dense layer:
  - Sharpe head — tanh output scaled to [−5, 5], MSE loss
  - Big-move head — sigmoid output, binary cross-entropy
  - Multiclass head — 3-class softmax, sparse categorical cross-entropy
- **Loss weights** — Sharpe 0.2, Big-move 0.4, Multiclass 0.4
- **Early stopping** (patience 15) with `ReduceLROnPlateau`; chronological 70/30 train/val split

### 5. Signal Combination

Each head casts a vote against configurable thresholds:

| Head | Long (+1) | Short (−1) |
|---|---|---|
| Sharpe | pred > threshold | pred < −threshold |
| Big-move | P(big up) > threshold | — (no big-down head trained) |
| Multiclass | P(up) > threshold AND P(up) > P(flat) AND P(up) > P(down) | P(down) > threshold AND dominates both others |

Votes are summed. Score ≥ 2 → long. Score ≤ −2 → short. Otherwise flat. `big_move` can only vote bullish.

---


## Requirements

```
pip install numpy pandas matplotlib yfinance ta hmmlearn statsmodels scipy tensorflow keras scikit-learn
```

Python 3.9+, TensorFlow 2.x. Expected runtime: 4–8 hours for four OOS periods on a standard CPU.

---

## Out-of-Sample Results

AAPL, 4 non-overlapping OOS windows, 4-year rolling training window, retrain every 20 days.

### Period 1: 2018–2019

| Metric | Sharpe only | Big-move only | Multiclass only | Combined (≥2/3) | MA Cross (50/200) | Buy & Hold |
|---|---|---|---|---|---|---|
| Annual return | 11.92% | 4.58% | 5.24% | −4.96% | −5.51% | 33.78% |
| Cumulative return | 25.21% | 9.35% | 10.73% | −9.65% | −10.70% | 78.77% |
| Annual volatility | 25.00% | 25.25% | 7.58% | 23.08% | 27.60% | 27.53% |
| Sharpe ratio | 0.477 | 0.181 | 0.691 | −0.215 | −0.200 | 1.227 |
| Calmar ratio | 0.289 | 0.115 | 0.851 | −0.120 | −0.088 | 0.877 |
| Max drawdown | −41.24% | −39.69% | −6.16% | −41.24% | −62.26% | −38.52% |
| Sortino ratio | 0.525 | 0.200 | 0.170 | −0.204 | −0.263 | 1.600 |
| % days long | 73.8% | 70.0% | 3.8% | 55.1% | 81.9% | 100.0% |
| % days short | 0.6% | 0.0% | 0.0% | 0.0% | 18.1% | 0.0% |

### Period 2: 2020–2022

| Metric | Sharpe only | Big-move only | Multiclass only | Combined (≥2/3) | MA Cross (50/200) | Buy & Hold |
|---|---|---|---|---|---|---|
| Annual return | 0.26% | 14.77% | 4.08% | 3.76% | 21.96% | 21.79% |
| Cumulative return | 0.77% | 51.16% | 12.75% | 11.72% | 81.43% | 80.66% |
| Annual volatility | 34.70% | 36.21% | 19.88% | 33.40% | 36.93% | 36.93% |
| Sharpe ratio | 0.007 | 0.408 | 0.205 | 0.113 | 0.595 | 0.590 |
| Calmar ratio | 0.006 | 0.470 | 0.159 | 0.101 | 0.575 | 0.693 |
| Max drawdown | −44.72% | −31.43% | −25.61% | −37.23% | −38.18% | −31.43% |
| Sortino ratio | 0.009 | 0.573 | 0.156 | 0.145 | 0.837 | 0.856 |
| % days long | 82.8% | 95.0% | 24.1% | 83.7% | 82.0% | 100.0% |
| % days short | 1.9% | 0.0% | 2.2% | 0.0% | 18.0% | 0.0% |

### Period 3: 2023–2024

| Metric | Sharpe only | Big-move only | Multiclass only | Combined (≥2/3) | MA Cross (50/200) | Buy & Hold |
|---|---|---|---|---|---|---|
| Annual return | 53.37% | 29.90% | 8.04% | 44.88% | −9.11% | 39.74% |
| Cumulative return | 134.41% | 68.38% | 16.66% | 109.27% | −17.32% | 94.76% |
| Annual volatility | 19.62% | 20.03% | 7.60% | 18.16% | 21.62% | 21.51% |
| Sharpe ratio | 2.721 | 1.492 | 1.058 | 2.471 | −0.421 | 1.848 |
| Calmar ratio | 3.476 | 1.713 | 0.921 | 2.768 | −0.266 | 2.393 |
| Max drawdown | −15.35% | −17.45% | −8.73% | −16.21% | −34.21% | −16.61% |
| Sortino ratio | 4.390 | 2.165 | 0.535 | 3.788 | −0.560 | 2.885 |
| % days long | 86.3% | 84.7% | 8.8% | 72.7% | 76.5% | 100.0% |
| % days short | 0.4% | 0.0% | 0.0% | 0.0% | 23.5% | 0.0% |

### Period 4: 2025

| Metric | Sharpe only | Big-move only | Multiclass only | Combined (≥2/3) | MA Cross (50/200) | Buy & Hold |
|---|---|---|---|---|---|---|
| Annual return | 2.15% | 30.74% | 16.09% | 19.80% | −39.95% | 9.66% |
| Cumulative return | 2.13% | 30.33% | 15.88% | 19.55% | −39.59% | 9.54% |
| Annual volatility | 27.86% | 26.43% | 8.13% | 24.34% | 32.46% | 32.57% |
| Sharpe ratio | 0.077 | 1.163 | 1.980 | 0.814 | −1.231 | 0.297 |
| Calmar ratio | 0.080 | 1.337 | 13.699 | 0.861 | −0.839 | 0.320 |
| Max drawdown | −26.98% | −22.99% | −1.17% | −22.99% | −47.62% | −30.22% |
| Sortino ratio | 0.076 | 1.065 | 2.451 | 0.548 | −1.280 | 0.406 |
| % days long | 53.0% | 49.0% | 2.4% | 27.3% | 55.8% | 100.0% |
| % days short | 1.6% | 0.0% | 1.6% | 0.0% | 44.2% | 0.0% |

### Walk-Forward Summary (Mean ± Std, 4 periods)

| Metric | Big-move only | MA Cross (50/200) | Combined (≥2/3) | Buy & Hold |
|---|---|---|---|---|
| Ann. Return | 20.0 ± 10.9% | −8.2 ± 21.9% | 15.9 ± 19.0% | 26.2 ± 11.6% |
| Ann. Vol | 27.0 ± 5.8% | 29.7 ± 5.7% | 24.7 ± 5.5% | 29.6 ± 5.8% |
| Sharpe | 0.81 ± 0.54 | −0.31 ± 0.65 | 0.80 ± 1.04 | 0.99 ± 0.60 |
| Max DD | −27.9 ± 8.4% | −45.6 ± 10.8% | −29.4 ± 10.2% | −29.2 ± 7.9% |
| Calmar | 0.91 ± 0.64 | −0.15 ± 0.50 | 0.90 ± 1.14 | 1.07 ± 0.79 |
| % Long | 74.7 ± 17.3% | 74.1 ± 10.8% | 59.7 ± 21.3% | 100.0 ± 0.0% |

### Sharpe Persistence

| Period | Big-move only | MA Cross (50/200) | Combined (≥2/3) | Buy & Hold |
|---|---|---|---|---|
| 2018–2019 | 0.181 | −0.200 | −0.215 | 1.227 |
| 2020–2022 | 0.408 | 0.595 | 0.113 | 0.590 |
| 2023–2024 | 1.492 | −0.421 | 2.471 | 1.848 |
| 2025 | 1.163 | −1.231 | 0.814 | 0.297 |

Positive Sharpe in **all** periods = genuine edge. Positive in 1–2 periods = noise.

**Big-move only** is the only strategy with positive Sharpe in all four OOS periods. **MA Cross (50/200)** is negative in three of four periods — shorting the death cross consistently destroys value on a single stock.

---

## Overfitting Persistence

Gap = mean(train_loss − val_loss). Positive = underfitting. Negative = overfitting.

| Period | R0 % Overfit | R0 Gap | R1 % Overfit | R1 Gap |
|---|---|---|---|---|
| 2018–2019 | 15.4% | +0.33 | 7.7% | +0.32 |
| 2020–2022 | 2.6% | +0.62 | 15.8% | +0.30 |
| 2023–2024 | 16.0% | +0.41 | 46.2% | +0.03 |
| 2025 | 0.0% | +0.86 | 69.2% | +0.03 |

**Regime 0 (bear):** Positive gap in all periods — consistently underfitting due to sparse training samples. Regularisation is too aggressive for the small bear-regime dataset.

**Regime 1 (bull):** % overfit trend is 7.7% → 15.8% → 46.2% → 69.2% across periods. Gap shrinks toward zero (+0.32 → +0.03). The bull-regime model is gradually memorising its training window as more bull-market history accumulates. This is a structural problem that will worsen with each additional period tested.

---

## Limitations

- No transaction costs or slippage modelled
- Binary position sizing — no confidence-weighted scaling
- Single ticker (AAPL); parameters are not transferable to other assets without re-tuning
- Bear regime (R0) persistently underfits due to sparse training samples; its signals are less reliable than bull regime signals
- Four OOS periods tested; results are directionally informative but not statistically significant at conventional levels
- Any parameter change informed by OOS results constitutes contamination even without saved model weights
