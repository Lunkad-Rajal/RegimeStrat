# RegimeStrat
Walk-forward trading system combining HMM regime detection with per-regime multi-output neural networks. Predicts 5-day Sharpe ratio, big moves, and direction simultaneously to generate long/flat/short signals on daily data.

# Regime-Adaptive Trading System with Multi-Output Neural Networks

A walk-forward algorithmic trading framework that combines Hidden Markov Model (HMM) regime detection with per-regime multi-output neural networks to generate daily trading signals. Also involves three simultaneous prediction targets, some overfitting guardrails, and a bunch of diagnostics I found useful along the way.

---

# Features

- **Three prediction targets trained simultaneously** — forward Sharpe ratio (regression), big up move >2% (binary classification), and directional outcome down/flat/up (ternary classification)
- **Separate neural network per HMM regime** — the model behaves differently depending on whether the market looks calm or stressed
- **Walk-forward validation** — signals are generated strictly out-of-sample using a rolling training window; no data leakage
- **Periodic retraining** — models retrain every N days (configurable) and are cached between cycles so you're not waiting forever
- **Combined voting signal** — the three head outputs get aggregated into a single long / flat / short signal using configurable thresholds
- **Overfitting diagnostics built in** — training vs. validation loss is tracked across every retraining window
- **Seed sensitivity test** — the first prediction window reruns under multiple random seeds so you can see how stable the signals actually are
- **No third-party backtesting dependencies** — all performance metrics computed from scratch

---

## How It Works

### 1. Data and Feature Engineering

Daily OHLCV data is pulled via `yfinance`. The full technical indicator suite from the `ta` library gets applied — momentum, trend, volatility, volume. Each indicator is then run through an Augmented Dickey-Fuller (ADF) test at the 5% level. Anything non-stationary gets converted to percentage changes. Indicators with invalid test statistics or fewer than 20 observations are dropped.

### 2. Target Construction

Three forward-looking targets are computed over a 5-day horizon:

| Target | Type | Description |
|---|---|---|
| `target_sharpe` | Regression | 5-day forward return divided by 5-day forward volatility |
| `target_big_up` | Binary classification | 1 if 5-day forward return exceeds +2%, else 0 |
| `target_multiclass` | Ternary classification | 0 = down (<−1%), 1 = flat (±1%), 2 = up (>+1%) |

### 3. HMM Regime Detection

A two-state Gaussian HMM is fitted to the daily return series within each rolling training window. Training observations get labelled with a regime (0 or 1), which loosely maps to different volatility or trend conditions. At prediction time, the HMM transition matrix estimates the probability of each regime on the next trading day, and whichever regime comes out on top determines which network runs inference.

### 4. Multi-Output Neural Network

A separate network is trained per regime. The architecture:

- **Shared trunk** — two dense layers (32 → 16 units) with ReLU, dropout (0.4), and L2 regularisation (1e-3)
- **Three task-specific heads**, each with a small dense layer (4 units) up front:
  - Sharpe head: linear output, MSE loss
  - Big Move head: sigmoid output, binary cross-entropy
  - Multi-Class head: softmax output (3 classes), sparse categorical cross-entropy
- **Combined loss weights**: Sharpe 0.3, Big Move 0.3, Multi-Class 0.4

### 5. Signal Combination

Each head casts a directional vote against configurable thresholds:

| Head | Long (+1) condition | Short (−1) condition |
|---|---|---|
| Sharpe | Predicted Sharpe > 0.1 | Predicted Sharpe < −0.1 |
| Big Move | P(big up) > 0.3 | — |
| Multi-Class | P(up) > 0.3 and P(up) > P(down) | P(down) > 0.3 and P(down) > P(up) |

Votes are summed. Positive total → long (+1), negative → short (−1), zero → flat (0).

---

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- yfinance
- ta
- hmmlearn
- statsmodels
- tensorflow (2.x)
- keras
- scikit-learn

---
