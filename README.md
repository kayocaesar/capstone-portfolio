# Sentiment & Macro-Augmented S&P 500 Long-Short Portfolio Allocation Capstone Project

> This repository contains a source code for a capstone project. The work augments technical features with pre-computed news sentiment scores and configurable macroeconomic indicators for a deep learning portfolio allocation framework, evaluated on S&P 500 stocks over 2017–2023.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Requirements](#requirements)
4. [Data](#data)
5. [Configuration](#configuration)
6. [Feature Layers](#feature-layers)
7. [Models](#models)
8. [Portfolio Strategy](#portfolio-strategy)
9. [Running the Notebook](#running-the-notebook)
10. [Extending the Notebook](#extending-the-notebook)
11. [Known Limitations](#known-limitations)
12. [Citation](#citation)
13. [Licence](#licence)

---

## Overview

In this project, a deep learning portfolio allocation framework, based on historical trading data, is developed by layering two additional information sources on top of the original technical features:

- **Sentiment features** — daily per-ticker sentiment scores aggregated from the Yahoo Finance financial news dataset (Drinkall et al., 2025), covering 2017–2023.
- **Macroeconomic features** — configurable market-wide indicators (S&P 500 index, VIX, 10-year Treasury yield, US Dollar Index) downloaded automatically at runtime.

Four deep learning architectures (MLP, CNN, LSTM, Transformer) are trained on four feature sets and evaluated using a daily equal-weight long-short strategy. The experiment measures whether richer features improve total return, Sharpe ratio, and maximum drawdown.

| Property | Value |
|---|---|
| Universe | S&P 500 (10 stocks) |
| Training period | 2017-01-01 – 2021-12-31 |
| Test period | 2022-01-01 – 2023-12-31 |
| Price download start | 2016-07-01 (warm-up buffer) |
| Models | MLP, CNN, LSTM, Transformer |
| Feature layers | Technical, Sentiment, Macro |
| Strategy | Equal-weight long-short, daily rebalance |

---

## Repository Structure

```
.
├── main.ipynb   # Main notebook
├── data/
│   └── YYYY_processed.json.xz        # Sentiment input files (one per year) [excluded here, but follow the link at 4. Data if you are replicating the work]
├── requirements.txt                  # requirements
└── README.md                         # This file
```

The notebook is self-contained. All outputs (figures, tables, summary statistics) are generated inline when cells are run top to bottom.

---

## Requirements

### Python packages

Install all dependencies with the first notebook cell, or manually:

```bash
pip install yfinance torch scikit-learn pandas numpy matplotlib seaborn tqdm
```

| Package | Version tested | Purpose |
|---|---|---|
| `torch` | ≥ 1.13 | MLP / CNN / LSTM / Transformer models |
| `yfinance` | ≥ 0.2 | Stock price and macro data download |
| `scikit-learn` | ≥ 1.0 | StandardScaler, train/test split |
| `pandas` | ≥ 1.5 | Data wrangling and merging |
| `numpy` | ≥ 1.23 | Array operations |
| `matplotlib` | ≥ 3.5 | Charts and visualisations |
| `seaborn` | ≥ 0.12 | Heatmap visualisations |
| `tqdm` | any | Progress bars |

### Google Colab

The notebook is designed to run on Google Colab with data stored in Google Drive. The Drive mount cell is included and guarded so it is skipped automatically outside Colab.

> **Note:** All packages except `torch` are pre-installed on Colab. Only `torch` may require an explicit install if a custom runtime is used.

---

## Data

### Sentiment files

The notebook expects pre-processed yearly JSON files produced by the [`financial-news-dataset`](https://github.com/FelixDrinkall/when_dimensionality_hurts) research repository (Drinkall et al., 2025) and found at ['data'](https://github.com/FelixDrinkall/financial-news-dataset/tree/main/data) (Drinkall et al., 2025).

| Setting | Details |
|---|---|
| File naming | `YYYY_processed.json.xz` (plain `.json` also supported) |
| Location | Set `SENTIMENT_DATA_DIR` in Section 1 of the notebook |
| Compression | LZMA (`.xz`). Decompressed automatically via Python's `lzma` module. |
| Coverage | 2017–2023 (must overlap the training window) |

**Required fields per article record:**

| Field | Type | Description |
|---|---|---|
| `date_publish` | string | Publication timestamp (UTC) |
| `mentioned_companies` | list | Stock ticker symbols mentioned in the article |
| `sentiment` | dict | Keys: `negative`, `neutral`, `positive` |
| `emotion` | dict | Keys: `neutral`, `joy`, `anger`, `sadness`, `fear`, `surprise`, `disgust` (optional) |

> **Note:** Records missing `date_publish`, `sentiment`, or `mentioned_companies` are silently dropped. Trading days with no article coverage for a ticker are forward-filled with the most recent known score, then set to a neutral default (`sent_neutral = 1`, all others `0`) at the very start of the series.

### Stock price data

OHLCV data is downloaded automatically at runtime from Yahoo Finance via `yfinance`. No manual download is required.

- Tickers are set in `SP500_TICKERS` in Section 1.
- Data is downloaded from `PRICE_DOWNLOAD_START` (`2016-07-01`) to `TEST_END` to allow a warm-up buffer for rolling indicators.
- Rows before `TRAIN_START` are dropped before any model sees the data.

### Macroeconomic data

Macro indicators are downloaded automatically from Yahoo Finance. The set of series is controlled by the `MACRO_SERIES` list in Section 1.

| Ticker | Description |
|---|---|
| `^GSPC` | S&P 500 index — broad market return and volatility |
| `^VIX` | CBOE Volatility Index — market fear gauge (level used directly) |
| `^TNX` | 10-year US Treasury yield — rate environment (level used directly) |
| `DX-Y.NYB` | US Dollar Index — currency pressure |

---

## Configuration

All user-configurable parameters live in **Section 1** of the notebook. No other cell needs to be edited for standard use.

| Parameter | Default | Description |
|---|---|---|
| `SENTIMENT_DATA_DIR` | `"./data"` | Path to folder containing `YYYY_processed.json.xz` files |
| `SP500_TICKERS` | 10 tickers | List of S&P 500 stock tickers to include |
| `PRICE_DOWNLOAD_START` | `"2016-07-01"` | Start of price download (6-month warm-up before training) |
| `TRAIN_START` | `"2017-01-01"` | First date used in training samples |
| `TRAIN_END` | `"2021-12-31"` | Last date used in training samples |
| `TEST_START` | `"2022-01-01"` | First date used in test samples |
| `TEST_END` | `"2023-12-31"` | Last date used in test samples |
| `MACRO_SERIES` | 4 series | List of `(yfinance_ticker, prefix)` tuples |
| `LOOKBACK` | `10` | Number of history days per input window |
| `EPOCHS` | `50` | Training epochs per model |
| `BATCH_SIZE` | `64` | Mini-batch size |
| `LEARNING_RATE` | `1e-3` | Adam optimiser learning rate |
| `HIDDEN_SIZE` | `64` | Hidden units for all models |
| `DROPOUT` | `0.2` | Dropout rate applied in all models |
| `SEED` | `42` | Random seed for reproducibility |

---

## Feature Layers

The notebook composes four feature sets and evaluates each independently, making it straightforward to isolate the contribution of each information source.

| Feature set | Columns included | Purpose |
|---|---|---|
| Base | `return`, `rsi`, `volume`, `volatility` | Replicates the original paper baseline |
| + Sentiment | Base + `sent_*`, `emotion_*` | Adds news sentiment and emotion scores |
| + Macro | Base + `{prefix}_ret/vol/lvl` per series | Adds macroeconomic context |
| Full | All of the above | Combined feature set |

### Technical features

| Feature | Formula / Description |
|---|---|
| `return` | `(P_t − P_{t−1}) / P_{t−1}` — daily percentage price change |
| `rsi` | 14-day Relative Strength Index (range 0–100) |
| `volume` | Raw daily trading volume |
| `volatility` | 5-day rolling standard deviation of `return` |

### Sentiment features

One value per ticker per trading day, computed as the mean across all articles mentioning that ticker on that date.

- `sent_negative`, `sent_neutral`, `sent_positive` — FinBERT-style sentiment probabilities
- `emotion_neutral`, `emotion_joy`, `emotion_anger`, `emotion_sadness`, `emotion_fear`, `emotion_surprise`, `emotion_disgust` — included when the `emotion` field is present in the JSON records

### Macro features

Each series in `MACRO_SERIES` produces three derived columns:

| Suffix | Description |
|---|---|
| `{prefix}_ret` | Daily return of the index or price series |
| `{prefix}_vol` | 5-day rolling volatility of the return |
| `{prefix}_lvl` | Raw closing level (especially informative for VIX and yield series) |

> **Tip:** To restrict which macro columns reach the models, edit `MACRO_FEATURE_COLS` after Section 3 runs. Comment out any column name to exclude it from all subsequent feature sets.

---

## Models

All four architectures are implemented in PyTorch and share the same training loop, loss function (MSE on next-day return), and optimiser (Adam with `ReduceLROnPlateau` scheduling). Input shape is `(batch, lookback, n_features)`.

| Model | Architecture |
|---|---|
| MLP | Flatten → Linear(2H) → ReLU → Dropout → Linear(H) → ReLU → Dropout → Linear(1) |
| CNN | Conv1d ×2 (kernel 3, padding 1) → AdaptiveAvgPool → Linear → ReLU → Dropout → Linear(1) |
| LSTM | 2-layer LSTM (hidden H, dropout) → last hidden state → Linear(1) |
| Transformer | Linear embed → 2-layer TransformerEncoder (4 heads, FFN 4H) → last token → Linear(1) |

Gradient clipping (max norm 1.0) is applied at every training step. The `StandardScaler` is fitted on training data only to prevent look-ahead bias.

---

## Portfolio Strategy

The strategy :

- At the end of each trading day, each model predicts the next-day return for every ticker.
- Tickers with a **positive** predicted return enter the **long** leg; **negative** predictions go **short**.
- All positions receive equal weight: `w = 1 / n_total` where `n_total = n_long + n_short`.
- The portfolio is fully invested and rebalanced daily. Transaction costs are assumed negligible.

**Daily portfolio return:**

$$R_{t+1} = \sum_{i \in \text{long}} w_i r_i - \sum_{i \in \text{short}} w_i r_i$$

### Performance metrics

| Metric | Definition |
|---|---|
| Total return | Cumulative product of `(1 + R_t) − 1` over the test period |
| Sharpe ratio | Annualised: `(μ_excess / σ_excess) × √252`, risk-free rate = 0 |
| Max drawdown | `abs(min drawdown)` — largest peak-to-trough loss in the test period |

---

## Running the Notebook

### Quick start

1. Download your pre-processed sentiment data from ['data'](https://github.com/FelixDrinkall/financial-news-dataset/tree/main/data)
2. Upload your `YYYY_processed.json.xz` files to Google Drive (or a local folder).
3. Open `main.ipynb` in Google Colab or JupyterLab. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kayocaesar/capstone-portfolio/blob/main/main.ipynb)
4. In **Section 1**, update `SENTIMENT_DATA_DIR` to point to your data folder.
5. Optionally adjust `SP500_TICKERS`, `MACRO_SERIES`, and any hyperparameters.
6. Run all cells: **Runtime → Run all**.

### Expected runtime

| Environment | Approximate duration |
|---|---|
| Colab T4 GPU | ~25–40 minutes (full experiment + ablation) |
| Colab CPU only | ~90–120 minutes |
| Local GPU | Depends on hardware |

> **Note:** The ablation study (Section 13) trains 16 additional model instances. Comment it out to reduce total runtime by roughly 40%.

### Outputs produced

| Section | Output |
|---|---|
| §10 | Full comparison table: 4 feature sets × 4 models × 3 metrics |
| §12a | Cumulative return curves (one panel per model, four lines per panel) |
| §12b | Heatmaps of each metric across feature sets and models |
| §12c | Incremental lift bar charts versus the technical baseline |
| §13 | Ablation bar chart: Sharpe ratio by feature set and model |
| §14 | Printed summary of improvement counts and best configuration |

---

## Extending the Notebook

### Adding more stocks

Edit `SP500_TICKERS` in Section 1. Ensure the new tickers appear in `mentioned_companies` in your sentiment files for meaningful coverage.

### Adding macro series

Append a tuple to `MACRO_SERIES` in Section 1:

```python
MACRO_SERIES = [
    ("^GSPC",   "sp500"),
    ("^VIX",    "vix"),
    ("^TNX",    "t10y"),
    ("DX-Y.NYB","usd"),
    # Add your own:
    ("GC=F",    "gold"),    # Gold futures
    ("CL=F",    "oil"),     # WTI Crude
    ("^IXIC",   "nasdaq"),  # NASDAQ Composite
]
```

Any `yfinance`-compatible ticker can be used. The three derived columns (`_ret`, `_vol`, `_lvl`) are generated automatically.

### Changing the lookback window

Change `LOOKBACK` in Section 1. Larger windows give sequence models more context but increase memory usage and slow training. CNN and MLP are less sensitive to this parameter than LSTM and Transformer.

### Adjusting the train/test split

Update `TRAIN_END` and `TEST_START` in Section 1. The split must remain within the 2017–2023 sentiment window. Ensure at least one full year of test data for reliable metric estimates.

---

## Known Limitations

- **Transaction costs** are assumed to be zero. Real-world returns would be lower due to bid-ask spread and market impact.
- **Small universe** — the experiment uses 10 stocks per run. Results may not generalise to larger universes without architectural changes.
- **Sentiment coverage** varies across tickers. Low-coverage tickers rely heavily on forward-filled scores, which may not reflect actual market conditions.
- **Sample bias** — the fixed ticker selection introduces bias. Results should be interpreted as illustrative rather than definitive.
- **Data availability** — macro indicators are downloaded at execution time. Historical series may be revised or discontinued by Yahoo Finance.
- **Transformer head constraint** — the Transformer uses 4 attention heads. `HIDDEN_SIZE` must be divisible by 4.

---


---

## Licence

The code in this repository is released under the **MIT Licence**.

The sentiment dataset used for this work is licensed under **CC BY-NC-SA 4.0** (non-commercial use only). See the [dataset repository](https://github.com/FelixDrinkall/when_dimensionality_hurts) for full terms.

Stock price data is retrieved from Yahoo Finance and subject to Yahoo's terms of service.
