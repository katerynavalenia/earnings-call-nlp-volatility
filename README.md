# Does Management Uncertainty in Earnings Calls Predict Short-Term Stock Volatility?

**Course:** Applied Big Data Analytics in Finance  
**Hypothesis:** Firms whose management uses more uncertainty-related language experience higher volatility in the days following the earnings call.

---

## Project Structure

```
├── run_pipeline.py                  # Master orchestrator – runs all steps
├── requirements.txt
│
├── data/
│   ├── raw/                         # Loughran-McDonald dictionary (auto-downloaded)
│   ├── processed/                   # Parquet files produced by the pipeline
│   ├── market/                      # Professor's daily market data CSVs
│   ├── marketCap/                   # Professor's market-cap CSVs
│   ├── earning/                     # Professor's earnings-surprise CSVs
│   └── fundamental/                 # (optional) fundamental data
│
├── scrapper/
│   ├── koyfin_scraper.py            # Selenium-based transcript scraper
│   ├── run_scraper.py               # Multi-worker scraper entry point
│   ├── utils.py                     # Helper functions
│   └── output/{year}/               # ~100K scraped transcripts (see Data Access)
│
├── src/
│   ├── preprocessing/
│   │   ├── parse_transcripts.py     # Step 1 – parse .txt → parquet
│   │   └── ticker_mapping.py        # Step 2 – fuzzy-match company→ticker
│   ├── labeling/
│   │   ├── lexicon_labeler.py       # Step 3 – LM Uncertainty word-list scoring
│   │   └── build_training_set.py    # Step 4 – balanced train/val/test sets
│   ├── modeling/
│   │   ├── train_uncertainty_model.py  # Step 5 – fine-tune FinBERT
│   │   ├── evaluate_model.py           # Step 6 – test-set metrics
│   │   └── predict_uncertainty.py      # Step 7 – score all transcripts
│   ├── features/
│   │   ├── compute_volatility.py    # Step 8 – pre/post-call realized vol
│   │   └── build_panel.py           # Step 9 – merge into panel dataset
│   └── analysis/
│       ├── regression.py            # Step 10 – main PanelOLS
│       ├── robustness.py            # Step 11 – robustness checks
│       └── visualizations.py        # Step 12 – all figures
│
├── models/
│   └── uncertainty_finbert/         # Fine-tuned FinBERT (weights via Git LFS)
├── output/
│   ├── figures/                     # PNG plots
│   └── tables/                      # Regression tables, evaluation metrics
└── colab_notebook.ipynb             # Google Colab fallback for GPU training
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place input data

| Data                   | Location                       | Source                                   |
|------------------------|--------------------------------|------------------------------------------|
| Earnings transcripts   | `scrapper/output/{year}/`      | See **Data Access** below (~100K files)  |
| Daily market prices    | `data/market/`                 | Professor's `market.zip` → extract here  |
| Market capitalisation  | `data/marketCap/`              | Professor's `marketCap.zip`              |
| Earnings surprises     | `data/earning/`                | Professor's `earning.zip`                |

### 3. Run the full pipeline

```bash
python run_pipeline.py
```

Or resume from a specific step / skip training:

```bash
python run_pipeline.py --from 7          # resume from step 7
python run_pipeline.py --skip-training   # use existing model in models/
python run_pipeline.py --only 10 11 12   # re-run regressions + plots only
```

---

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `parse_transcripts.py` | Parse ~100K .txt transcripts → structured parquet |
| 2 | `ticker_mapping.py` | Fuzzy-match company names to ticker symbols |
| 3 | `lexicon_labeler.py` | Score sentences with LM Uncertainty word list |
| 4 | `build_training_set.py` | Create balanced train/val/test from lexicon labels |
| 5 | `train_uncertainty_model.py` | Fine-tune ProsusAI/finbert for uncertainty classification |
| 6 | `evaluate_model.py` | Evaluate model on held-out test set |
| 7 | `predict_uncertainty.py` | Score all transcripts with fine-tuned model |
| 8 | `compute_volatility.py` | Compute pre- and post-call realized volatility |
| 9 | `build_panel.py` | Merge uncertainty + volatility + controls → panel |
| 10 | `regression.py` | Main PanelOLS regression with FE and clustered SE |
| 11 | `robustness.py` | Alternative windows, subsamples, lexicon-only |
| 12 | `visualizations.py` | All figures for the report |

---

## Methodology

1. **Lexicon labeling**: Sentences are scored using the Loughran-McDonald Uncertainty word list. Top/bottom quartile sentences form training labels.
2. **FinBERT fine-tuning**: ProsusAI/finbert is fine-tuned on the lexicon-labelled sentences for binary uncertainty classification.
3. **Transcript scoring**: Each transcript gets an uncertainty score (mean predicted probability) and uncertainty share (fraction of sentences classified as uncertain).
4. **Volatility**: Post-call volatility = std(log returns) over [+1, +5] trading days; pre-call volatility = std(log returns) over [-30, -6].
5. **Regression**: `PostCallVol ~ UncertaintyScore + PreCallVol + log(MarketCap) + EarningsSurprise + Sector_FE + YearQuarter_FE`, firm-clustered standard errors.

---

## Notes

- **GPU recommended** for step 5 (FinBERT training). On CPU, expect ~2-3 hours. If using Google Colab, upload `data/processed/train.parquet`, `val.parquet`, `test.parquet` and run `train_uncertainty_model.py` there, then copy the `models/uncertainty_finbert/` folder back.
- The Loughran-McDonald dictionary is auto-downloaded on first run (step 3).
- All intermediate data is stored as Parquet in `data/processed/` for fast I/O.

---

## Data Access

| Asset | Size | Location | How to obtain |
|-------|------|----------|---------------|
| Scraped transcripts (~100K) | ~5.5 GB | `scrapper/output/{year}/` | [Google Drive](<!-- INSERT DRIVE LINK -->) |
| Fine-tuned FinBERT model | ~400 MB | `models/uncertainty_finbert/` | Included via Git LFS (auto-downloaded on `git clone`) |
| Sample transcripts (27) | ~1.2 MB | `data/test_transcripts/` | Included in repo |
| Market data (OHLCV, MarketCap, Earnings) | ~1.6 GB | `data/market/`, `data/marketCap/`, `data/earning/` | Professor-provided zips → extract into respective folders |

> **Note:** After cloning, run `git lfs pull` if the model weights were not downloaded automatically.  
> Replace `<!-- INSERT DRIVE LINK -->` with the actual Google Drive sharing link after uploading the transcripts.
