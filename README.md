# JPMorgan Quantitative Research — Virtual Job Simulation

A four-task quantitative research project covering commodity pricing and credit risk modelling, completed as part of the JPMorgan Chase Quantitative Research virtual experience on [Forage](https://www.theforage.com/).

## Overview

| Task | Domain | Deliverable |
|------|--------|-------------|
| 1 | Commodity pricing | Natural gas price forecasting |
| 2 | Commodity pricing | Storage contract valuation model |
| 3 | Credit risk | Probability of default & expected loss model |
| 4 | Credit risk | FICO score quantization |

## Tasks

### Task 1 — Natural Gas Price Forecasting
Fitted a seasonal-trend time series model on monthly natural gas prices, interpolated to daily granularity, and produced forward price projections.

- **Input**: [`Nat_Gas.csv`](./Nat_Gas.csv) — monthly historical prices
- **Notebook**: [`Task 1.ipynb`](./Task%201.ipynb)
- **Output**: [`Nat_Gas_Predicted.csv`](./Nat_Gas_Predicted.csv) — daily interpolated + forecast

### Task 2 — Commodity Storage Contract Valuation
Built a prototype pricing model for commodity storage contracts, factoring in injection costs, withdrawal costs, storage fees, and forward curves.

- **Notebook**: [`Task 2.ipynb`](./Task%202.ipynb)
- **Module**: [`Task 2 Example.py`](./Task%202%20Example.py)

### Task 3 — Personal Loan Default Prediction
Implemented logistic regression and decision tree classifiers **from scratch in NumPy** (no scikit-learn) to predict probability of default (PD) on a 10,000-loan portfolio.

Expected loss is derived as:

```
EL = PD × (1 − Recovery Rate) × Exposure
```

with a recovery rate of 10%.

- **Input**: [`Task 3 and 4_Loan_Data.csv`](./Task%203%20and%204_Loan_Data.csv)
- **Notebook**: [`Task 3.ipynb`](./Task%203.ipynb)
- **Evaluation**: stratified train/val/test split, ROC-AUC

### Task 4 — FICO Score Quantization
Designed an optimal FICO score bucketing algorithm to convert continuous credit scores into discrete ratings for a categorical-input mortgage default model.

Two methods implemented and compared:

| Method | Objective | Algorithm | Complexity |
|--------|-----------|-----------|------------|
| MSE | Minimise within-bucket variance | K-means (1D) | O(N · K · iter) |
| Log-likelihood | Maximise Σ [kᵢ ln pᵢ + (nᵢ − kᵢ) ln(1 − pᵢ)] | Dynamic programming | O(K · N²) |

The DP approach guarantees the globally optimal partition. Buckets are mapped to ratings where **lower rating = better credit**.

- **Input**: [`Task 3 and 4_Loan_Data.csv`](./Task%203%20and%204_Loan_Data.csv)
- **Notebook**: [`Task 4.ipynb`](./Task%204.ipynb)

## Tech Stack

- **Language**: Python 3
- **Libraries**: NumPy, pandas, matplotlib
- **Models built from scratch**: logistic regression, decision tree, dynamic programming for optimal quantization

## Repository Structure

```
.
├── Nat_Gas.csv                    # Task 1 input
├── Nat_Gas_Predicted.csv          # Task 1 output
├── Task 1.ipynb                   # Task 1 — gas price forecasting
├── Task 2.ipynb                   # Task 2 — storage contract pricing
├── Task 2 Example.py              # Task 2 — pricing module
├── Task 3 and 4_Loan_Data.csv     # Task 3 & 4 input
├── Task 3.ipynb                   # Task 3 — PD & expected loss
├── Task 4.ipynb                   # Task 4 — FICO quantization
└── Certificate.pdf                # Completion certificate
```

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook
```

Open any of the `.ipynb` files and run all cells.
