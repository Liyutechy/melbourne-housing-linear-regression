# Melbourne Housing Price Prediction – Linear Regression

Simple linear regression model trained on the classic **Melbourne Housing Market** dataset to predict house prices.

## Results

| Metric | Value |
|--------|-------|
| **MAE** | ≈ $296,711 |
| **RMSE** | ≈ $442,330 |
| **R²** | ≈ 0.507 |

### Features used
- Rooms
- Bathroom
- Landsize
- BuildingArea
- YearBuilt
- Lattitude
- Longtitude
- Distance
- Car

Missing values were handled with **median imputation**.

## How to run

```bash
# 1. Download the dataset (melb_data.csv)
#    You can get it from the original Melbourne housing dataset
#    or place the file in this directory.

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train_model.py
```

## Dataset

`melb_data.csv` – Melbourne housing data (13,580 rows).

Place the CSV file in the same folder as `train_model.py` before running.

Source: Public Melbourne housing market data (commonly used on Kaggle).

## Project structure

```
melbourne-housing-linear-regression/
├── melb_data.csv          # (add this file)
├── train_model.py
├── requirements.txt
└── README.md
```

## Author
Created with Grok
