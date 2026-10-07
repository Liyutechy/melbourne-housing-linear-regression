# Melbourne Housing Price Prediction – Linear Regression + A/B Testing

End-to-end project that trains a Linear Regression model on the Melbourne Housing dataset **and** runs a proper A/B test comparing two feature sets with statistical significance testing.

---

## 1. Linear Regression Model

| Metric | Value |
|--------|-------|
| **MAE** | ≈ $296,711 |
| **RMSE** | ≈ $442,330 |
| **R²** | ≈ 0.507 |

**Features used:** Rooms, Bathroom, Landsize, BuildingArea, YearBuilt, Lattitude, Longtitude, Distance, Car  
Missing values handled with median imputation.

```bash
pip install -r requirements.txt
python train_model.py
```

---

## 2. A/B Test

We compared two versions of the model:

| Group | Features | MAE |
|-------|----------|-----|
| **A (Control)** | Rooms, Bathroom, Distance, Car | **$346,837** |
| **B (Treatment)** | + Landsize, BuildingArea, YearBuilt, Lat, Long | **$297,172** |

### Results
- **Improvement**: Version B reduced MAE by **14.3%** ($49,665 absolute)
- **p-value**: `0.000000` (highly significant)
- **t-statistic**: 14.08
- **95% CI of difference**: [$42,751 – $56,579]
- **Sample size**: 3,395 properties

**Conclusion**: Version B is statistically significantly better (p < 0.05). We reject the null hypothesis.

```bash
python ab_test.py
```

---

## Project Structure

```
melbourne-housing-linear-regression/
├── train_model.py      # Train & evaluate Linear Regression
├── ab_test.py          # A/B test with statistical significance
├── requirements.txt
├── README.md
└── melb_data.csv       # (add this file locally)
```

## How to run everything

1. Place `melb_data.csv` in the project root
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train_model.py
   ```
4. Run the A/B test:
   ```bash
   python ab_test.py
   ```

## Dataset
Melbourne Housing Market data (13,580 rows). Commonly used on Kaggle.

---

Created with Grok
