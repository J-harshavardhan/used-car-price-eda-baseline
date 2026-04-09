# Used Car Price EDA & Baseline Model

## Overview
Exploratory data analysis, systematic data cleaning, and baseline MAE evaluation on a messy used car dataset as part of the IIT Patna AIML Mini Project.

---

## Tasks

### Task 1 — Explore and Identify Issues
Loaded the dataset and used `df.info()`, `df.describe()`, and `df.shape` to identify **9 data quality problems**:

| # | Column | Problem |
|---|--------|---------|
| 1 | `selling_price` | 158 null rows — target column cannot have nulls |
| 2 | `mileage` | Object dtype — some values contain `"kmpl"` unit text |
| 3 | `km_driven` | Object dtype — values like `"42000 kms"`, `"7,369"` with commas |
| 4 | `brand` | Leading/trailing whitespace and inconsistent casing |
| 5 | `seller_type` | Typos and mixed casing — `"Delaer"`, `"Individuall"`, `"DEALER"` |
| 6 | `selling_price` | Corrupt outliers — values like ₹999,999,999 and ₹500 |
| 7 | `seats` | Impossible value of 0 in 2 rows |
| 8 | `engine` / `max_power` / `seats` | Heavy missing values requiring imputation |
| 9 | Rows | 169 exact duplicate rows |

---

### Task 2 — Clean the Data

| Step | Action |
|------|--------|
| 1 | Dropped `Unnamed: 0` index column |
| 2 | Dropped 158 rows with null `selling_price` |
| 3 | Removed 15 corrupt outlier price rows |
| 4 | Stripped whitespace and lowercased `brand` column |
| 5 | Standardised `seller_type` (fixed typos and casing) |
| 6 | Extracted digits from `km_driven` — removed commas and `"kms"` text |
| 7 | Extracted numeric value from `mileage` — removed `"kmpl"` unit |
| 8 | Replaced impossible `seats == 0` with NaN |
| 9 | Median-imputed all remaining nulls (`mileage`, `engine`, `max_power`, `seats`) |
| 10 | Removed 281 duplicate rows |

**Final clean shape: 15,207 rows × 13 columns — zero nulls remaining**

---

### Task 3 — Baseline MAE

Built a mean predictor baseline that predicts the mean `selling_price` for every record.

```
Mean Selling Price : ₹ 7,68,422
Baseline MAE       : ₹ 4,43,825
```

Any trained ML model must score **below ₹4,43,825 MAE** to be considered useful.

---

## Dataset
`used_cars_messy.csv` — 15,661 rows × 14 columns (raw)

## Files

| File | Description |
|------|-------------|
| `used_car_price_data_baseline.ipynb` | Main notebook with all 3 tasks |
| `used_cars_messy.csv` | Raw messy dataset |
| `baseline_mae_plot.png` | Actual vs baseline visualisation |

---

## Tech Stack
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)
![NumPy](https://img.shields.io/badge/NumPy-1.24-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-red)

---

## Author
**Harshavardhan** | IIT Patna AIML Program | Masai School
