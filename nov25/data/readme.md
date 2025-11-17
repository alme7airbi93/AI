Here’s a **simple beginner-friendly `README_Data.md`** (you can rename it to just `README.md` inside your `/data` folder).
It clearly explains what each dataset is for, how to use it in Google Colab, and any credits or notes.

---

# 📊 Dataset Guide — AI Technical Training (Days 1 – 5)

Welcome!
This folder contains all datasets used during the **AI Technical Training (Days 1–5)**.
Each file supports a different hands-on exercise in your Colab notebooks.

---

## 📁 Folder Structure

```
data/
│
├── day1_sample_data.csv
├── day2_sales_data.csv
├── day3_powerplant_data.csv
├── day4_images.zip
├── day5_predictions_sample.json
└── README_Data.md
```

---

## 🧠 Overview of Each Dataset

### 🗓 **Day 1 – Python & AI Foundations**

**File:** `day1_sample_data.csv`
**Description:** A small toy dataset (students + scores + attendance) used to learn how to:

* Read CSV files
* Perform basic operations with `pandas`
* Compute statistics and visualize results

**Example Code:**

```python
import pandas as pd
df = pd.read_csv('/content/data/day1_sample_data.csv')
df.head()
```

---

### 📈 **Day 2 – Exploratory Data Analysis**

**File:** `day2_sales_data.csv`
**Description:** Sales transactions (date, region, product, revenue).
Used for cleaning, profiling, detecting outliers, and creating visual insights.

**Skills Practiced:**
Data cleaning | Descriptive stats | Matplotlib & Seaborn visualization

---

### ⚙️ **Day 3 – Machine Learning Models**

**File:** `day3_powerplant_data.csv`
**Description:** Realistic numeric dataset (temperature, pressure, humidity, etc.) for predicting **power output**.
Used to train regression models such as Linear Regression, Random Forest, and XGBoost.

**Skills Practiced:**
Feature engineering | Model training | Evaluation (MAE, RMSE)

---

### 🧠 **Day 4 – Deep Learning Intro**

**File:** `day4_images.zip`
**Description:** A small image folder containing 2 categories (e.g., *cats vs dogs*).
Used to build and train a simple **neural network / CNN**.

**How to Use in Colab:**

```python
from zipfile import ZipFile
with ZipFile('/content/data/day4_images.zip', 'r') as zip_ref:
    zip_ref.extractall('images')

import os
os.listdir('images')
```

---

### ☁️ **Day 5 – MLOps & Deployment**

**File:** `day5_predictions_sample.json`
**Description:** Example input/output JSON for API or Gradio demo.
Used to test model serving, predictions, and response format.

**Example Code:**

```python
import json
with open('/content/data/day5_predictions_sample.json') as f:
    sample = json.load(f)
print(sample)
```

---

## ⚠️ Data Policy & Usage

* All datasets are **for training and learning only**.
* You can freely modify or extend them for personal practice.
* Do **not** use these datasets in production or commercial projects.
* For larger versions, ask your instructor or mentor.

---

## 💾 Tip for Google Colab

When cloning or mounting your repo:

```python
!git clone https://github.com/<your-org>/<your-repo>.git
%cd <your-repo>/data
```

Then access any dataset path directly, e.g.:

```python
pd.read_csv('day2_sales_data.csv')
```

---

## 📚 Credits

Some example datasets are adapted from:

* [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
* [Kaggle Sample Datasets](https://www.kaggle.com/datasets)
* [Open Data Portal](https://data.gov/)

---

Would you like me to create **sample CSV and JSON files** (tiny mock datasets — 5–10 rows each) so you can upload them directly into the `/data` folder and link them to these notebooks?
