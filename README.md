# fraud-detection-system-with-explainable-ai

# 🔍 Real-Time Fraud Detection System with Explainable AI & Live Dashboard

> **Week 3 & 4 Project | AI & Data Analyst Internship**

An end-to-end fraud detection pipeline combining state-of-the-art machine learning, severe class imbalance handling, Explainable AI (SHAP), and a live interactive Streamlit dashboard — built to simulate real analyst work at a fintech or banking firm.

---

## 📌 Problem Statement

Financial fraud costs the global economy over **$5 trillion annually**. Credit card fraud, identity theft, and transaction anomalies demand systems that can detect suspicious activity in milliseconds — not after the damage is done. Legacy rule-based systems miss novel attack patterns; simple ML models are black boxes that compliance teams can't trust.

This project builds an end-to-end Fraud Detection System that:
- Handles severe class imbalance using **SMOTE**
- Trains and compares multiple advanced ML models
- Uses **SHAP** to explain every prediction in plain English
- Packages results in a **live Streamlit dashboard** a real fraud analyst could deploy and trust

---

## 📁 Repository Structure

```
fraud-detection-system-with-explainable-ai/
│
├── Analysis.ipynb          # Main Colab Notebook (Tasks 1–5, 7–8)
├── Dashboard/              # Streamlit multi-page dashboard application
├── Chart/                  # All generated visualization charts
├── Dataset/                # Kaggle dataset link (CSVs not included due to large file size)
├── sample_data.csv         # Sample dataset for dashboard demo
├── Model_Comparison.png    # Side-by-side model performance comparison
├── Shap_Summary.png        # SHAP global feature importance summary plot
├── Summary.pdf             # Final insights & business recommendations report
├── requirements.txt        # All required Python libraries
└── .devcontainer/          # Dev container config for reproducible environment
```

---

## ✅ Tasks Completed

### Task 1 — Data Loading, Merging & Exploratory Analysis
- Loaded both CSVs and **merged on TransactionID** using Pandas
- Displayed shape, dtypes, and first 10 rows of the merged dataset
- Analysed the `isFraud` target column — quantified and visualized **class imbalance**
- Identified missing values column-by-column
- Applied drop vs. impute threshold — dropped columns with **>50% missing values**
- Plotted **TransactionAmt distribution** for fraud vs. non-fraud using log scale
- Computed a **correlation heatmap** of the top 20 numerical features using Seaborn

### Task 2 — Preprocessing, Imbalance Handling & Feature Engineering
- Dropped high-missing columns (>50% threshold)
- Imputed remaining values: **Median** (numerical), **Mode** (categorical)
- Applied **Label Encoding** on high-cardinality categorical columns (strategy justified in Markdown)
- Engineered 3 new features:
  - `AmtToMeanRatio` = TransactionAmt / mean(TransactionAmt)
  - `HourOfDay` = extracted from TransactionDT
  - `DeviceRisk` = binary flag based on DeviceType and DeviceInfo
- Applied **SMOTE** on the training set only to handle class imbalance
- Scaled numerical features using **RobustScaler**
- Performed **stratified 80/20 train-test split**
- Reported class ratio before and after SMOTE

### Task 3 — Model Training, Comparison & Threshold Optimization
- Trained 3 classification models:
  - **Model 1:** LightGBM Classifier
  - **Model 2:** XGBoost Classifier
  - **Model 3:** Isolation Forest
- Evaluated all models with: Accuracy, Precision, Recall, F1-Score, ROC-AUC, **PR-AUC**
- Plotted **Confusion Matrix** for each model
- Plotted **ROC Curve** and **Precision-Recall Curve** for all models
- Optimized classification threshold using **Threshold vs F1-Score plot** *(Advanced)*
- Tuned best model using **Optuna / RandomizedSearchCV** *(Advanced)*
- Documented all improvements in Markdown

### Task 4 — Explainable AI with SHAP Values *(Advanced)*
- Installed and ran the **SHAP** library
- Generated **Global SHAP Summary Plot** (top 20 features) → [`Shap_Summary.png`](./Shap_Summary.png)
- Generated **SHAP Waterfall Plots** for 3 transaction cases:
  - Confirmed fraud transaction
  - Borderline case (~0.50 probability)
  - Legitimate transaction
- Generated **SHAP Dependence Plot** for key features
- Explained all 3 transactions in **plain English** for non-technical stakeholders
- Compared SHAP importance vs model feature importance

### Task 5 — Risk Segmentation & Fraud Pattern Analysis *(Advanced)*
- Segmented all transactions using fraud probability into 3 risk tiers:
  - 🔴 **Critical Risk** → probability ≥ 0.75
  - 🟡 **Suspicious** → probability between 0.40 and 0.74
  - 🟢 **Clear** → probability < 0.40
- Counted transactions per tier and computed per-tier averages:
  - Average `TransactionAmt`, device type distribution, hour-of-day pattern
- Created **grouped bar chart** comparing all 3 tiers across key features
- Identified **top 3 fraud patterns** from Critical Risk transactions

### Task 6 — Streamlit Fraud Operations Dashboard *(Advanced)*
Built a **multi-page Streamlit application** with:

- **Page 1 — Overview:** Total transactions, fraud count, detection rate, average fraud amount
- **Page 2 — Transaction Explorer:** Searchable & filterable table with live risk score by TransactionID
- **Page 3 — SHAP Explainer:** User enters TransactionID → SHAP waterfall plot + plain-English explanation

Additional features:
- Sidebar filters
- Plotly interactive charts
- Deployed on **Streamlit Community Cloud**

> 🔗 **Live Dashboard:** *(https://fraud-detection-system-with-explainable-ai-5rtlqu5k7cchtzgsfee.streamlit.app/)*

### Task 7 — Visualizations (5+ Charts)
- **Chart 1:** SHAP Global Summary Plot → [`Shap_Summary.png`](./Shap_Summary.png)
- **Chart 2:** Fraud rate by hour of day
- **Chart 3:** TransactionAmt distribution (fraud vs. non-fraud, log scale)
- **Chart 4:** Risk tier donut chart (Critical / Suspicious / Clear)
- **Chart 5:** Precision-Recall Curve with optimal threshold marked
- **Bonus:** Interactive Plotly scatter plot — TransactionAmt vs HourOfDay, colored by fraud probability

All charts saved in the `Chart/` folder. Model comparison: [`Model_Comparison.png`](./Model_Comparison.png)

### Task 8 — Insights & Business Recommendations
A structured analysis written inside the notebook covering:
- Which model performed best and why
- Why **PR-AUC matters more than accuracy** in fraud detection
- Top 3 fraud signals identified by SHAP
- Common characteristics of Critical Risk transactions
- 2 actionable fraud prevention policies
- Estimated annual money saved
- Model limitations and additional data that could improve performance

Full written summary: [`Summary.pdf`](./Summary.pdf)

---

## 📦 Dataset

The dataset used in this project is the **IEEE-CIS Fraud Detection** dataset from Kaggle. The raw CSV files are not included in this repository due to their large file size.

> 🔗 **Download the dataset here:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)

The dataset contains two files that are merged on `TransactionID`:
- `train_transaction.csv` — transaction-level features and the `isFraud` target label
- `train_identity.csv` — identity and device information per transaction

After downloading, place both files inside the `Dataset/` folder before running the notebook.

---

## 🛠️ Tools & Libraries Used

| Tool | Purpose |
|---|---|
| Python 3.x | Main programming language |
| Google Colab | Notebook development environment |
| Pandas / NumPy | Data loading and manipulation |
| LightGBM | Primary classifier |
| XGBoost | Comparison model |
| Scikit-learn | ML utilities, metrics, preprocessing |
| imbalanced-learn | SMOTE for class imbalance |
| SHAP | Explainable AI |
| Optuna | Hyperparameter tuning |
| Plotly | Interactive visualizations |
| Streamlit | Live fraud operations dashboard |
| Matplotlib / Seaborn | Static charts and heatmaps |

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run This Project

### Option 1 — Open Notebook in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smohan-21/fraud-detection-system-with-explainable-ai/blob/main/Analysis.ipynb)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open Notebook → GitHub**
3. Paste: `https://github.com/smohan-21/fraud-detection-system-with-explainable-ai`
4. Select `Analysis.ipynb`
5. Run the first cell to install dependencies:
```python
!pip install -r requirements.txt
```

### Option 2 — Run the Streamlit Dashboard Locally

```bash
# Clone the repository
git clone https://github.com/smohan-21/fraud-detection-system-with-explainable-ai.git
cd fraud-detection-system-with-explainable-ai

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run Dashboard/app.py
```

> Or visit the **live deployed version** directly: *(https://fraud-detection-system-with-explainable-ai-5rtlqu5k7cchtzgsfee.streamlit.app/)*

---

## 📊 Key Outputs

| Output | Location |
|---|---|
| Full analysis notebook | `Analysis.ipynb` |
| Model performance comparison | `Model_Comparison.png` |
| SHAP global summary plot | `Shap_Summary.png` |
| All visualizations | `Chart/` folder |
| Live dashboard app | `Dashboard/` folder |
| Business insights report | `Summary.pdf` |
| Sample data for dashboard | `sample_data.csv` |

---

## 👤 Author

**Mohan S**
AI & Data Analyst Intern — Week 3 & 4 Project
GitHub: [@smohan-21](https://github.com/smohan-21)

---

## 📄 License

This project is built for internship and educational purposes.
