# 📈 Predicting Weekly Stock Market Returns (Panel Data Approach)

**Course:** CS31A0850 AI-Driven Data Analytics and Insights (LUT University)  
**Author:** [Sinun Nimesi]  
**Dataset:** Finland Stock Market Prices Prediction (Dataset 3)

## 📌 Project Overview
This repository contains a comprehensive data science project aimed at predicting the weekly percentage change (`%Chg`) of five major Finnish listed companies (Nokia, Kone, Nordea, UPM, Olvi). The project demonstrates a rigorous end-to-end machine learning pipeline, from data cleaning and exploratory data analysis (EDA) to the deployment of a predictive model using Streamlit.

A significant focus of this project is placed on **methodological purity and preventing data leakage** in time-series forecasting, a common pitfall in financial data modeling.

## 🛠️ Methodological Approach

Financial data is inherently noisy with a low signal-to-noise ratio. To build a robust model, the following strict methodologies were applied:

1. **Panel Data Construction:** Data from all five companies were combined into a single panel dataset, sorted meticulously by company and chronologically to prevent temporal mixing.
2. **Target Variable (`target_actual_chg`):** The objective was explicitly defined as predicting the *next week's* percentage change using *current week's* data (`%Chg.shift(-1)`).
3. **Feature Engineering without Leakage:** * Created technical features including short-term momentum (`chg_lag1`, `chg_lag2`), mid-term momentum (`chg_lag4`), intra-week volatility (`range_pct`), and rolling volatility (`chg_rolling_std_4`).
   * **Crucial:** All features were calculated *before* any missing values were dropped to maintain the integrity of the rolling calculations.
4. **Strict Time-Series Split:** To prevent looking into the future, the data was split chronologically rather than randomly:
   * **Train Set:** Data up to the end of 2023.
   * **Validation Set:** Year 2024 (Used for hyperparameter tuning and early stopping).
   * **Test (Holdout) Set:** Year 2025 onwards.
5. **Pipeline Architecture:** Scikit-learn's `Pipeline` with `RobustScaler` was utilized for linear models to ensure scaling parameters were learned exclusively from the training data, eliminating leakage.

## 🔬 Exploratory Data Analysis (EDA) Highlights

The EDA phase uncovered several characteristics typical of financial markets that guided model selection:
* **Scale Differences:** Absolute closing prices varied massively between companies (e.g., Nokia at ~5€ vs. Kone at ~50€), confirming the necessity of using `%Chg` as a normalized target.
* **Fat Tails:** The distribution of returns showed "fat tails" (extreme weekly drops or spikes), favoring the use of robust models (like `HuberRegressor` or heavily regularized `Ridge`).
* **Volatility Clustering:** Rolling standard deviation plots confirmed that periods of high market turbulence cluster together, making volatility a key predictive feature.

## 🤖 Models Evaluated

The project evaluated several approaches, refining them iteratively:
1. **Naïve Baseline:** Predicting that next week's return will be identical to this week's.
2. **Always-Up Baseline:** A market trend baseline predicting constant positive growth.
3. **Optimized Ridge Regression:** A highly regularized linear model (`alpha=5000`) optimized via `GridSearchCV` on the validation set.
4. **Optimized XGBoost:** A gradient boosting model utilizing the validation set for `early_stopping_rounds` to explicitly prevent test-set leakage.

## 📊 Results & Interpretation

The final evaluation ("Reality Check") on the unseen 2025+ holdout dataset yielded the following key insights:

| Model | RMSE | MAE | R² | Directional Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline: Always-Up** | 0.0341 | 0.0256 | -0.0285 | 50.2 % |
| **Optimized Ridge** | **0.0337** | **0.0253** | **-0.0073** | **51.9 %** |
| **Optimized XGBoost** | 0.0338 | 0.0256 | -0.0132 | 47.4 % |

* **The Reality of R²:** As expected in short-term stock market forecasting (Efficient Market Hypothesis), the R² values remained slightly negative. This honestly reflects that historical price data alone cannot explain the full variance caused by external news, macroeconomics, or earnings reports.
* **The "Edge":** The **Optimized Ridge Regression** emerged as the winner. Its high regularization filtered out the market noise, allowing it to achieve a **51.9% Directional Accuracy**. Crucially, this mathematically outperforms the blind "Always-Up" market trend (50.2%), proving the model found a genuine, albeit small, statistical signal.
* **Complexity vs. Performance:** The more complex XGBoost model underperformed the simpler Ridge model on the holdout set, demonstrating how easily tree-based models overfit to the noise in financial data when properly restricted from data leakage.

## 🚀 Deployment (Streamlit Dashboard)

The winning model (**Optimized Ridge**) has been deployed via a local **Streamlit Dashboard** designed as a "Decision Support Tool" for analysts. 

The dashboard provides:
* Company-specific prediction filtering.
* The forecasted percentage change for the upcoming week.
* A clear, prescriptive recommendation (NOUSU 🟢 / LASKU 🔴).
* Visual comparison of historical simulated performance.
* Feature importance insights explaining the model's logic.

### How to Run the App Locally
1. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn streamlit joblib
