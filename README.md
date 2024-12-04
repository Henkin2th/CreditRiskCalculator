# Loan Application Risk Dashboard

## Overview

The **Loan Application Risk Dashboard** is an interactive web application built using Streamlit. This dashboard allows users to:

1. Input loan application data (both numerical and categorical factors).
2. Predict the risk grade for a loan application using a trained Random Forest model.
3. Visualize feature contributions through **Permutation Importance**.

The project leverages machine learning and explainable AI techniques to help financial professionals assess loan risks more transparently.

---

## Features

1. **Dynamic Loan Risk Prediction**:
   - Users can input various loan-related factors (e.g., annual income, loan amount, homeownership, loan purpose) to predict the risk grade.

2. **Feature Contribution Visualization**:
   - Bar chart and table showcasing **Permutation Importance** to highlight which features influence the model's predictions the most.

3. **Expanded Feature Set**:
   - Includes both numerical and categorical features:
     - **Numerical**: `annual_income`, `debt_to_income`, `emp_length`, `loan_amount`, `interest_rate`, `installment`.
     - **Categorical**: `homeownership`, `verified_income`, `loan_purpose`, `term`.

4. **Interactive Design**:
   - Built with Streamlit for ease of use and fast prototyping.

---

## Prerequisites

- Python 3.7 or higher
- Libraries:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

Install the required libraries:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

---

## How to Run

1. Clone or download the repository.

2. Place the dataset (`loans_full_schema.csv`) in the same directory as the `app.py` file.

3. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

4. Open the provided URL in your browser to access the dashboard.

---

## Dataset

The dataset used (`loans_full_schema.csv`) contains loan application records with the following key fields:

- **Numerical**:
  - `annual_income`: Applicant's annual income.
  - `debt_to_income`: Debt-to-income ratio.
  - `emp_length`: Length of employment in years.
  - `loan_amount`: Requested loan amount.
  - `interest_rate`: Interest rate on the loan.
  - `installment`: Monthly installment amount.

- **Categorical**:
  - `homeownership`: Applicant's homeownership status.
  - `verified_income`: Whether the income is verified.
  - `loan_purpose`: Purpose of the loan.
  - `term`: Loan repayment term (e.g., 36 or 60 months).

The target variable is `sub_grade`, which is encoded into numerical values for modeling.

---

## File Structure

```
.
├── app.py                  # Main Streamlit application file
├── loans_full_schema.csv   # Dataset file (not included in this repository)
├── README.md               # Project documentation
```

---

## How It Works

1. **Model Training**:
   - A Random Forest Classifier is trained using a combination of numerical and encoded categorical features.
   - The target variable (`sub_grade`) represents the loan risk grade.

2. **Feature Contributions**:
   - The model's predictions are explained using **Permutation Importance**.
   - A bar chart and table highlight the importance of each feature in the prediction.

3. **User Interaction**:
   - Users input loan application data through an interactive panel.
   - The dashboard predicts the loan risk grade and visualizes the importance of each feature.
