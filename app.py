import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

@st.cache_data
def load_and_preprocess_data():
    file_path = 'loans_full_schema.csv'  # Replace with your file path
    data = pd.read_csv(file_path)

    # Encode `sub_grade` to numerical values
    label_encoder = LabelEncoder()
    data['sub_grade_encoded'] = label_encoder.fit_transform(data['sub_grade'])

    # Define numeric and categorical columns
    numeric_cols = [
        'annual_income',
        'debt_to_income',
        'emp_length',
        'loan_amount',
        'interest_rate',
        'installment'
    ]
    categorical_cols = ['homeownership', 'verified_income', 'loan_purpose', 'term']

    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated parameter
    encoded_features = encoder.fit_transform(data[categorical_cols])

    # Combine numeric and encoded categorical features
    processed_data = pd.concat(
        [data[numeric_cols].reset_index(drop=True), pd.DataFrame(encoded_features)],
        axis=1
    )

    # Ensure feature names are consistent
    processed_data.columns = processed_data.columns.astype(str)

    # Drop rows with missing values
    processed_data = processed_data.dropna()

    # Define features (X) and target (y)
    X = processed_data
    y = data.loc[processed_data.index, 'sub_grade_encoded']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, encoder, label_encoder, list(X.columns), X_train, y_train

# Load preprocessed data and trained components
model, encoder, label_encoder, feature_names, X_train, y_train = load_and_preprocess_data()

def dashboard():
    st.title("Loan Application Risk Dashboard with Expanded Features")
    
    # Display feature contributions using Permutation Importance
    st.write("### Feature Contributions (Permutation Importance)")
    perm_importance = permutation_importance(model, X_train, y_train, scoring='accuracy', n_repeats=5, random_state=42)
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]

    importance_df = pd.DataFrame({
        "Feature": [feature_names[idx] for idx in sorted_idx],
        "Mean Importance": perm_importance.importances_mean[sorted_idx],
        "Std Dev": perm_importance.importances_std[sorted_idx]
    })
    st.dataframe(importance_df)

    # Bar plot of feature contributions
    st.write("### Feature Contributions Bar Plot")
    fig, ax = plt.subplots()
    ax.barh(
        [feature_names[idx] for idx in sorted_idx],  # Use string feature names
        perm_importance.importances_mean[sorted_idx]
    )
    ax.set_xlabel("Mean Permutation Importance")
    ax.set_ylabel("Features")
    ax.set_title("Permutation Importance of Features")
    st.pyplot(fig)

    # User inputs for loan application
    st.sidebar.header("Input Loan Application Data")
    annual_income = st.sidebar.number_input("Annual Income", min_value=0, value=50000, step=1000)
    debt_to_income = st.sidebar.number_input("Debt to Income Ratio", min_value=0.0, value=20.0, step=0.1)
    emp_length = st.sidebar.number_input("Employment Length (Years)", min_value=0, value=3, step=1)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=1000, value=15000, step=1000)
    interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, value=10.0, step=0.1)
    installment = st.sidebar.number_input("Installment", min_value=0.0, value=400.0, step=1.0)

    homeownership = st.sidebar.selectbox("Homeownership", encoder.categories_[0])
    verified_income = st.sidebar.selectbox("Verified Income", encoder.categories_[1])
    loan_purpose = st.sidebar.selectbox("Loan Purpose", encoder.categories_[2])
    term = st.sidebar.selectbox("Term", encoder.categories_[3])

    # Encode categorical inputs
    encoded_inputs = encoder.transform([[homeownership, verified_income, loan_purpose, term]])

    # Prepare input array for prediction
    input_data = np.concatenate((
        [annual_income, debt_to_income, emp_length, loan_amount, interest_rate, installment],
        encoded_inputs[0]
    )).reshape(1, -1)
    
    # Convert input_data to DataFrame with feature names
    input_df = pd.DataFrame(input_data, columns=feature_names)

    st.write(f"### Input Data for Prediction")
    st.dataframe(input_df)

    # Generate prediction
    if st.sidebar.button("Generate Prediction"):
        prediction = model.predict(input_data)
        predicted_class = prediction[0]
        risk_score = label_encoder.inverse_transform([predicted_class])[0]
        st.write(f"### Predicted Risk Grade: {risk_score}")

if __name__ == "__main__":
    dashboard()
