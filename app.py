import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and prepare data
@st.cache_data
def load_and_train_model():
    # Load data
    insurance_data = pd.read_csv("insurance.csv")
    
    # Prepare features and target
    x = insurance_data.drop(columns=["charges", "region"])
    y = insurance_data["charges"]
    
    # Encode categorical variables
    x["sex"] = x["sex"].map({"female": 0, "male": 1})
    x["smoker"] = x["smoker"].map({"no": 0, "yes": 1})
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # Calculate R² score
    y_pred = model.predict(x_test)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    
    n = x_test.shape[0]
    p = x_test.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return model, r2, adjusted_r2

# Page configuration
st.set_page_config(page_title="Insurance Charge Predictor", page_icon="🏥", layout="wide")

# Title and description
st.title("🏥 Insurance Charge Predictor")
st.markdown("### Predict medical insurance charges based on personal information")

# Load model
model, r2, adjusted_r2 = load_and_train_model()

# Display model performance
st.sidebar.header("📊 Model Performance")
st.sidebar.metric("R² Score", f"{r2:.4f}")
st.sidebar.metric("Adjusted R² Score", f"{adjusted_r2:.4f}")

# Create input form
st.header("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("Sex", options=["Female", "Male"])
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

with col2:
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
    smoker = st.selectbox("Smoker", options=["No", "Yes"])

# Predict button
if st.button("🔮 Predict Insurance Charges", type="primary"):
    # Prepare input data
    sex_encoded = 0 if sex == "Female" else 1
    smoker_encoded = 0 if smoker == "No" else 1
    
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_encoded],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_encoded]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display result
    st.success("### Prediction Complete!")
    st.metric("Estimated Annual Insurance Charges", f"${prediction:,.2f}")
    
    # Additional insights
    st.info(f"""
    **Input Summary:**
    - Age: {age} years
    - Sex: {sex}
    - BMI: {bmi:.1f}
    - Children: {children}
    - Smoker: {smoker}
    """)

# Footer
st.markdown("---")
st.markdown("*This prediction is based on a Linear Regression model trained on insurance data.*")
