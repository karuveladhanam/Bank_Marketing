import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load Saved Files
# -------------------------------

with open("bank_logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("bank_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("bank_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("bank_feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Term Deposit Prediction")
st.write("Predict whether a customer will subscribe to a term deposit.")

st.markdown("---")

# -------------------------------
# User Inputs
# -------------------------------

age = st.number_input("Age", 18, 100, 35)

job = st.selectbox(
    "Job",
    encoders["job"].classes_
)

marital = st.selectbox(
    "Marital Status",
    encoders["marital"].classes_
)

education = st.selectbox(
    "Education",
    encoders["education"].classes_
)

default = st.selectbox(
    "Credit in Default",
    encoders["default"].classes_
)

balance = st.number_input("Account Balance", value=1000)

housing = st.selectbox(
    "Housing Loan",
    encoders["housing"].classes_
)

loan = st.selectbox(
    "Personal Loan",
    encoders["loan"].classes_
)

contact = st.selectbox(
    "Contact Type",
    encoders["contact"].classes_
)

day = st.slider("Last Contact Day", 1, 31, 15)

month = st.selectbox(
    "Month",
    encoders["month"].classes_
)

duration = st.number_input(
    "Last Contact Duration (seconds)",
    value=300
)

campaign = st.number_input(
    "Number of Contacts During Campaign",
    min_value=1,
    value=1
)

pdays = st.number_input(
    "Days Since Previous Contact",
    value=-1
)

previous = st.number_input(
    "Previous Contacts",
    value=0
)

poutcome = st.selectbox(
    "Previous Campaign Outcome",
    encoders["poutcome"].classes_
)

# -------------------------------
# Encode Inputs
# -------------------------------

job = encoders["job"].transform([job])[0]
marital = encoders["marital"].transform([marital])[0]
education = encoders["education"].transform([education])[0]
default = encoders["default"].transform([default])[0]
housing = encoders["housing"].transform([housing])[0]
loan = encoders["loan"].transform([loan])[0]
contact = encoders["contact"].transform([contact])[0]
month = encoders["month"].transform([month])[0]
poutcome = encoders["poutcome"].transform([poutcome])[0]

# -------------------------------
# Create DataFrame
# -------------------------------

input_df = pd.DataFrame(
    [[
        age,
        job,
        marital,
        education,
        default,
        balance,
        housing,
        loan,
        contact,
        day,
        month,
        duration,
        campaign,
        pdays,
        previous,
        poutcome
    ]],
    columns=feature_names
)

# -------------------------------
# Scale Data
# -------------------------------

scaled_input = scaler.transform(input_df)

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict"):

    prediction = model.predict(scaled_input)[0]

    probability = model.predict_proba(scaled_input)[0]

    result = encoders["deposit"].inverse_transform([prediction])[0]

    st.markdown("---")

    if result == "yes":
        st.success("✅ Customer is likely to subscribe to the Term Deposit.")
    else:
        st.error("❌ Customer is not likely to subscribe to the Term Deposit.")

    st.subheader("Prediction Probability")

    st.write(f"**No :** {probability[0]*100:.2f}%")

    st.write(f"**Yes :** {probability[1]*100:.2f}%")
