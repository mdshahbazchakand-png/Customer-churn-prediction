"""Streamlit demo: predict telecom customer churn with the tuned Random Forest."""
import joblib
import streamlit as st

from features import predict_churn_probability

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")


model = load_model()

st.title("Customer Churn Predictor")
st.caption("Enter a telecom customer's details to estimate the probability that they will churn.")

with st.sidebar:
    st.header("About this model")
    st.write(
        "Tuned Random Forest trained on the Telco Customer Churn dataset "
        "(7,043 customers, 26.5% churn rate)."
    )
    st.write("Held-out test set: **F1 0.63**, **ROC-AUC 0.84**, **77% recall** on churners.")
    st.write("This is a portfolio demo, not a tool for real business decisions.")

YES_NO = ["Yes", "No"]
ADDON = ["Yes", "No", "No internet service"]

c1, c2 = st.columns(2)
with c1:
    st.subheader("Customer")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has partner", YES_NO)
    dependents = st.selectbox("Has dependents", YES_NO)
    tenure = st.slider("Tenure (months)", 0, 72, 12)

    st.subheader("Billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", YES_NO)
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.slider("Monthly charges", 18.0, 120.0, 70.0, step=0.5)

with c2:
    st.subheader("Services")
    phone = st.selectbox("Phone service", YES_NO)
    lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online security", ADDON)
    backup = st.selectbox("Online backup", ADDON)
    protection = st.selectbox("Device protection", ADDON)
    support = st.selectbox("Tech support", ADDON)
    tv = st.selectbox("Streaming TV", ADDON)
    movies = st.selectbox("Streaming movies", ADDON)

if st.button("Predict churn risk", type="primary"):
    customer = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": round(monthly * tenure, 2),  # approximated from tenure x monthly
    }
    p = predict_churn_probability(model, customer)
    st.metric("Churn probability", f"{p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))
    if p >= 0.6:
        st.error("High churn risk: consider a retention offer or a contract upgrade.")
    elif p >= 0.3:
        st.warning("Medium churn risk: worth monitoring.")
    else:
        st.success("Low churn risk.")
    st.caption("Total charges are estimated as monthly charges x tenure. "
               "Risk bands (30% / 60%) are illustrative.")
