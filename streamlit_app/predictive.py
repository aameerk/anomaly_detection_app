import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import plotly.express as px
import mlflow
import mlflow.sklearn
import pygwalker as pyg
import streamlit.components.v1 as components
import os
from io import StringIO

# Setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

st.set_page_config(page_title="Predictive Analytics App", page_icon="🧠", layout="wide")

# Authentication (re-use your logic)
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        with st.form("login"):
            st.title("Predictive Analytics App")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            if submit and username == "superuser" and password == "superuser23":
                st.session_state.authenticated = True
            elif submit:
                st.error("Invalid credentials")
        return False
    return True

if not check_password():
    st.stop()

# Upload file
st.title("📈 Predictive Analytics App")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(30).style.highlight_null(color="red"))

    st.markdown("## 🔍 Exploratory Data Analysis with PyGWalker")
    try:
        pyg_html = pyg.walk(df, return_html=True)
        components.html(pyg_html, height=600, scrolling=True)
    except Exception as e:
        st.warning(f"PyGWalker failed to render: {str(e)}")

    # Column selection
    target = st.selectbox("🎯 Select Target Column", df.columns)
    features = st.multiselect("🧠 Select Feature Columns", [col for col in df.columns if col != target], default=[col for col in df.select_dtypes(include=[np.number]).columns if col != target])

    if st.button("Run Prediction"):
        X = df[features].copy()
        y = df[target].copy()

        # Fill nulls
        X.fillna(X.median(numeric_only=True), inplace=True)
        y.fillna(y.mode()[0], inplace=True)

        # Determine regression/classification
        task_type = "classification" if y.nunique() <= 10 else "regression"
        model = RandomForestClassifier(random_state=42) if task_type == "classification" else RandomForestRegressor(random_state=42)

        # Train model
        with mlflow.start_run():
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mlflow.sklearn.log_model(model, "predictive_model")
            mlflow.log_param("task_type", task_type)
            mlflow.log_param("features", features)
            mlflow.log_param("target", target)

            if task_type == "regression":
                mse = mean_squared_error(y_test, y_pred)
                mlflow.log_metric("mse", mse)
                st.success(f"✅ Regression Model Trained — MSE: {mse:.2f}")
            else:
                acc = accuracy_score(y_test, y_pred)
                mlflow.log_metric("accuracy", acc)
                st.success(f"✅ Classification Model Trained — Accuracy: {acc:.2%}")
                cm = confusion_matrix(y_test, y_pred)
                st.write("Confusion Matrix")
                st.dataframe(pd.DataFrame(cm))

            # Feature importance
            importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            st.subheader("📊 Feature Importance")
            fig = px.bar(importance, orientation='h', title="Feature Importance")
            st.plotly_chart(fig)

            # Prediction distribution
            st.subheader("🔮 Prediction Distribution")
            fig = px.histogram(y_pred, nbins=30, title="Predicted Values Distribution")
            st.plotly_chart(fig)
