import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import mlflow
import os
import time
from datetime import datetime
import warnings
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from PIL import Image
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# MLflow config
mlflow.set_tracking_uri("sqlite:///mlflow.db")
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

# Page setup
st.set_page_config(page_title="Anomaly Detection Tool", page_icon="📊", layout="wide")
import streamlit as st



st.markdown("""
    <style>
        /* Main Title Style */
        .main-title {
            font-size: 3em;
            font-weight: 700;
            color:rgb(14, 46, 86);
            text-align: center;
        }
        /* Subtitle Style */
        .sub-title {
            font-size: 1.2em;
            color: #555;
            text-align: center;
            margin-bottom: 30px;
        }
     
     
        /* Logo Style */
        .logo {
            position: absolute;
            top: 10px;
            left: 20px;
            width: 50px;
        }
       
    </style>
""", unsafe_allow_html=True)




# --- Auth ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("<h1 class='main-title'>🔐 Anomaly Detection Tool</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Please login to continue</p>", unsafe_allow_html=True)

        with st.container():
            with st.form("login_form"):
                st.markdown("<div class='login-box'>", unsafe_allow_html=True)

                # Login fields
                username = st.text_input("👤 Username")
                password = st.text_input("🔑 Password", type="password")
                login_button = st.form_submit_button("🚪 Login")

                st.markdown("</div>", unsafe_allow_html=True)

                # Validate login on button click
                if login_button:
                    if username == "superuser" and password == "superuser23":
                        st.session_state.authenticated = True
                        st.success("✅ Login successful! Loading dashboard...")
                        st.rerun()  # Refresh and show the app
                    else:
                        st.error("❌ Invalid username or password.")

        return False

    return True

# Check authentication status and stop until logged in
if not check_password():
    st.stop()

# If logged in, show the app content
st.markdown("<h1 class='main-title'>📊 Anomaly Detection Tool</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='sub-title'>Welcome, <b>{st.session_state.get('username', 'superuser')}</b></p>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # --- Pydantic Validation ---
        st.subheader("✅ Data Validation with Pydantic")

        class HousingRecord(BaseModel):
            longitude: float = Field(..., ge=-180, le=180)
            latitude: float = Field(..., ge=-90, le=90)
            housing_median_age: float = Field(..., ge=0)
            total_rooms: float = Field(..., ge=0)
            total_bedrooms: Optional[float] = Field(None, ge=0)
            population: float = Field(..., ge=0)
            households: float = Field(..., ge=0)
            median_income: float = Field(..., ge=0)
            median_house_value: float = Field(..., ge=0)

        validated_rows, invalid_rows = [], []

        for i, row in df.iterrows():
            try:
                validated = HousingRecord(**row.to_dict())
                validated_rows.append(validated.model_dump())
            except ValidationError as e:
                invalid_rows.append((i, str(e)))

        if invalid_rows:
            st.warning(f"⚠️ {len(invalid_rows)} rows failed validation and will be excluded.")
            if st.checkbox("Show invalid rows"):
                st.write(invalid_rows[:10])


        # --- Dataset Info ---
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Missing", df.isnull().sum().sum())
        col4.metric("Duplicates", df.duplicated().sum())

        if df.isnull().sum().any():
            st.warning("Null values detected:")
            st.write(df.isnull().sum()[df.isnull().sum() > 0])

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        pd_df = pd.DataFrame(validated_rows)
        st.write(pd_df)
        # --- Anomaly Detection ---
        st.subheader("🧪 Anomaly Detection")
        anomaly_indices = []
        if len(numeric_cols) > 0:
            with mlflow.start_run():
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomalies = iso_forest.fit_predict(df[numeric_cols])
                mlflow.sklearn.log_model(iso_forest, "isolation_forest_model")
                anomaly_indices = np.where(anomalies == -1)[0]
                df["Anomaly"] = anomalies

                if len(anomaly_indices) > 0:
                    st.success(f"🔍 {len(anomaly_indices)} anomalies detected.")
                    st.subheader("📌 Anomalous Data Points")
                    st.dataframe(df.loc[anomaly_indices].head(20))

                    st.subheader("📊 Column-wise Anomaly Summary")
                    anomaly_means = df.loc[anomaly_indices, numeric_cols].mean()
                    normal_means = df.loc[df["Anomaly"] != -1, numeric_cols].mean()
                    delta = (anomaly_means - normal_means).abs().sort_values(ascending=False)
                    st.write("Top contributing columns:")
                    st.write(delta.head(5))

        # --- Data Visualization ---
        st.subheader("📈 Data Visualization")
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X Axis", numeric_cols)
                y_col = st.selectbox("Y Axis", numeric_cols)
                fig = px.scatter(df, x=x_col, y=y_col, color=df["Anomaly"].astype(str), title="Scatter Plot")
                st.plotly_chart(fig)
            with col2:
                col = st.selectbox("Histogram Column", numeric_cols)
                st.plotly_chart(px.histogram(df, x=col, title=f"Distribution of {col}"))

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            st.subheader("🕒 Time-Series Anomaly View")
            st.plotly_chart(px.line(df, x='timestamp', y=numeric_cols[0], color=df['Anomaly'].astype(str)))

        # --- Gemini Integration ---
        from pandasai import SmartDataframe
        from pandasai.llm.google_gemini import GoogleGemini

        st.subheader("🤖 Ask Gemini About Your Data")

        safe_cache_dir = "tmp/pandasai_cache"
        os.makedirs(safe_cache_dir, exist_ok=True)
        llm = GoogleGemini(api_key=st.secrets["gemini"]["api_key"], model="gemini-1.5-flash")

        


        sdf = SmartDataframe(df, config={"llm": llm, "verbose": True, "cache_dir":"safe_cache_dir"})
        user_prompt = st.text_input("Ask a question about your data")

        if user_prompt:
            start_time = time.time()
            timeout = 30
            with st.spinner("Gemini is thinking..."):
                try:
                    result = sdf.chat(user_prompt)
                    elapsed_time = time.time() - start_time
                    if elapsed_time > timeout:
                        st.warning("⏳ Gemini took too long. Please try again.")
                    else:
                        st.success("✅ Gemini's Response:")
                        if isinstance(result, pd.DataFrame):
                            st.dataframe(result)
                        else:
                            st.write(result)
                except Exception as e:
                    st.error(f"❌ Gemini Error: {e}")
        else:
            st.warning("⚠️ Please enter a question to ask Gemini.")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)
