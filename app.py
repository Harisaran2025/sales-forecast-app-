# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import warnings
import os

warnings.filterwarnings("ignore")  # Hides unnecessary warning messages

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="🛍️ Sales Prediction (CatBoost)", page_icon="🛒", layout="centered")

# ✅ Create temp folder for CatBoost (IMPORTANT FIX)
os.makedirs("tmp", exist_ok=True)

# -----------------------------
# Background Style
# -----------------------------
page_bg = """  
<style>
body {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    background-attachment: fixed;
    color: white;
}
div.stApp { background: transparent; }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -----------------------------
# Title and Description
# -----------------------------
st.markdown("<h1 style='text-align:center; color:white;'>🛍️ Sales Prediction Using CatBoost</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#f0f0f0;'>Upload your Excel file containing <b>Date</b> and <b>Sales</b> columns. Then click <b>Predict</b> to forecast future sales.</p>", unsafe_allow_html=True)

# -----------------------------
# File Upload Section
# -----------------------------
uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    
    df = pd.read_excel(uploaded_file)
    df.columns = [col.strip().capitalize() for col in df.columns]

    if 'Date' not in df.columns or 'Sales' not in df.columns:
        st.error("❌ The Excel file must contain 'Date' and 'Sales' columns.")
    else:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        st.success("✅ File uploaded successfully!")
        st.write("### 📊 Uploaded Data Preview:")
        st.dataframe(df.head())

        st.write("### 📈 Sales Trend:")
        fig, ax = plt.subplots()
        ax.plot(df['Date'], df['Sales'], marker='o', color='cyan')
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.set_title("Sales Over Time")
        st.pyplot(fig)

        # -----------------------------
        # Prediction Section
        # -----------------------------
        st.markdown("### 🔮 Click below to Predict Future Sales:")
        if st.button("🔮 Predict"):
            with st.spinner("Training CatBoost model... please wait ⏳"):

                df_model = df.copy()

                df_model["lag1"] = df_model["Sales"].shift(1)
                df_model["lag7"] = df_model["Sales"].shift(7)
                df_model["lag30"] = df_model["Sales"].shift(30)

                df_model = df_model.dropna()

                X = df_model[["lag1", "lag7", "lag30"]]
                y = df_model["Sales"]

                # ✅ FIXED CatBoost model (tmp directory added)
                model = CatBoostRegressor(
                    iterations=500,
                    depth=6,
                    learning_rate=0.05,
                    loss_function='RMSE',
                    train_dir="tmp",   # <<< IMPORTANT FIX
                    verbose=False
                )

                model.fit(X, y)

                # -----------------------------
                # Predict Next 7 Days
                # -----------------------------
                future_predictions = []
                last_date = df['Date'].iloc[-1]
                temp_data = df.copy()

                for i in range(7):
                    lag1 = temp_data["Sales"].iloc[-1]
                    lag7 = temp_data["Sales"].iloc[-7] if len(temp_data) >= 7 else lag1
                    lag30 = temp_data["Sales"].iloc[-30] if len(temp_data) >= 30 else lag7

                    pred = model.predict([[lag1, lag7, lag30]])[0]

                    next_date = last_date + pd.Timedelta(days=i+1)
                    future_predictions.append([next_date, round(pred, 2)])

                    temp_data = pd.concat([
                        temp_data,
                        pd.DataFrame({"Date": [next_date], "Sales": [pred]})
                    ], ignore_index=True)

                forecast_df = pd.DataFrame(future_predictions, columns=["Date", "Predicted_Sales"])

                st.success("✅ Prediction Complete!")
                st.write("### 📅 Predicted Sales for Next 7 Days:")
                st.dataframe(forecast_df)

                st.write("### 📊 Actual vs Predicted Sales:")
                fig2, ax2 = plt.subplots()
                ax2.plot(df['Date'], df['Sales'], label='Actual Sales', marker='o')
                ax2.plot(
                    forecast_df['Date'], 
                    forecast_df['Predicted_Sales'],
                    label='Predicted Sales', 
                    marker='x', 
                    color='orange'
                )
                ax2.legend()
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Sales")
                ax2.set_title("Sales Forecast using CatBoost")
                st.pyplot(fig2)

                csv = forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Predictions (CSV)",
                    data=csv,
                    file_name="predicted_sales_catboost.csv",
                    mime="text/csv"
                )

else:
    st.warning("Please upload a .xlsx file to start.")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    "<hr><p style='text-align:center; color:white;'>© 2025 Sales Predictor • Powered by CatBoost Model</p>",
    unsafe_allow_html=True
)
