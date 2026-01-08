import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import os

# Page Config
st.set_page_config(page_title="Demand Forecast Analytics", layout="wide")

st.title("📦 E-Commerce Demand Forecasting Dashboard")
st.markdown("Interactive forecasting tool for product inventory management.")

# Sidebar Controls
st.sidebar.header("Forecast Settings")
horizon = st.sidebar.selectbox("Select Forecast Horizon (Days)", [7, 14, 30], index=0)

if st.sidebar.button("Generate Forecast"):
    with st.spinner(f"Fetching {horizon}-day forecast from Production API..."):
        try:
            # We assume the FastAPI app is running on localhost:8000
            # For demonstration, we'll try to call the API or fallback to direct logic
            response = requests.get(f"http://localhost:8000/predict?horizon={horizon}")
            
            if response.status_code == 200:
                data = response.json()
                predictions = data['predictions']
                
                # Convert to DataFrame
                forecast_df = pd.DataFrame(list(predictions.items()), columns=['Date', 'Forecasted_Units'])
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
                
                # Visualizations
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"{horizon}-Day Demand Forecast Trend")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(forecast_df['Date'], forecast_df['Forecasted_Units'], marker='o', linestyle='-', color='#1f77b4')
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Demand Units")
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Forecast Data")
                    st.dataframe(forecast_df, height=350)
                    
                st.success(f"Successfully generated forecast using the Production model.")
                
                # Download button
                st.download_button(
                    label="Download Forecast CSV",
                    data=forecast_df.to_csv(index=False),
                    file_name=f"demand_forecast_{horizon}days.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                st.info("💡 Ensure the FastAPI server is running with: `python src/deployment/api.py`")
                
        except Exception as e:
            st.error(f"Connection Error: {e}")
            st.info("💡 Start the FastAPI server first: `python src/deployment/api.py`")

# Background Info Section
st.divider()
st.subheader("Business Context & Metrics")
st.write("This dashboard utilizes the latest **XGBoost/LightGBM** production model registered in MLflow. Selection is primarily driven by **WAPE** performance.")

# Display historical snippet if available
if os.path.exists("data/processed/processed.csv"):
    hist_df = pd.read_csv("data/processed/processed.csv", index_col=0)
    st.markdown("**Historical Context (Last 7 Days):**")
    st.table(hist_df.tail(7)[['Units_Sold', 'Price', 'Discount', 'Holiday_Promotion']])
