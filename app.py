import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Air Passengers Forecast",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Airline Passengers Time Series Forecasting")
st.markdown("This app forecasts monthly airline passengers using the Prophet model.")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url)
    df.columns = ['Month', 'Passengers']
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.set_index('Month')
    return df

df = load_data()


st.sidebar.header("Forecast Settings")
forecast_months = st.sidebar.slider(
    "How many months to forecast?",
    min_value=6,
    max_value=36,
    value=12,
    step=6
)


st.subheader("📋 Raw Data")
col1, col2, col3 = st.columns(3)
col1.metric("Total Months", len(df))
col2.metric("Min Passengers", df['Passengers'].min())
col3.metric("Max Passengers", df['Passengers'].max())

if st.checkbox("Show raw data table"):
    st.dataframe(df)


st.subheader("📈 Historical Data")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df.index, df['Passengers'],
         color='steelblue', linewidth=2)
ax1.set_title('Monthly Airline Passengers 1949-1960')
ax1.set_xlabel('Year')
ax1.set_ylabel('Passengers (thousands)')
ax1.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig1)


st.subheader("🤖 Training Prophet Model...")

df_prophet = df.reset_index()
df_prophet.columns = ['ds', 'y']

split_index = int(len(df_prophet) * 0.8)
train = df_prophet.iloc[:split_index]
test = df_prophet.iloc[split_index:]

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)
model.fit(train)
st.success("Model trained successfully!")


future = model.make_future_dataframe(
    periods=len(test) + forecast_months,
    freq='MS'
)
forecast = model.predict(future)


test_forecast = forecast.iloc[split_index:split_index+len(test)]
actual = test['y'].values
predicted = test_forecast['yhat'].values

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100


st.subheader("📐 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("MAPE", f"{mape:.2f}%")


st.subheader(f"🔮 Forecast for Next {forecast_months} Months")
fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(df.index, df['Passengers'],
         color='steelblue', linewidth=2, label='Historical data')

future_forecast = forecast.tail(forecast_months)
ax2.plot(future_forecast['ds'], future_forecast['yhat'],
         color='green', linewidth=2,
         linestyle='--', label='Forecast')
ax2.fill_between(future_forecast['ds'],
                 future_forecast['yhat_lower'],
                 future_forecast['yhat_upper'],
                 color='green', alpha=0.1,
                 label='Confidence interval')

ax2.set_title(f'Airline Passengers Forecast - Next {forecast_months} Months')
ax2.set_xlabel('Year')
ax2.set_ylabel('Passengers (thousands)')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)


st.subheader("📥 Download Forecast Results")
forecast_download = future_forecast[['ds', 'yhat',
                                     'yhat_lower', 'yhat_upper']].copy()
forecast_download.columns = ['Month', 'Forecast',
                              'Lower Bound', 'Upper Bound']
forecast_download['Month'] = forecast_download['Month'].dt.strftime('%Y-%m')
forecast_download = forecast_download.round(2)

st.dataframe(forecast_download)

csv = forecast_download.to_csv(index=False)
st.download_button(
    label="Download Forecast as CSV",
    data=csv,
    file_name='forecast_results.csv',
    mime='text/csv'
)

st.markdown("---")
st.markdown("Built with Prophet and Streamlit")