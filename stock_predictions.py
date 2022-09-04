import streamlit as st
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = datetime.now() - relativedelta(years=10)
START = START.strftime("%Y-%m-%d")
TODAY = datetime.today().strftime("%Y-%m-%d")

st.title("Grier's Stock Prediciton")

stocks = ("AAPL","GOOG","MSFT","GME","DKNG")
selected_stock = st.selectbox("Select TICKR for Prediction", stocks)

n_years = st.slider("# of Days to Predict Ahead...", 1, 30)
period = n_years # * 365

@st.cache ##Caches the selected tickr
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True) #This will put the date in the very first column
    return data

data_load_state = st.text("Load Data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name = 'stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name = 'stock_close'))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting 
df_train = data[['Date', "Close"]]
df_train = df_train.rename(columns = {"Date":"ds", "Close":"y"})##how prophet is expecting data

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast[["ds", "yhat"]].rename(columns = {"ds":"Date", "yhat":"Predicted Value"}).set_index("Date").tail(n_years))

st.write("Forecast Data: {}".format(selected_stock))
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)


