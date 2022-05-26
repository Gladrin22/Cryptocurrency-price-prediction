import streamlit as st                
from datetime import date 
import yfinance as yf

from fbprophet import Prophet 
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2010-01-01"    

TODAY = date.today().strftime("%Y-%m-%d")  


st.title('Cryptocurrency Price Prediction Using Machine Learning ')  
st.write('(Bitcoin , Ethereum , Binance Coin, USD Coin , Doge Coin , Shiba Inu,Lite Coin)')

cryptos = ('BTC-USD','ETH-USD','BNB_USD','USDC-USD','DOGE-USD','SHIB-USD','LTC-USD')  

selected_cryptos = st.selectbox('Select Dataset for prediction',cryptos)   

n_years = st.slider("Years of prediction :",1,10)   
period = n_years * 365

@st.cache

def load_data(ticker):                              
    data = yf.download(ticker,START , TODAY)
    data.reset_index(inplace = True)
    return data

data_load_state = st.text('Loading crypto Data...')         
data = load_data(selected_cryptos)
data_load_state.text('Done !!')

st.subheader('The Data:')
st.write(data.tail())

def plot_data():
    fig= go.Figure()
    fig.add_trace(go.scatter(x = data['Date'], y =data['Open'], name = 'crypto_Open'))         
    fig.add_trace(go.scatter(x = data['Date'], y =data['Close'], name = 'crypto_Close'))      
    fig.layout.update(title_text ="Time Series data with ranger slider",xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

    plot_data



data_frame_train = data [['Date','Close']]  
data_frame_train = data_frame_train.rename(columns={"Date":"ds","Close":"y"})

p = Prophet()
p.fit(data_frame_train)

future = p.make_future_dataframe(periods=period)
forecast = p.predict(future)

# Visualizing the output ....

st.subheader('Predicted output:')
st.write(forecast.tail())

st.subheader(f'The Forecast for {n_years} year :')
fig1 = plot_plotly(p, forecast)
st.plotly_chart(fig1)



st.write("Forecast components")
fig2 = p.plot_components(forecast)
st.write(fig2)



