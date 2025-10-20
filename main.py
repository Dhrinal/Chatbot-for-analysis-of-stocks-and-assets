import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import warnings
import traceback

warnings.filterwarnings('ignore')

# Import our chatbot modules
from chatbot.core import MultiAssetAnalysisChatbot

# Page configuration
st.set_page_config(
    page_title="AI Multi-Asset Analysis Platform",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 AI Multi-Asset Analysis Platform")
st.markdown("Analyze stocks, crypto, forex, commodities, and more with AI-powered insights")

# Initialize chatbot in session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = MultiAssetAnalysisChatbot()

# Sidebar for inputs
st.sidebar.header("Asset Analysis Parameters")

# Asset type selection
asset_type = st.sidebar.selectbox(
    "Asset Type:",
    ["Stock", "Cryptocurrency", "Forex", "Commodity", "ETF", "Index", "Bond"]
)

# Dynamic symbol examples based on asset type
symbol_examples = {
    "Stock": "AAPL, TSLA, MSFT, GOOGL",
    "Cryptocurrency": "BTC-USD, ETH-USD, ADA-USD, BNB-USD",
    "Forex": "EURUSD=X, GBPUSD=X, USDJPY=X",
    "Commodity": "GC=F (Gold), SI=F (Silver), CL=F (Oil)",
    "ETF": "SPY, QQQ, VTI, GLD",
    "Index": "^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (Nasdaq)",
    "Bond": "^TNX (10Y Treasury), ^FVX (5Y Treasury)"
}


# Get default symbol based on asset type
def get_default_symbol(asset_type):
    defaults = {
        "Stock": "AAPL",
        "Cryptocurrency": "BTC-USD",
        "Forex": "EURUSD=X",
        "Commodity": "GC=F",
        "ETF": "SPY",
        "Index": "^GSPC",
        "Bond": "^TNX"
    }
    return defaults.get(asset_type, "AAPL")


symbol = st.sidebar.text_input(
    f"Symbol ({symbol_examples[asset_type]}):",
    get_default_symbol(asset_type)
)

# Enhanced period selection with daily options
period = st.sidebar.selectbox(
    "Time Period:",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=5
)

# Warning for short periods
if period in ["1d", "5d"]:
    st.sidebar.warning("⚠️ Short periods may not work well with technical indicators")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat Interface")
    user_input = st.text_input(
        "Ask about asset analysis:",
        placeholder="e.g., Show technical analysis, Predict price, Fundamental data"
    )

    if user_input:
        with st.spinner(f"Analyzing {asset_type} data..."):
            try:
                response, charts = st.session_state.chatbot.process_query(user_input, asset_type, symbol, period)

                st.markdown("### 🤖 Analysis Results")
                st.markdown(response)

                if charts:
                    st.markdown("### 📊 Charts")
                    for chart in charts:
                        st.plotly_chart(chart, use_container_width=True)

            except Exception as e:
                st.error(f"Error analyzing {asset_type}: {str(e)}")
                # Show more detailed error in expander for debugging
                with st.expander("Debug Details"):
                    st.code(traceback.format_exc())

with col2:
    st.subheader("Quick Actions")

    if st.button("📈 Technical Analysis"):
        with st.spinner("Running technical analysis..."):
            try:
                analysis = st.session_state.chatbot.technical_analysis(asset_type, symbol, period)
                st.markdown("### Technical Analysis")
                st.markdown(analysis)
            except Exception as e:
                st.error(f"Error in technical analysis: {str(e)}")

    if st.button("💰 Current Price"):
        with st.spinner("Fetching current price..."):
            try:
                data = st.session_state.chatbot.data_fetcher.get_asset_data(asset_type, symbol, "1d")
                current_price = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100

                # Simple price display without complex formatting
                st.metric(
                    label=f"{symbol} Current Price",
                    value=f"${current_price:.2f}",
                    delta=f"{change:+.2f} ({change_pct:+.2f}%)"
                )
            except Exception as e:
                st.error(f"Error fetching price: {str(e)}")

    if st.button("📊 Fundamental Data"):
        with st.spinner("Fetching fundamental data..."):
            try:
                fundamental = st.session_state.chatbot.get_fundamental_data(asset_type, symbol)
                st.markdown("### Fundamental Analysis")
                st.markdown(fundamental)
            except Exception as e:
                st.error(f"Error fetching fundamental data: {str(e)}")

    if st.button("🔮 Price Prediction"):
        with st.spinner("Generating prediction..."):
            try:
                prediction = st.session_state.chatbot.price_prediction(asset_type, symbol)
                st.markdown("### Price Prediction")
                st.markdown(prediction)
            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")

    # Asset-specific actions
    if asset_type == "Cryptocurrency":
        if st.button("₿ Crypto Metrics"):
            with st.spinner("Fetching crypto metrics..."):
                try:
                    metrics = st.session_state.chatbot.get_crypto_metrics(symbol)
                    st.markdown("### Cryptocurrency Metrics")
                    st.markdown(metrics)
                except Exception as e:
                    st.error(f"Error fetching crypto metrics: {str(e)}")

# Display recent data
st.sidebar.markdown("---")
if st.sidebar.button("Show Recent Data"):
    with st.spinner("Loading recent data..."):
        try:
            data = st.session_state.chatbot.data_fetcher.get_asset_data(asset_type, symbol, "1mo")
            st.sidebar.markdown(f"**Recent {symbol} Data (Last 5 days)**")
            display_data = data.tail()[['Open', 'High', 'Low', 'Close']]
            if 'Volume' in data.columns:
                display_data['Volume'] = data['Volume'].tail()

            st.sidebar.dataframe(display_data.style.format({
                'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}',
                'Close': '{:.2f}', 'Volume': '{:,.0f}'
            }))
        except Exception as e:
            st.sidebar.error(f"Error loading data: {str(e)}")