import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import warnings
import traceback

warnings.filterwarnings('ignore')

# Import our chatbot modules
from chatbot.core import MultiAssetAnalysisChatbot
from config import INTERNATIONAL_MARKETS

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

# Country selection
countries = list(INTERNATIONAL_MARKETS.keys())
selected_country = st.sidebar.selectbox(
    "Market/Country:",
    countries,
    index=0  # Default to US
)

# Asset type selection - updated to include International Stocks
asset_type = st.sidebar.selectbox(
    "Asset Type:",
    ["Stock", "International Stock", "Cryptocurrency", "Forex", "Commodity", "ETF", "Index", "Bond"]
)

# Dynamic symbol examples based on asset type and country
symbol_examples = {
    "Stock": INTERNATIONAL_MARKETS[selected_country]["examples"],
    "International Stock": INTERNATIONAL_MARKETS[selected_country]["examples"],
    "Cryptocurrency": "BTC-USD, ETH-USD, ADA-USD, BNB-USD",
    "Forex": "EURUSD=X, GBPUSD=X, USDJPY=X",
    "Commodity": "GC=F (Gold), SI=F (Silver), CL=F (Oil)",
    "ETF": "SPY, QQQ, VTI, GLD",
    "Index": "^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (Nasdaq)",
    "Bond": "^TNX (10Y Treasury), ^FVX (5Y Treasury)"
}


def get_default_symbol(asset_type, country):
    """Get default symbol based on asset type and country"""
    if asset_type in ["Stock", "International Stock"]:
        return INTERNATIONAL_MARKETS[country]["examples"][0]
    else:
        defaults = {
            "Cryptocurrency": "BTC-USD",
            "Forex": "EURUSD=X",
            "Commodity": "GC=F",
            "ETF": "SPY",
            "Index": "^GSPC",
            "Bond": "^TNX"
        }
        return defaults.get(asset_type, "AAPL")


# Symbol input with country-specific formatting
if asset_type in ["Stock", "International Stock"]:
    symbol = st.sidebar.selectbox(
        f"Symbol ({selected_country} Market):",
        INTERNATIONAL_MARKETS[selected_country]["examples"],
        index=0
    )
else:
    symbol = st.sidebar.text_input(
        f"Symbol ({symbol_examples[asset_type]}):",
        get_default_symbol(asset_type, selected_country)
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

# Market information
st.sidebar.markdown("---")
st.sidebar.markdown(f"**🌍 {selected_country} Market Info**")
st.sidebar.markdown(f"Currency: {INTERNATIONAL_MARKETS[selected_country]['currency']}")
st.sidebar.markdown(f"Suffix: {INTERNATIONAL_MARKETS[selected_country]['suffix']}")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Chat Interface")
    user_input = st.text_input(
        "Ask about asset analysis:",
        placeholder=f"e.g., Show technical analysis, Predict price, Fundamental data for {selected_country} market"
    )

    if user_input:
        with st.spinner(f"Analyzing {asset_type} data from {selected_country}..."):
            try:
                # Handle international stocks
                if asset_type == "International Stock":
                    actual_asset_type = "Stock"
                else:
                    actual_asset_type = asset_type

                response, charts = st.session_state.chatbot.process_query(
                    user_input, actual_asset_type, symbol, period, selected_country
                )

                st.markdown("### 🤖 Analysis Results")
                st.markdown(response)

                if charts:
                    st.markdown("### 📊 Charts")
                    for chart in charts:
                        st.plotly_chart(chart, use_container_width=True)

            except Exception as e:
                st.error(f"Error analyzing {asset_type}: {str(e)}")
                with st.expander("Debug Details"):
                    st.code(traceback.format_exc())

with col2:
    st.subheader("Quick Actions")

    if st.button("📈 Technical Analysis"):
        with st.spinner("Running technical analysis..."):
            try:
                actual_asset_type = "Stock" if asset_type == "International Stock" else asset_type
                analysis = st.session_state.chatbot.technical_analysis(
                    actual_asset_type, symbol, period, selected_country
                )
                st.markdown("### Technical Analysis")
                st.markdown(analysis)
            except Exception as e:
                st.error(f"Error in technical analysis: {str(e)}")

    if st.button("💰 Current Price"):
        with st.spinner("Fetching current price..."):
            try:
                actual_asset_type = "Stock" if asset_type == "International Stock" else asset_type
                data = st.session_state.chatbot.data_fetcher.get_asset_data(
                    actual_asset_type, symbol, "1d", selected_country
                )
                current_price = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100

                currency_symbol = INTERNATIONAL_MARKETS[selected_country]["currency"]

                st.metric(
                    label=f"{symbol} Current Price ({currency_symbol})",
                    value=f"{current_price:.2f}",
                    delta=f"{change:+.2f} ({change_pct:+.2f}%)"
                )
            except Exception as e:
                st.error(f"Error fetching price: {str(e)}")

    if st.button("📊 Fundamental Data"):
        with st.spinner("Fetching fundamental data..."):
            try:
                actual_asset_type = "Stock" if asset_type == "International Stock" else asset_type
                fundamental = st.session_state.chatbot.get_fundamental_data(
                    actual_asset_type, symbol, selected_country
                )
                st.markdown("### Fundamental Analysis")
                st.markdown(fundamental)
            except Exception as e:
                st.error(f"Error fetching fundamental data: {str(e)}")

    if st.button("🔮 Price Prediction"):
        with st.spinner("Generating prediction..."):
            try:
                actual_asset_type = "Stock" if asset_type == "International Stock" else asset_type
                prediction = st.session_state.chatbot.price_prediction(
                    actual_asset_type, symbol, selected_country
                )
                st.markdown("### Price Prediction")
                st.markdown(prediction)
            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")

    # Market-specific actions
    if asset_type in ["Stock", "International Stock"]:
        if st.button("🌍 Market Overview"):
            with st.spinner(f"Getting {selected_country} market overview..."):
                try:
                    overview = st.session_state.chatbot.get_market_overview(selected_country)
                    st.markdown(f"### {selected_country} Market Overview")
                    st.markdown(overview)
                except Exception as e:
                    st.error(f"Error fetching market overview: {str(e)}")

# Display recent data
st.sidebar.markdown("---")
if st.sidebar.button("Show Recent Data"):
    with st.spinner("Loading recent data..."):
        try:
            actual_asset_type = "Stock" if asset_type == "International Stock" else asset_type
            data = st.session_state.chatbot.data_fetcher.get_asset_data(
                actual_asset_type, symbol, "1mo", selected_country
            )
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