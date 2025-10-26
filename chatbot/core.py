import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re
from .data_fetcher import MultiAssetDataFetcher
from .analysis import TechnicalAnalyzer, SentimentAnalyzer, PricePredictor
from config import INTERNATIONAL_MARKETS


class MultiAssetAnalysisChatbot:
    def __init__(self):
        self.data_fetcher = MultiAssetDataFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.price_predictor = PricePredictor()

        # Enhanced intent patterns for multi-asset
        self.intent_patterns = {
            'price': r'(price|current|how much|value|stock price|crypto price|exchange rate)',
            'technical': r'(technical|indicators|RSI|MACD|moving average|bollinger)',
            'sentiment': r'(sentiment|news|social media|opinion|mood|market sentiment)',
            'prediction': r'(predict|forecast|future price|where.*going|tomorrow|next week)',
            'fundamental': r'(fundamental|earnings|revenue|P/E|ratio|dividend|market cap|metrics)',
            'comparison': r'(compare|vs|versus|difference|better|worse)',
            'trend': r'(trend|performance|how.*doing|performance)',
            'volume': r'(volume|trading|liquidity)',
            'history': r'(history|historical|past performance)',
            'market': r'(market overview|country|economy|{})'.format('|'.join(INTERNATIONAL_MARKETS.keys()))
        }

    def process_query(self, query: str, asset_type: str, symbol: str, period: str = "1y", country: str = "US") -> Tuple[str, List]:
        """Process user query for any asset type with country support"""
        intent = self._classify_intent(query)
        charts = []

        try:
            # Fetch asset data with country information
            asset_data = self.data_fetcher.get_asset_data(asset_type, symbol, period, country)

            if intent == 'price':
                response = self._get_price_info(asset_data, asset_type, symbol, country)

            elif intent == 'technical':
                response, charts = self._get_technical_analysis(asset_data, asset_type, symbol, country)

            elif intent == 'sentiment':
                response = self._get_sentiment_analysis(asset_type, symbol, country)

            elif intent == 'prediction':
                response, charts = self._get_price_prediction(asset_data, asset_type, symbol, country)

            elif intent == 'fundamental':
                response = self._get_fundamental_analysis(asset_type, symbol, country)

            elif intent == 'trend':
                response, charts = self._get_trend_analysis(asset_data, asset_type, symbol, country)

            elif intent == 'volume':
                response = self._get_volume_analysis(asset_data, asset_type, symbol, country)

            elif intent == 'history':
                response = self._get_historical_analysis(asset_data, asset_type, symbol, period, country)

            elif intent == 'market':
                response = self._get_market_overview(country)

            else:
                response = self._get_general_analysis(asset_data, asset_type, symbol, country)

        except Exception as e:
            response = f"❌ Sorry, I encountered an error while analyzing {asset_type} {symbol} from {country}: {str(e)}\n\nPlease check the symbol and try again."

        return response, charts

    def _classify_intent(self, query: str) -> str:
        """Classify user intent from query"""
        query = query.lower()

        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query):
                return intent

        return 'general'

    def _get_price_info(self, data: pd.DataFrame, asset_type: str, symbol: str, country: str) -> str:
        """Get price information for any asset type with currency"""
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

        currency = INTERNATIONAL_MARKETS[country]["currency"]

        # Format based on currency
        if currency in ["JPY", "INR"]:
            current_price_str = f"{current_price:.0f}"
            prev_close_str = f"{prev_close:.0f}"
            change_str = f"{change:+.0f}"
        else:
            current_price_str = f"{current_price:.2f}"
            prev_close_str = f"{prev_close:.2f}"
            change_str = f"{change:+.2f}"

        # Asset-specific emojis and labels
        asset_emojis = {
            "Stock": "📊",
            "Cryptocurrency": "₿",
            "Forex": "💱",
            "Commodity": "🛢️",
            "ETF": "📈",
            "Index": "📊",
            "Bond": "📋"
        }

        emoji = asset_emojis.get(asset_type, "📊")

        return f"""
{emoji} **{symbol} ({asset_type}) - {country} Market**

💰 **Current Price:** {currency} {current_price_str}
📈 **Previous Close:** {currency} {prev_close_str}
🎯 **Change:** {change_str} ({change_pct:+.2f}%)
📅 **Last Updated:** {data.index[-1].strftime('%Y-%m-%d %H:%M') if hasattr(data.index[-1], 'strftime') else 'N/A'}
"""

    def _get_technical_analysis(self, data: pd.DataFrame, asset_type: str, symbol: str, country: str) -> Tuple[str, List]:
        """Get technical analysis for any asset"""
        analysis = self.technical_analyzer.analyze(data)
        charts = self.technical_analyzer.create_technical_charts(data, symbol)

        response = f"""
**🔧 Technical Analysis for {symbol} ({asset_type}) - {country} Market:**

{analysis}
        """

        return response, charts

    def _get_sentiment_analysis(self, asset_type: str, symbol: str, country: str) -> str:
        """Get sentiment analysis for any asset"""
        if asset_type == "Cryptocurrency":
            return self.sentiment_analyzer.analyze_crypto_sentiment(symbol)
        else:
            return self.sentiment_analyzer.analyze_international_sentiment(symbol, country)

    def _get_price_prediction(self, data: pd.DataFrame, asset_type: str, symbol: str, country: str) -> Tuple[str, List]:
        """Get price prediction for any asset"""
        prediction, forecast_chart = self.price_predictor.predict(data, symbol)
        return prediction, [forecast_chart]

    def _get_fundamental_analysis(self, asset_type: str, symbol: str, country: str) -> str:
        """Get fundamental analysis for any asset type"""
        return self.data_fetcher.get_fundamental_data(asset_type, symbol, country)

    def _get_trend_analysis(self, data: pd.DataFrame, asset_type: str, symbol: str, country: str) -> Tuple[str, List]:
        """Get trend analysis for any asset"""
        trend_response = self.technical_analyzer.generate_trend_report(data, symbol)
        trend_chart = self.technical_analyzer.create_technical_charts(data, symbol)
        return trend_response, trend_chart

    def _get_volume_analysis(self, data: pd.DataFrame, asset_type: str, symbol: str, country: str) -> str:
        """Get volume analysis for any asset"""
        return self.technical_analyzer._analyze_volume_trends(data)

    def _get_historical_analysis(self, data: pd.DataFrame, asset_type: str, symbol: str, period: str, country: str) -> str:
        """Get historical analysis for any asset"""
        return f"""
**📊 Historical Performance for {symbol} ({asset_type}) - {country} Market:**

**Period:** {period}
**Total Return:** {((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100:.2f}%
**Volatility:** {data['Close'].pct_change().std() * 100:.2f}%
**Highest Price:** ${data['High'].max():.2f}
**Lowest Price:** ${data['Low'].min():.2f}

*Analysis based on {len(data)} trading periods*
"""

    def _get_market_overview(self, country: str) -> str:
        """Get market overview for a specific country"""
        return self.data_fetcher.get_market_overview(country)

    def _get_general_analysis(self, data: pd.DataFrame, asset_type: str, symbol: str, country: str) -> str:
        """Get general analysis for any asset"""
        price_info = self._get_price_info(data, asset_type, symbol, country)
        trend_info = self._get_trend_analysis(data, asset_type, symbol, country)[0]

        return f"""
{price_info}

{trend_info}

**💡 Pro Tip:** Ask me about:
- 📈 "Technical analysis"
- 🔮 "Price prediction" 
- 📊 "Fundamental data"
- 😊 "Market sentiment"
- 📉 "Volume analysis"
- 🌍 "Market overview"
        """

    # Public methods for direct access
    def technical_analysis(self, asset_type: str, symbol: str, period: str = "1y", country: str = "US") -> str:
        data = self.data_fetcher.get_asset_data(asset_type, symbol, period, country)
        return self.technical_analyzer.analyze(data)

    def get_fundamental_data(self, asset_type: str, symbol: str, country: str = "US") -> str:
        return self.data_fetcher.get_fundamental_data(asset_type, symbol, country)

    def price_prediction(self, asset_type: str, symbol: str, country: str = "US") -> str:
        data = self.data_fetcher.get_asset_data(asset_type, symbol, "1y", country)
        prediction, _ = self.price_predictor.predict(data, symbol)
        return prediction

    def get_crypto_metrics(self, symbol: str) -> str:
        """Get cryptocurrency-specific metrics"""
        return self.data_fetcher.get_crypto_fundamentals(symbol)

    def get_market_overview(self, country: str) -> str:
        """Get market overview for a specific country"""
        return self.data_fetcher.get_market_overview(country)