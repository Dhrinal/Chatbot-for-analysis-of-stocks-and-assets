import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re
from .data_fetcher import MultiAssetDataFetcher
from .analysis import TechnicalAnalyzer, SentimentAnalyzer, PricePredictor


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
            'sentiment': r'(sentiment|news|social media|opinion|mood)',
            'prediction': r'(predict|forecast|future price|where.*going|tomorrow|next week)',
            'fundamental': r'(fundamental|earnings|revenue|P/E|ratio|dividend|market cap|metrics)',
            'comparison': r'(compare|vs|versus|difference|better)',
            'trend': r'(trend|performance|how.*doing|performance)',
            'volume': r'(volume|trading|liquidity)',
            'history': r'(history|historical|past performance)'
        }

    def process_query(self, query: str, asset_type: str, symbol: str, period: str = "1y") -> Tuple[str, List]:
        """Process user query for any asset type"""
        intent = self._classify_intent(query)
        charts = []

        try:
            # Fetch asset data
            asset_data = self.data_fetcher.get_asset_data(asset_type, symbol, period)

            if intent == 'price':
                response = self._get_price_info(asset_data, asset_type, symbol)

            elif intent == 'technical':
                response, charts = self._get_technical_analysis(asset_data, asset_type, symbol)

            elif intent == 'sentiment':
                response = self._get_sentiment_analysis(asset_type, symbol)

            elif intent == 'prediction':
                response, charts = self._get_price_prediction(asset_data, asset_type, symbol)

            elif intent == 'fundamental':
                response = self._get_fundamental_analysis(asset_type, symbol)

            elif intent == 'trend':
                response, charts = self._get_trend_analysis(asset_data, asset_type, symbol)

            elif intent == 'volume':
                response = self._get_volume_analysis(asset_data, asset_type, symbol)

            elif intent == 'history':
                response = self._get_historical_analysis(asset_data, asset_type, symbol, period)

            else:
                response = self._get_general_analysis(asset_data, asset_type, symbol)

        except Exception as e:
            response = f"❌ Sorry, I encountered an error while analyzing {asset_type} {symbol}: {str(e)}\n\nPlease check the symbol and try again."

        return response, charts

    def _classify_intent(self, query: str) -> str:
        """Classify user intent from query"""
        query = query.lower()

        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query):
                return intent

        return 'general'

    def _get_price_info(self, data: pd.DataFrame, asset_type: str, symbol: str) -> str:
        """Get price information for any asset type"""
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

        # Asset-specific formatting - FIXED SYNTAX
        if asset_type == "Forex":
            current_price_str = f"{current_price:.4f}"
            prev_close_str = f"{prev_close:.4f}"
            change_str = f"{change:+.4f}"
        else:
            current_price_str = f"${current_price:.2f}"
            prev_close_str = f"${prev_close:.2f}"
            change_str = f"${change:+.2f}"

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
{emoji} **{symbol} ({asset_type}) Current Price Information:**

💰 **Current Price:** {current_price_str}
📈 **Previous Close:** {prev_close_str}
🎯 **Change:** {change_str} ({change_pct:+.2f}%)
📅 **Last Updated:** {data.index[-1].strftime('%Y-%m-%d %H:%M') if hasattr(data.index[-1], 'strftime') else 'N/A'}
"""

    def _get_technical_analysis(self, data: pd.DataFrame, asset_type: str, symbol: str) -> Tuple[str, List]:
        """Get technical analysis for any asset"""
        analysis = self.technical_analyzer.analyze(data)
        charts = self.technical_analyzer.create_technical_charts(data, symbol)

        response = f"""
**🔧 Technical Analysis for {symbol} ({asset_type}):**

{analysis}
        """

        return response, charts

    def _get_sentiment_analysis(self, asset_type: str, symbol: str) -> str:
        """Get sentiment analysis for any asset"""
        if asset_type == "Cryptocurrency":
            return self.sentiment_analyzer.analyze_crypto_sentiment(symbol)
        else:
            return self.sentiment_analyzer.analyze_stock_sentiment(symbol)

    def _get_price_prediction(self, data: pd.DataFrame, asset_type: str, symbol: str) -> Tuple[str, List]:
        """Get price prediction for any asset"""
        prediction, forecast_chart = self.price_predictor.predict(data, symbol)
        return prediction, [forecast_chart]

    def _get_fundamental_analysis(self, asset_type: str, symbol: str) -> str:
        """Get fundamental analysis for any asset type"""
        return self.data_fetcher.get_fundamental_data(asset_type, symbol)

    def _get_trend_analysis(self, data: pd.DataFrame, asset_type: str, symbol: str) -> Tuple[str, List]:
        """Get trend analysis for any asset"""
        trend_response = self.technical_analyzer.trend_analysis(data, symbol)
        trend_chart = self.technical_analyzer.create_trend_chart(data, symbol)
        return trend_response, [trend_chart]

    def _get_volume_analysis(self, data: pd.DataFrame, asset_type: str, symbol: str) -> str:
        """Get volume analysis for any asset"""
        return self.technical_analyzer.volume_analysis(data, symbol)

    def _get_historical_analysis(self, data: pd.DataFrame, asset_type: str, symbol: str, period: str) -> str:
        """Get historical analysis for any asset"""
        return self.technical_analyzer.historical_analysis(data, symbol, period)

    def _get_general_analysis(self, data: pd.DataFrame, asset_type: str, symbol: str) -> str:
        """Get general analysis for any asset"""
        price_info = self._get_price_info(data, asset_type, symbol)
        trend_info = self._get_trend_analysis(data, asset_type, symbol)[0]

        return f"""
{price_info}

{trend_info}

**💡 Pro Tip:** Ask me about:
- 📈 "Technical analysis"
- 🔮 "Price prediction" 
- 📊 "Fundamental data"
- 😊 "Market sentiment"
- 📉 "Volume analysis"
        """

    # Public methods for direct access
    def technical_analysis(self, asset_type: str, symbol: str, period: str = "1y") -> str:
        data = self.data_fetcher.get_asset_data(asset_type, symbol, period)
        return self.technical_analyzer.analyze(data)

    def get_fundamental_data(self, asset_type: str, symbol: str) -> str:
        return self.data_fetcher.get_fundamental_data(asset_type, symbol)

    def price_prediction(self, asset_type: str, symbol: str) -> str:
        data = self.data_fetcher.get_asset_data(asset_type, symbol, "1y")
        prediction, _ = self.price_predictor.predict(data, symbol)
        return prediction

    def get_crypto_metrics(self, symbol: str) -> str:
        """Get cryptocurrency-specific metrics"""
        return self.data_fetcher.get_crypto_fundamentals(symbol)