import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from typing import Tuple, List, Dict
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import time  # Fixed: Added import for time module

from config import NEWS_API_KEY, COUNTRY_CODES, ALPHA_VANTAGE_API_KEY, FINNHUB_API_KEY


class TechnicalAnalyzer:
    """Analyzes stock patterns and trends using technical indicators"""

    def __init__(self):
        self.indicators = {}

    def analyze(self, price_data: pd.DataFrame) -> str:
        """Comprehensive technical analysis summary"""
        analysis_parts = []

        # Calculate all technical indicators
        enhanced_data = self._calculate_all_indicators(price_data)

        # Run various analyses
        analyses = [
            self._analyze_price_trend(enhanced_data),
            self._analyze_rsi_signal(enhanced_data),
            self._analyze_macd_signal(enhanced_data),
            self._analyze_moving_average_alignment(enhanced_data),
            self._identify_support_resistance_levels(enhanced_data),
            self._analyze_volume_trends(enhanced_data)
        ]

        return "\n".join(analyses)

    def _calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive set of technical indicators"""
        # Momentum indicators
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()

        # Trend indicators
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Histogram'] = macd.macd_diff()

        # Moving averages
        data['SMA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
        data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        data['EMA_12'] = ta.trend.EMAIndicator(data['Close'], window=12).ema_indicator()
        data['EMA_26'] = ta.trend.EMAIndicator(data['Close'], window=26).ema_indicator()

        # Volatility indicators
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['BB_Upper'] = bollinger.bollinger_hband()
        data['BB_Lower'] = bollinger.bollinger_lband()

        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()

        return data

    def _analyze_price_trend(self, data: pd.DataFrame) -> str:
        """Determine the current price trend direction"""
        recent_prices = data['Close'].tail(30)

        if len(recent_prices) < 2:
            return "**📈 Trend Analysis:** Need more data for trend analysis"

        price_change_percent = ((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]) * 100

        if price_change_percent > 5:
            trend_strength = "Strong Uptrend 📈"
        elif price_change_percent > 0:
            trend_strength = "Mild Uptrend ↗️"
        elif price_change_percent < -5:
            trend_strength = "Strong Downtrend 📉"
        else:
            trend_strength = "Mild Downtrend ↘️"

        return f"**📈 Trend Analysis:** {trend_strength} ({price_change_percent:+.2f}% over last 30 days)"

    def _analyze_rsi_signal(self, data: pd.DataFrame) -> str:
        """Analyze RSI for overbought/oversold conditions"""
        if 'RSI' not in data.columns or data['RSI'].isna().all():
            return "**📊 RSI:** Data not available"

        current_rsi = data['RSI'].iloc[-1]

        if current_rsi > 70:
            rsi_signal = "Overbought ⚠️ (Consider selling)"
        elif current_rsi < 30:
            rsi_signal = "Oversold 💡 (Potential buying opportunity)"
        else:
            rsi_signal = "Neutral ↔️ (Normal trading range)"

        return f"**📊 RSI ({current_rsi:.1f}):** {rsi_signal}"

    def _analyze_macd_signal(self, data: pd.DataFrame) -> str:
        """Analyze MACD momentum signal"""
        if 'MACD' not in data.columns or data['MACD'].isna().all():
            return "**🔍 MACD:** Data not available"

        current_macd = data['MACD'].iloc[-1]
        current_signal = data['MACD_Signal'].iloc[-1]

        if current_macd > current_signal:
            momentum = "Bullish 🟢 (Positive momentum)"
        else:
            momentum = "Bearish 🔴 (Negative momentum)"

        return f"**🔍 MACD:** {momentum}"

    def _analyze_moving_average_alignment(self, data: pd.DataFrame) -> str:
        """Check moving average alignment for trend confirmation"""
        if 'SMA_20' not in data.columns or data['SMA_20'].isna().all():
            return "**📏 Moving Averages:** Data not available"

        current_price = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]

        if current_price > sma_20 > sma_50:
            alignment = "Strong Bullish Alignment 🟢"
        elif current_price < sma_20 < sma_50:
            alignment = "Strong Bearish Alignment 🔴"
        else:
            alignment = "Mixed Signals ↔️"

        return f"**📏 Moving Averages:** {alignment}"

    def _identify_support_resistance_levels(self, data: pd.DataFrame) -> str:
        """Identify key support and resistance levels"""
        recent_high = data['High'].tail(20).max()
        recent_low = data['Low'].tail(20).min()
        current_price = data['Close'].iloc[-1]

        resistance_distance = ((recent_high - current_price) / current_price) * 100
        support_distance = ((current_price - recent_low) / current_price) * 100

        return f"**🎯 Support/Resistance:** Resistance: ${recent_high:.2f} ({resistance_distance:+.1f}% above), Support: ${recent_low:.2f} ({support_distance:+.1f}% below)"

    def _analyze_volume_trends(self, data: pd.DataFrame) -> str:
        """Analyze trading volume patterns"""
        if 'Volume' not in data.columns:
            return "**📦 Volume:** Data not available"

        current_volume = data['Volume'].iloc[-1]
        average_volume = data['Volume'].mean()
        volume_ratio = current_volume / average_volume if average_volume > 0 else 1

        if volume_ratio > 1.5:
            volume_signal = "High volume 📈 (Strong interest)"
        elif volume_ratio < 0.5:
            volume_signal = "Low volume 📉 (Weak interest)"
        else:
            volume_signal = "Normal volume ↔️"

        return f"**📦 Volume:** {volume_signal} ({volume_ratio:.1f}x average)"

    def create_technical_charts(self, data: pd.DataFrame, symbol: str) -> List[go.Figure]:
        """Generate interactive technical analysis charts"""
        enhanced_data = self._calculate_all_indicators(data)
        charts = []

        # Price chart with moving averages
        price_chart = self._create_price_chart(enhanced_data, symbol)
        charts.append(price_chart)

        # RSI chart if available
        if 'RSI' in enhanced_data.columns and not enhanced_data['RSI'].isna().all():
            rsi_chart = self._create_rsi_chart(enhanced_data)
            charts.append(rsi_chart)

        return charts

    def _create_price_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create candlestick chart with moving averages"""
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))

        # Moving averages
        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_20'],
            line=dict(color='orange', width=1),
            name='SMA 20'
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_50'],
            line=dict(color='red', width=1),
            name='SMA 50'
        ))

        fig.update_layout(
            title=f'{symbol} Price with Moving Averages',
            yaxis_title='Price ($)',
            xaxis_title='Date'
        )

        return fig

    def _create_rsi_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create RSI indicator chart"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            line=dict(color='purple', width=2),
            name='RSI'
        ))
        # Overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")

        fig.update_layout(
            title='RSI Indicator',
            yaxis_title='RSI Value',
            xaxis_title='Date'
        )

        return fig

    def generate_trend_report(self, data: pd.DataFrame, symbol: str) -> str:
        """Generate comprehensive trend analysis report"""
        enhanced_data = self._calculate_all_indicators(data)

        if len(enhanced_data) < 2:
            return "Insufficient data for trend analysis"

        # Calculate returns for different time periods
        daily_return = self._calculate_period_return(enhanced_data, 1)
        weekly_return = self._calculate_period_return(enhanced_data, 5)
        monthly_return = self._calculate_period_return(enhanced_data, 21)

        volatility = enhanced_data['Close'].pct_change().std() * np.sqrt(252) * 100

        report = f"""
**📈 Trend Analysis for {symbol}:**

**📊 Performance:**
- 1 Day: {daily_return:+.2f}%
- 1 Week: {weekly_return:+.2f}%
- 1 Month: {monthly_return:+.2f}%

**📉 Volatility:** {volatility:.1f}% (annualized)

**🔍 Current Technical Position:**
{self.analyze(enhanced_data)}
        """

        return report

    def _calculate_period_return(self, data: pd.DataFrame, days: int) -> float:
        """Calculate return for specified number of days"""
        if len(data) <= days:
            return 0.0

        period_data = data.tail(days + 1)
        start_price = period_data['Close'].iloc[0]
        end_price = period_data['Close'].iloc[-1]

        return ((end_price - start_price) / start_price) * 100


class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.news_cache = {}  # Cache news to avoid repeated API calls
        self.cache_timeout = 300  # 5 minutes

    def analyze_with_real_news(self, symbol: str, asset_type: str = "Stock", country: str = "US") -> str:
        """Analyze sentiment using real news data"""
        try:
            # Get real news headlines
            headlines = self._fetch_real_news_headlines(symbol, asset_type, country)

            if not headlines:
                return self._fallback_to_sample_analysis(symbol, asset_type, country)

            return self._analyze_news_sentiment(headlines, symbol, asset_type, country)

        except Exception as e:
            print(f"Error in real news analysis: {e}")
            return self._fallback_to_sample_analysis(symbol, asset_type, country)

    def _fetch_real_news_headlines(self, symbol: str, asset_type: str, country: str) -> List[str]:
        """Fetch real news headlines from various APIs"""
        cache_key = f"{symbol}_{asset_type}_{country}"

        # Check cache first
        if cache_key in self.news_cache:
            cached_data, timestamp = self.news_cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data

        headlines = []

        try:
            # Try NewsAPI first
            newsapi_headlines = self._fetch_from_newsapi(symbol, country)
            if newsapi_headlines:
                headlines.extend(newsapi_headlines)

            # Try Alpha Vantage as backup
            if len(headlines) < 5:
                alpha_headlines = self._fetch_from_alphavantage(symbol)
                if alpha_headlines:
                    headlines.extend(alpha_headlines)

            # Try Finnhub as third option
            if len(headlines) < 3:
                finnhub_headlines = self._fetch_from_finnhub(symbol)
                if finnhub_headlines:
                    headlines.extend(finnhub_headlines)

            # Cache the results
            if headlines:
                self.news_cache[cache_key] = (headlines, time.time())

        except Exception as e:
            print(f"Error fetching news: {e}")

        return headlines

    def _fetch_from_newsapi(self, symbol: str, country: str) -> List[str]:
        """Fetch news from NewsAPI"""
        try:
            # Remove exchange suffixes for better search
            clean_symbol = self._clean_symbol(symbol)
            country_code = COUNTRY_CODES.get(country, "us")

            params = {
                'q': clean_symbol,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10,
                'apiKey': NEWS_API_KEY
            }

            response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                headlines = [article['title'] for article in data.get('articles', [])]
                return headlines[:8]  # Return top 8 headlines

        except Exception as e:
            print(f"NewsAPI error: {e}")

        return []

    def _fetch_from_alphavantage(self, symbol: str) -> List[str]:
        """Fetch news from Alpha Vantage"""
        try:
            clean_symbol = self._clean_symbol(symbol)

            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': clean_symbol,
                'apikey': ALPHA_VANTAGE_API_KEY,
                'limit': 10
            }

            response = requests.get("https://www.alphavantage.co/query", params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                headlines = [item['title'] for item in data.get('feed', [])]
                return headlines[:6]

        except Exception as e:
            print(f"Alpha Vantage news error: {e}")

        return []

    def _fetch_from_finnhub(self, symbol: str) -> List[str]:
        """Fetch news from Finnhub"""
        try:
            clean_symbol = self._clean_symbol(symbol)

            params = {
                'symbol': clean_symbol,
                'token': FINNHUB_API_KEY
            }

            response = requests.get("https://finnhub.io/api/v1/company-news", params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                headlines = [item['headline'] for item in data if item.get('headline')]
                return headlines[:5]

        except Exception as e:
            print(f"Finnhub news error: {e}")

        return []

    def _clean_symbol(self, symbol: str) -> str:
        """Remove exchange suffixes from symbols"""
        # Remove common exchange suffixes
        suffixes = ['.L', '.DE', '.PA', '.T', '.TO', '.AX', '.HK', '.NS', '.SA']
        clean_symbol = symbol
        for suffix in suffixes:
            if symbol.endswith(suffix):
                clean_symbol = symbol.replace(suffix, '')
                break
        return clean_symbol

    def _analyze_news_sentiment(self, headlines: List[str], symbol: str, asset_type: str, country: str) -> str:
        """Analyze sentiment from real news headlines"""
        if not headlines:
            return self._fallback_to_sample_analysis(symbol, asset_type, country)

        vader_scores = []
        textblob_scores = []

        for headline in headlines:
            # VADER sentiment
            vader_score = self.vader_analyzer.polarity_scores(headline)['compound']
            vader_scores.append(vader_score)

            # TextBlob sentiment
            blob = TextBlob(headline)
            textblob_scores.append(blob.sentiment.polarity)

        avg_vader = np.mean(vader_scores)
        avg_textblob = np.mean(textblob_scores)

        sentiment_result = self._determine_overall_sentiment(avg_vader, avg_textblob)

        return self._format_real_news_report(sentiment_result, avg_vader, avg_textblob,
                                             symbol, asset_type, country, headlines)

    def _determine_overall_sentiment(self, vader_score: float, textblob_score: float) -> Dict:
        """Determine overall sentiment from scores"""
        if vader_score > 0.05 and textblob_score > 0:
            return {"sentiment": "Bullish", "emoji": "📈", "color": "🟢"}
        elif vader_score < -0.05 and textblob_score < 0:
            return {"sentiment": "Bearish", "emoji": "📉", "color": "🔴"}
        else:
            return {"sentiment": "Neutral", "emoji": "↔️", "color": "🟡"}

    def _format_real_news_report(self, sentiment: Dict, vader_score: float, textblob_score: float,
                                 symbol: str, asset_type: str, country: str, headlines: List[str]) -> str:
        """Format real news sentiment analysis report"""

        # Sample of recent headlines (show 3-4)
        sample_headlines = headlines[:4]
        headlines_text = "\n".join([f"• {headline}" for headline in sample_headlines])

        return f"""
**📰 Real-Time Sentiment Analysis for {symbol} ({asset_type}) - {country} Market**

{sentiment['color']} **Overall Sentiment:** {sentiment['sentiment']} {sentiment['emoji']}
📊 **VADER Score:** {vader_score:.3f} ({self._interpret_sentiment_score(vader_score)})
📈 **TextBlob Score:** {textblob_score:.3f}

**📋 Recent News Headlines:**
{headlines_text}

**📊 Analysis Details:**
- Analyzed {len(headlines)} recent news articles
- Combined VADER and TextBlob sentiment analysis
- Real-time market sentiment assessment

**💡 Interpretation:**
- Positive scores (> 0.05): Bullish sentiment
- Negative scores (< -0.05): Bearish sentiment  
- Neutral scores: Mixed or uncertain sentiment

*Data sourced from multiple news APIs. Sentiment analysis updated in real-time.*
        """

    def _interpret_sentiment_score(self, score: float) -> str:
        """Convert numerical score to descriptive term"""
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def _fallback_to_sample_analysis(self, symbol: str, asset_type: str, country: str) -> str:
        """Fallback to sample analysis when real news is unavailable"""
        if asset_type == "Cryptocurrency":
            return self.analyze_crypto_sentiment(symbol)
        else:
            return self.analyze_international_sentiment(symbol, country)

    # Update existing methods to use real news
    def analyze_stock_sentiment(self, symbol: str, use_real_news: bool = True) -> str:
        """Analyze stock sentiment with real news option"""
        if use_real_news:
            return self.analyze_with_real_news(symbol, "Stock", "US")
        else:
            return self._analyze_sentiment(self._generate_sample_headlines(symbol), symbol, "Stock")

    def analyze_crypto_sentiment(self, symbol: str, use_real_news: bool = True) -> str:
        """Analyze crypto sentiment with real news option"""
        if use_real_news:
            return self.analyze_with_real_news(symbol, "Cryptocurrency", "US")
        else:
            crypto_headlines = [
                f"{symbol} shows strong adoption growth",
                f"Regulatory concerns for {symbol}",
                f"Institutional investors buying {symbol}",
                f"Network activity increases for {symbol}",
                f"Market volatility affects {symbol} price"
            ]
            return self._analyze_sentiment(crypto_headlines, symbol, "Crypto")

    def analyze_international_sentiment(self, symbol: str, country: str, use_real_news: bool = True) -> str:
        """Analyze international sentiment with real news option"""
        if use_real_news:
            return self.analyze_with_real_news(symbol, "Stock", country)
        else:
            headlines = self._get_country_specific_headlines(symbol, country)
            return self._analyze_sentiment(headlines, symbol, f"{country} Market")

    def _analyze_sentiment(self, headlines: List[str], symbol: str, analysis_type: str) -> str:
        """Core sentiment analysis logic"""
        vader_scores = []
        textblob_scores = []

        for headline in headlines:
            # VADER sentiment
            vader_score = self.vader_analyzer.polarity_scores(headline)['compound']
            vader_scores.append(vader_score)

            # TextBlob sentiment
            blob = TextBlob(headline)
            textblob_scores.append(blob.sentiment.polarity)

        avg_vader = np.mean(vader_scores)
        avg_textblob = np.mean(textblob_scores)

        sentiment_result = self._determine_overall_sentiment(avg_vader, avg_textblob)

        return self._format_sentiment_report(sentiment_result, avg_vader, avg_textblob, symbol, analysis_type)

    def _format_sentiment_report(self, sentiment: Dict, vader_score: float,
                                 textblob_score: float, symbol: str, analysis_type: str) -> str:
        """Format sentiment analysis results into readable report"""
        return f"""
**😊 {analysis_type} Sentiment Analysis for {symbol}:**

{sentiment['color']} **Overall Sentiment:** {sentiment['sentiment']} {sentiment['emoji']}
📊 **VADER Score:** {vader_score:.3f} ({self._interpret_sentiment_score(vader_score)})
📈 **TextBlob Score:** {textblob_score:.3f}

**💡 Analysis Notes:**
- Based on market sentiment indicators
- Combining multiple analysis methods
- Reflects general market mood

*Note: For real-time sentiment, integrate with news APIs*
        """

    def _generate_sample_headlines(self, symbol: str) -> List[str]:
        """Generate sample headlines for analysis"""
        return [
            f"{symbol} reports strong earnings growth",
            f"Investors optimistic about {symbol} future prospects",
            f"Market analysts recommend buying {symbol}",
            f"{symbol} faces competition in the market",
            f"{symbol} announces new product launch"
        ]

    def _get_country_specific_headlines(self, symbol: str, country: str) -> List[str]:
        """Get country-specific sample headlines"""
        country_templates = {
            "UK": [
                f"{symbol} shows strong performance in London market",
                f"Brexit impact on {symbol} remains uncertain",
                f"{symbol} announces dividend increase",
            ],
            "Germany": [
                f"{symbol} leads DAX performance",
                f"European market trends favor {symbol}",
                f"{symbol} expands operations in EU",
            ],
            "Japan": [
                f"{symbol} benefits from BoJ policies",
                f"Asian markets show confidence in {symbol}",
                f"{symbol} reports strong export numbers",
            ],
        }

        return country_templates.get(country, self._generate_sample_headlines(symbol))


class PricePredictor:
    """Provides simple price predictions based on trends"""

    def predict(self, data: pd.DataFrame, symbol: str) -> Tuple[str, go.Figure]:
        """Generate price prediction and visualization"""
        if len(data) < 10:
            return "Insufficient data for prediction", go.Figure()

        # Calculate recent trend
        recent_trend = data['Close'].tail(10).pct_change().mean()
        current_price = data['Close'].iloc[-1]

        # Generate 5-day forecast
        forecast_prices = self._generate_forecast(current_price, recent_trend, days=5)
        forecast_chart = self._create_forecast_chart(data, forecast_prices, symbol)

        prediction_text = self._format_prediction(
            symbol, recent_trend, current_price, forecast_prices[-1]
        )

        return prediction_text, forecast_chart

    def _generate_forecast(self, current_price: float, daily_trend: float, days: int) -> List[float]:
        """Generate price forecast based on trend"""
        forecast = []
        last_price = current_price

        for _ in range(days):
            next_price = last_price * (1 + daily_trend)
            forecast.append(next_price)
            last_price = next_price

        return forecast

    def _create_forecast_chart(self, data: pd.DataFrame, forecast: List[float], symbol: str) -> go.Figure:
        """Create visualization of price forecast"""
        fig = go.Figure()

        # Historical data (last 30 days)
        historical_data = data.tail(30)
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue', width=2)
        ))

        # Forecast data
        future_dates = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=len(forecast)
        )

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast,
            mode='lines+markers',
            name='Forecasted Prices',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title=f'{symbol} Price Forecast',
            xaxis_title='Date',
            yaxis_title='Price ($)'
        )

        return fig

    def _format_prediction(self, symbol: str, trend: float,
                           current_price: float, forecast_price: float) -> str:
        """Format prediction results"""
        direction = "up" if trend > 0 else "down"
        confidence = min(abs(trend * 1000), 80)  # Simple confidence metric

        return f"""
**🔮 Price Prediction for {symbol}:**

📈 **Short-term Trend:** {direction.capitalize()}
📊 **Expected Daily Change:** {trend * 100:+.3f}%
🎯 **Confidence Level:** {confidence:.1f}%
💰 **Current Price:** ${current_price:.2f}
📅 **5-Day Forecast:** ${forecast_price:.2f}

**⚠️ Important:** This is a simple trend-based estimate. 
Always conduct thorough research before making investment decisions.
        """