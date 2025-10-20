import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from typing import Tuple, List, Dict
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests


class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}

    def analyze(self, data: pd.DataFrame) -> str:
        """Perform comprehensive technical analysis"""
        analysis = []

        # Calculate indicators
        data = self._calculate_indicators(data)

        # Price trend analysis
        trend = self._analyze_trend(data)
        analysis.append(trend)

        # RSI analysis
        rsi_signal = self._analyze_rsi(data)
        analysis.append(rsi_signal)

        # MACD analysis
        macd_signal = self._analyze_macd(data)
        analysis.append(macd_signal)

        # Moving averages analysis
        ma_signal = self._analyze_moving_averages(data)
        analysis.append(ma_signal)

        # Support and resistance
        support_resistance = self._analyze_support_resistance(data)
        analysis.append(support_resistance)

        # Volume analysis
        volume_analysis = self._analyze_volume(data)
        analysis.append(volume_analysis)

        return "\n".join(analysis)

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # RSI
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()

        # MACD
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Histogram'] = macd.macd_diff()

        # Moving Averages
        data['SMA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
        data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        data['EMA_12'] = ta.trend.EMAIndicator(data['Close'], window=12).ema_indicator()
        data['EMA_26'] = ta.trend.EMAIndicator(data['Close'], window=26).ema_indicator()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['BB_Upper'] = bollinger.bollinger_hband()
        data['BB_Lower'] = bollinger.bollinger_lband()

        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()

        return data

    def _analyze_trend(self, data: pd.DataFrame) -> str:
        """Analyze price trend"""
        recent_prices = data['Close'][-30:]
        if len(recent_prices) < 2:
            return "**📈 Trend Analysis:** Insufficient data for trend analysis"

        price_change = ((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]) * 100

        if price_change > 5:
            trend = "Strong Uptrend 📈"
        elif price_change > 0:
            trend = "Mild Uptrend ↗️"
        elif price_change < -5:
            trend = "Strong Downtrend 📉"
        else:
            trend = "Mild Downtrend ↘️"

        return f"**📈 Trend Analysis:** {trend} ({price_change:+.2f}% over last 30 days)"

    def _analyze_rsi(self, data: pd.DataFrame) -> str:
        """Analyze RSI indicator"""
        if 'RSI' not in data.columns or data['RSI'].isna().all():
            return "**📊 RSI:** Data not available"

        current_rsi = data['RSI'].iloc[-1]

        if current_rsi > 70:
            signal = "Overbought ⚠️ (Consider selling)"
        elif current_rsi < 30:
            signal = "Oversold 💡 (Potential buying opportunity)"
        else:
            signal = "Neutral ↔️ (Normal trading range)"

        return f"**📊 RSI ({current_rsi:.1f}):** {signal}"

    def _analyze_macd(self, data: pd.DataFrame) -> str:
        """Analyze MACD indicator"""
        if 'MACD' not in data.columns or data['MACD'].isna().all():
            return "**🔍 MACD:** Data not available"

        macd = data['MACD'].iloc[-1]
        signal = data['MACD_Signal'].iloc[-1]

        if macd > signal:
            macd_signal = "Bullish 🟢 (Positive momentum)"
        else:
            macd_signal = "Bearish 🔴 (Negative momentum)"

        return f"**🔍 MACD:** {macd_signal}"

    def _analyze_moving_averages(self, data: pd.DataFrame) -> str:
        """Analyze moving averages"""
        if 'SMA_20' not in data.columns or data['SMA_20'].isna().all():
            return "**📏 Moving Averages:** Data not available"

        price = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]

        if price > sma_20 > sma_50:
            ma_signal = "Strong Bullish Alignment 🟢"
        elif price < sma_20 < sma_50:
            ma_signal = "Strong Bearish Alignment 🔴"
        else:
            ma_signal = "Mixed Signals ↔️"

        return f"**📏 Moving Averages:** {ma_signal}"

    def _analyze_support_resistance(self, data: pd.DataFrame) -> str:
        """Identify support and resistance levels"""
        recent_high = data['High'][-20:].max()
        recent_low = data['Low'][-20:].min()
        current_price = data['Close'].iloc[-1]

        resistance_distance = ((recent_high - current_price) / current_price) * 100
        support_distance = ((current_price - recent_low) / current_price) * 100

        return f"**🎯 Support/Resistance:** Resistance: ${recent_high:.2f} ({resistance_distance:+.1f}% above), Support: ${recent_low:.2f} ({support_distance:+.1f}% below)"

    def _analyze_volume(self, data: pd.DataFrame) -> str:
        """Analyze volume trends"""
        if 'Volume' not in data.columns:
            return "**📦 Volume:** Data not available"

        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].mean()

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        if volume_ratio > 1.5:
            volume_signal = "High volume 📈 (Strong interest)"
        elif volume_ratio < 0.5:
            volume_signal = "Low volume 📉 (Weak interest)"
        else:
            volume_signal = "Normal volume ↔️"

        return f"**📦 Volume:** {volume_signal} ({volume_ratio:.1f}x average)"

    def create_technical_charts(self, data: pd.DataFrame, symbol: str) -> List[go.Figure]:
        """Create technical analysis charts"""
        data = self._calculate_indicators(data)

        charts = []

        # Price with Moving Averages
        fig1 = go.Figure()
        fig1.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
        fig1.add_trace(go.Scatter(x=data.index, y=data['SMA_20'],
                                  line=dict(color='orange', width=1),
                                  name='SMA 20'))
        fig1.add_trace(go.Scatter(x=data.index, y=data['SMA_50'],
                                  line=dict(color='red', width=1),
                                  name='SMA 50'))
        fig1.update_layout(
            title=f'{symbol} Price with Moving Averages',
            yaxis_title='Price ($)',
            xaxis_title='Date'
        )
        charts.append(fig1)

        # RSI Chart
        if 'RSI' in data.columns and not data['RSI'].isna().all():
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'],
                                      line=dict(color='purple', width=2),
                                      name='RSI'))
            fig2.add_hline(y=70, line_dash="dash", line_color="red")
            fig2.add_hline(y=30, line_dash="dash", line_color="green")
            fig2.update_layout(
                title='RSI Indicator',
                yaxis_title='RSI Value',
                xaxis_title='Date'
            )
            charts.append(fig2)

        return charts

    def create_trend_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create trend analysis chart"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title=f'{symbol} Price Trend',
            yaxis_title='Price ($)',
            xaxis_title='Date'
        )
        return fig

    def trend_analysis(self, data: pd.DataFrame, symbol: str) -> str:
        """Comprehensive trend analysis"""
        data = self._calculate_indicators(data)

        if len(data) < 2:
            return "Insufficient data for trend analysis"

        # Calculate various period returns
        returns_1d = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100 if len(
            data) > 1 else 0

        weekly_data = data.iloc[-min(6, len(data)):]
        returns_1w = (weekly_data['Close'].iloc[-1] - weekly_data['Close'].iloc[0]) / weekly_data['Close'].iloc[
            0] * 100 if len(weekly_data) > 1 else 0

        monthly_data = data.iloc[-min(22, len(data)):]
        returns_1m = (monthly_data['Close'].iloc[-1] - monthly_data['Close'].iloc[0]) / monthly_data['Close'].iloc[
            0] * 100 if len(monthly_data) > 1 else 0

        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized

        analysis = f"""
**📈 Trend Analysis for {symbol}:**

**📊 Performance:**
- 1 Day: {returns_1d:+.2f}%
- 1 Week: {returns_1w:+.2f}%
- 1 Month: {returns_1m:+.2f}%

**📉 Volatility:** {volatility:.1f}% (annualized)

**🔍 Current Technical Position:**
{self.analyze(data)}
        """

        return analysis

    def volume_analysis(self, data: pd.DataFrame, symbol: str) -> str:
        """Volume analysis"""
        if 'Volume' not in data.columns:
            return "Volume data not available"

        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].mean()
        volume_trend = "increasing" if data['Volume'].iloc[-1] > data['Volume'].iloc[-5] else "decreasing"

        return f"""
**📦 Volume Analysis for {symbol}:**

- **Current Volume:** {current_volume:,.0f}
- **Average Volume:** {avg_volume:,.0f}
- **Volume Trend:** {volume_trend}
- **Volume vs Average:** {current_volume / avg_volume:.1f}x
        """

    def historical_analysis(self, data: pd.DataFrame, symbol: str, period: str) -> str:
        """Historical performance analysis"""
        if len(data) < 2:
            return "Insufficient historical data"

        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        total_return = ((end_price - start_price) / start_price) * 100
        max_price = data['Close'].max()
        min_price = data['Close'].min()

        return f"""
**📜 Historical Analysis for {symbol} ({period}):**

- **Starting Price:** ${start_price:.2f}
- **Current Price:** ${end_price:.2f}
- **Total Return:** {total_return:+.2f}%
- **Highest Price:** ${max_price:.2f}
- **Lowest Price:** ${min_price:.2f}
- **Price Range:** ${min_price:.2f} - ${max_price:.2f}
        """


class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def analyze_stock_sentiment(self, symbol: str) -> str:
        """Analyze sentiment for a stock using VADER and TextBlob"""
        # Sample news headlines (in real application, fetch from news API)
        sample_headlines = [
            f"{symbol} reports strong earnings growth",
            f"Investors optimistic about {symbol} future prospects",
            f"Market analysts recommend buying {symbol}",
            f"{symbol} faces competition in the market",
            f"{symbol} announces new product launch"
        ]

        # Analyze sentiment using both VADER and TextBlob
        vader_scores = []
        textblob_scores = []

        for headline in sample_headlines:
            # VADER sentiment
            vader_score = self.vader_analyzer.polarity_scores(headline)['compound']
            vader_scores.append(vader_score)

            # TextBlob sentiment
            blob = TextBlob(headline)
            textblob_scores.append(blob.sentiment.polarity)

        avg_vader = np.mean(vader_scores)
        avg_textblob = np.mean(textblob_scores)

        # Determine overall sentiment
        if avg_vader > 0.05 and avg_textblob > 0:
            overall_sentiment = "Bullish 😊"
            sentiment_color = "🟢"
        elif avg_vader < -0.05 and avg_textblob < 0:
            overall_sentiment = "Bearish 😞"
            sentiment_color = "🔴"
        else:
            overall_sentiment = "Neutral 😐"
            sentiment_color = "🟡"

        return f"""
**😊 Sentiment Analysis for {symbol}:**

{sentiment_color} **Overall Sentiment:** {overall_sentiment}
📊 **VADER Score:** {avg_vader:.3f} ({self._interpret_vader_score(avg_vader)})
📈 **TextBlob Score:** {avg_textblob:.3f}

**💡 Sample Analysis:**
- Based on market sentiment indicators
- Combining VADER and TextBlob analysis
- Reflects general market mood

*Note: For real-time sentiment, integrate with news APIs like NewsAPI or Alpha Vantage*
        """

    def analyze_crypto_sentiment(self, symbol: str) -> str:
        """Analyze sentiment for cryptocurrency"""
        # Sample crypto-specific headlines
        crypto_headlines = [
            f"{symbol} shows strong adoption growth",
            f"Regulatory concerns for {symbol}",
            f"Institutional investors buying {symbol}",
            f"Network activity increases for {symbol}",
            f"Market volatility affects {symbol} price"
        ]

        # Analyze sentiment
        vader_scores = []
        textblob_scores = []

        for headline in crypto_headlines:
            vader_score = self.vader_analyzer.polarity_scores(headline)['compound']
            vader_scores.append(vader_score)

            blob = TextBlob(headline)
            textblob_scores.append(blob.sentiment.polarity)

        avg_vader = np.mean(vader_scores)
        avg_textblob = np.mean(textblob_scores)

        # Determine overall sentiment
        if avg_vader > 0.05 and avg_textblob > 0:
            overall_sentiment = "Bullish 🚀"
            sentiment_color = "🟢"
        elif avg_vader < -0.05 and avg_textblob < 0:
            overall_sentiment = "Bearish 📉"
            sentiment_color = "🔴"
        else:
            overall_sentiment = "Neutral ⚖️"
            sentiment_color = "🟡"

        return f"""
**😊 Crypto Sentiment Analysis for {symbol}:**

{sentiment_color} **Overall Sentiment:** {overall_sentiment}
📊 **VADER Score:** {avg_vader:.3f} ({self._interpret_vader_score(avg_vader)})
📈 **TextBlob Score:** {avg_textblob:.3f}

**💡 Crypto Market Notes:**
- Highly volatile and sentiment-driven
- 24/7 trading market
- Influenced by regulatory news and adoption

*Note: For real-time crypto sentiment, integrate with specialized APIs*
        """

    def _interpret_vader_score(self, score: float) -> str:
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"


class PricePredictor:
    def __init__(self):
        pass

    def predict(self, data: pd.DataFrame, symbol: str) -> Tuple[str, go.Figure]:
        """Simple price prediction using moving average trend"""
        if len(data) < 10:
            return "Insufficient data for prediction", go.Figure()

        # Simple prediction based on recent trend
        recent_trend = data['Close'][-10:].pct_change().mean()
        current_price = data['Close'].iloc[-1]

        # Predict next 5 days
        future_days = 5
        predicted_prices = []

        last_price = current_price
        for i in range(future_days):
            next_price = last_price * (1 + recent_trend)
            predicted_prices.append(next_price)
            last_price = next_price

        # Create prediction chart
        fig = self._create_prediction_chart(data, predicted_prices, symbol)

        prediction_direction = "up" if recent_trend > 0 else "down"
        confidence = min(abs(recent_trend * 1000), 80)  # Simple confidence metric

        prediction_text = f"""
**🔮 Price Prediction for {symbol}:**

📈 **Short-term Trend:** {prediction_direction.capitalize()}
📊 **Predicted Daily Movement:** {recent_trend * 100:+.3f}%
🎯 **Confidence Level:** {confidence:.1f}%
💰 **Current Price:** ${current_price:.2f}
📅 **5-Day Forecast:** ${predicted_prices[-1]:.2f}

**⚠️ Disclaimer:** This is a simple trend-based prediction. 
Always conduct thorough research before making investment decisions.
        """

        return prediction_text, fig

    def _create_prediction_chart(self, data: pd.DataFrame, predictions: List[float], symbol: str) -> go.Figure:
        """Create prediction visualization"""
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predictions))

        fig = go.Figure()

        # Historical data (last 30 days)
        historical_dates = data.index[-30:]
        historical_prices = data['Close'][-30:]

        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_prices,
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue', width=2)
        ))

        # Predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted Prices',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title=f'{symbol} Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price ($)'
        )

        return fig