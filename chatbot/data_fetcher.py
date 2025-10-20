import yfinance as yf
import pandas as pd
import requests
from typing import Optional, Dict
import time
import json


class MultiAssetDataFetcher:
    def __init__(self):
        self.cache = {}
        # Crypto API endpoint
        self.crypto_api = "https://api.coingecko.com/api/v3"

    def get_asset_data(self, asset_type: str, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch data for any asset type"""
        cache_key = f"{asset_type}_{symbol}_{period}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            if asset_type in ["Stock", "ETF", "Index", "Commodity", "Bond"]:
                data = self._get_yfinance_data(symbol, period)
            elif asset_type == "Cryptocurrency":
                data = self._get_crypto_data(symbol, period)
            elif asset_type == "Forex":
                data = self._get_forex_data(symbol, period)
            else:
                raise ValueError(f"Unsupported asset type: {asset_type}")

            if data.empty:
                raise ValueError(f"No data found for {asset_type} symbol: {symbol}")

            # Calculate additional features
            data['Daily Return'] = data['Close'].pct_change()
            data['Volume SMA'] = data['Volume'].rolling(window=20).mean() if 'Volume' in data.columns else 0

            self.cache[cache_key] = data
            return data

        except Exception as e:
            raise Exception(f"Error fetching data for {asset_type} {symbol}: {str(e)}")

    def _get_yfinance_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Get data using yfinance (stocks, ETFs, indices, commodities, bonds)"""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data

    def _get_crypto_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Get cryptocurrency data"""
        try:
            # First try yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if not data.empty:
                return data
        except:
            pass

        # Fallback to CoinGecko API
        return self._get_coingecko_data(symbol, period)

    def _get_forex_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Get forex data using yfinance"""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data

    def _get_coingecko_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Get cryptocurrency data from CoinGecko API"""
        try:
            # Convert symbol to CoinGecko ID
            crypto_id = self._convert_crypto_symbol(symbol)

            # Map period to days
            period_days = {
                "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
                "6mo": 180, "1y": 365, "2y": 730, "5y": 1825
            }

            days = period_days.get(period, 365)

            url = f"{self.crypto_api}/coins/{crypto_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }

            response = requests.get(url, params=params)
            data = response.json()

            # Convert to DataFrame
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.rename(columns={'price': 'Close'}, inplace=True)

            # Create OHLC data (simplified)
            df['Open'] = df['Close'].shift(1)
            df['High'] = df[['Open', 'Close']].max(axis=1)
            df['Low'] = df[['Open', 'Close']].min(axis=1)
            df['Volume'] = 0  # Volume not available in basic API

            return df

        except Exception as e:
            raise Exception(f"CoinGecko API error: {str(e)}")

    def get_fundamental_data(self, asset_type: str, symbol: str) -> str:
        """Get fundamental analysis data for any asset type"""
        try:
            if asset_type == "Stock":
                return self._get_stock_fundamentals(symbol)
            elif asset_type == "Cryptocurrency":
                return self._get_crypto_fundamentals(symbol)
            elif asset_type == "ETF":
                return self._get_etf_fundamentals(symbol)
            elif asset_type == "Forex":
                return self._get_forex_fundamentals(symbol)
            elif asset_type == "Commodity":
                return self._get_commodity_fundamentals(symbol)
            elif asset_type == "Index":
                return self._get_index_fundamentals(symbol)
            elif asset_type == "Bond":
                return self._get_bond_fundamentals(symbol)
            else:
                return f"Fundamental analysis not available for {asset_type}"

        except Exception as e:
            return f"Error fetching fundamental data: {str(e)}"

    def _get_stock_fundamentals(self, symbol: str) -> str:
        """Get stock fundamental data"""
        ticker = yf.Ticker(symbol)
        info = ticker.info

        fundamental_info = {
            'Company Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Market Cap': self._format_number(info.get('marketCap', 0)),
            'P/E Ratio': info.get('trailingPE', 'N/A'),
            'EPS': info.get('trailingEps', 'N/A'),
            'Dividend Yield': f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else 'N/A',
            '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
            'Beta': info.get('beta', 'N/A')
        }

        analysis = f"""
**📊 Fundamental Analysis for {symbol} (Stock):**

🏢 **Company:** {fundamental_info['Company Name']}
🏭 **Sector:** {fundamental_info['Sector']}
💰 **Market Cap:** {fundamental_info['Market Cap']}
📈 **P/E Ratio:** {fundamental_info['P/E Ratio']}
📊 **EPS:** {fundamental_info['EPS']}
🎯 **Dividend Yield:** {fundamental_info['Dividend Yield']}
📅 **52 Week Range:** {fundamental_info['52 Week Low']} - {fundamental_info['52 Week High']}
📉 **Beta:** {fundamental_info['Beta']}
        """

        return analysis

    def _get_crypto_fundamentals(self, symbol: str) -> str:
        """Get cryptocurrency fundamental data"""
        try:
            # For demonstration, using sample data
            # In production, integrate with CoinGecko API
            crypto_data = {
                "BTC-USD": {"name": "Bitcoin", "rank": 1, "market_cap": 900000000000},
                "ETH-USD": {"name": "Ethereum", "rank": 2, "market_cap": 400000000000},
                "ADA-USD": {"name": "Cardano", "rank": 8, "market_cap": 15000000000},
            }

            crypto_info = crypto_data.get(symbol, {
                "name": symbol.replace("-USD", ""),
                "rank": "N/A",
                "market_cap": 0
            })

            return f"""
**₿ Cryptocurrency Analysis for {symbol}:**

🔗 **Name:** {crypto_info['name']}
🏆 **Market Rank:** #{crypto_info['rank']}
💰 **Market Cap:** ${crypto_info['market_cap']:,.0f}
📊 **Type:** Cryptocurrency
🌐 **Blockchain:** Proof of Work/Stake

*Note: For detailed crypto metrics, integrate with CoinGecko API*
            """

        except Exception as e:
            return f"Error fetching crypto data: {str(e)}"

    def _get_etf_fundamentals(self, symbol: str) -> str:
        """Get ETF fundamental data"""
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return f"""
**📈 ETF Analysis for {symbol}:**

🏷️ **Name:** {info.get('longName', 'N/A')}
📂 **Category:** {info.get('category', 'N/A')}
💰 **Total Assets:** {self._format_number(info.get('totalAssets', 0))}
📊 **Expense Ratio:** {info.get('expenseRatio', 'N/A')}
🎯 **YTD Return:** {info.get('ytdReturn', 'N/A')}
        """

    def _get_forex_fundamentals(self, symbol: str) -> str:
        """Get forex pair fundamental data"""
        return f"""
**💱 Forex Pair Analysis for {symbol}:**

🌍 **Currency Pair:** {symbol.replace('=X', '')}
📊 **Type:** Foreign Exchange
💹 **Trading:** 24/5 Market
🔄 **Spread:** Typically low
📈 **Volatility:** Medium

*Note: Forex analysis requires economic indicators and central bank data*
        """

    def _get_commodity_fundamentals(self, symbol: str) -> str:
        """Get commodity fundamental data"""
        commodity_names = {
            "GC=F": "Gold",
            "SI=F": "Silver",
            "CL=F": "Crude Oil",
            "NG=F": "Natural Gas"
        }

        name = commodity_names.get(symbol, "Commodity")

        return f"""
**🛢️ Commodity Analysis for {symbol}:**

💎 **Commodity:** {name}
🏭 **Type:** Physical Asset
📊 **Trading:** Futures Contract
🌍 **Global Market:** Yes
📈 **Inflation Hedge:** {name in ['Gold', 'Silver']}

*Note: Commodity prices affected by supply/demand and geopolitical factors*
        """

    def _get_index_fundamentals(self, symbol: str) -> str:
        """Get index fundamental data"""
        index_names = {
            "^GSPC": "S&P 500 Index",
            "^DJI": "Dow Jones Industrial Average",
            "^IXIC": "NASDAQ Composite"
        }

        name = index_names.get(symbol, "Market Index")

        return f"""
**📊 Index Analysis for {symbol}:**

📈 **Index Name:** {name}
🏢 **Type:** Market Index
📊 **Composition:** Multiple Stocks
🌍 **Market Representation:** Broad Market
📉 **Volatility:** Varies by index

*Note: Indices represent overall market performance*
        """

    def _get_bond_fundamentals(self, symbol: str) -> str:
        """Get bond fundamental data"""
        bond_names = {
            "^TNX": "10-Year Treasury Yield",
            "^FVX": "5-Year Treasury Yield",
            "^TYX": "30-Year Treasury Yield"
        }

        name = bond_names.get(symbol, "Bond Yield")

        return f"""
**📋 Bond Analysis for {symbol}:**

🏛️ **Bond Type:** {name}
💰 **Yield Type:** Interest Rate
📊 **Risk:** Low (Government Backed)
📈 **Inverse Relationship:** With stock prices
🏦 **Issuer:** US Government

*Note: Bond yields move inversely to bond prices*
        """

    def _convert_crypto_symbol(self, symbol: str) -> str:
        """Convert trading symbol to CoinGecko ID"""
        crypto_map = {
            "BTC-USD": "bitcoin",
            "ETH-USD": "ethereum",
            "ADA-USD": "cardano",
            "BNB-USD": "binancecoin",
            "XRP-USD": "ripple",
            "SOL-USD": "solana",
            "DOT-USD": "polkadot",
            "DOGE-USD": "dogecoin"
        }
        return crypto_map.get(symbol, symbol.lower().replace("-usd", ""))

    def _format_number(self, num: float) -> str:
        """Format large numbers for readability"""
        if num >= 1e12:
            return f"${num / 1e12:.2f}T"
        elif num >= 1e9:
            return f"${num / 1e9:.2f}B"
        elif num >= 1e6:
            return f"${num / 1e6:.2f}M"
        else:
            return f"${num:,.2f}"