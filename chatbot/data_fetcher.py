import yfinance as yf
import pandas as pd
import requests
from typing import Optional, Dict
import time
import json
from config import INTERNATIONAL_MARKETS


class MultiAssetDataFetcher:
    def __init__(self):
        self.cache = {}
        # Crypto API endpoint
        self.crypto_api = "https://api.coingecko.com/api/v3"

    def get_asset_data(self, asset_type: str, symbol: str, period: str = "1y", country: str = "US") -> pd.DataFrame:
        """Fetch data for any asset type with country support"""
        cache_key = f"{asset_type}_{symbol}_{period}_{country}"

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
            raise Exception(f"Error fetching data for {asset_type} {symbol} from {country}: {str(e)}")

    def _get_yfinance_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Get data using yfinance (stocks, ETFs, indices, commodities, bonds)"""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data

    # ... (rest of the data fetching methods remain similar)

    def get_fundamental_data(self, asset_type: str, symbol: str, country: str = "US") -> str:
        """Get fundamental analysis data for any asset type with country support"""
        try:
            if asset_type == "Stock":
                return self._get_international_stock_fundamentals(symbol, country)
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

    def _get_international_stock_fundamentals(self, symbol: str, country: str) -> str:
        """Get international stock fundamental data"""
        ticker = yf.Ticker(symbol)
        info = ticker.info

        currency = INTERNATIONAL_MARKETS[country]["currency"]

        fundamental_info = {
            'Company Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap': self._format_number(info.get('marketCap', 0), currency),
            'P/E Ratio': info.get('trailingPE', 'N/A'),
            'EPS': info.get('trailingEps', 'N/A'),
            'Dividend Yield': f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else 'N/A',
            '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
            'Beta': info.get('beta', 'N/A'),
            'Currency': currency,
            'Country': country
        }

        analysis = f"""
**📊 Fundamental Analysis for {symbol} ({country} Market):**

🏢 **Company:** {fundamental_info['Company Name']}
🌍 **Country:** {fundamental_info['Country']}
🏭 **Sector:** {fundamental_info['Sector']}
🏭 **Industry:** {fundamental_info['Industry']}
💰 **Market Cap:** {fundamental_info['Market Cap']}
📈 **P/E Ratio:** {fundamental_info['P/E Ratio']}
📊 **EPS:** {fundamental_info['EPS']}
🎯 **Dividend Yield:** {fundamental_info['Dividend Yield']}
📅 **52 Week Range:** {fundamental_info['52 Week Low']} - {fundamental_info['52 Week High']}
📉 **Beta:** {fundamental_info['Beta']}
💱 **Currency:** {fundamental_info['Currency']}
        """

        return analysis

    def get_market_overview(self, country: str) -> str:
        """Get market overview for a specific country"""
        market_info = INTERNATIONAL_MARKETS[country]

        # Sample market data - in production, you'd fetch real data
        market_data = {
            "US": {"index": "^GSPC", "performance": "+15.2% YTD", "volatility": "Medium"},
            "UK": {"index": "^FTSE", "performance": "+8.7% YTD", "volatility": "Low"},
            "Germany": {"index": "^GDAXI", "performance": "+12.4% YTD", "volatility": "Medium"},
            "Japan": {"index": "^N225", "performance": "+10.1% YTD", "volatility": "Medium"},
            "Canada": {"index": "^GSPTSE", "performance": "+9.8% YTD", "volatility": "Medium"},
        }

        current_data = market_data.get(country, {
            "index": "N/A",
            "performance": "N/A",
            "volatility": "N/A"
        })

        return f"""
**🌍 {country} Market Overview**

💱 **Currency:** {market_info['currency']}
📊 **Market Suffix:** {market_info['suffix']}
📈 **Main Index:** {current_data['index']}
🚀 **YTD Performance:** {current_data['performance']}
📉 **Volatility:** {current_data['volatility']}

**🏛️ Key Sectors:**
- Financial Services
- Technology  
- Healthcare
- Industrial Goods
- Consumer Services

**💡 Trading Hours:**
- Typically 9:00 AM - 5:00 PM Local Time
- Closed on weekends and national holidays

*Note: Market data is sample data. Integrate with real market data APIs for live information.*
        """

    def _format_number(self, num: float, currency: str) -> str:
        """Format large numbers for readability with currency"""
        if num >= 1e12:
            return f"{currency} {num / 1e12:.2f}T"
        elif num >= 1e9:
            return f"{currency} {num / 1e9:.2f}B"
        elif num >= 1e6:
            return f"{currency} {num / 1e6:.2f}M"
        else:
            return f"{currency} {num:,.2f}"