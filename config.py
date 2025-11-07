# API Keys and Configuration
ALPHA_VANTAGE_API_KEY = " "
NEWS_API_KEY = " "
ALPHA_VANTAGE_NEWS_KEY = " "
FINNHUB_API_KEY = " "

# Model Settings
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Analysis Parameters
TECHNICAL_INDICATORS = ['RSI', 'MACD', 'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Lower']
PREDICTION_DAYS = 30

# News API Configuration
NEWS_API_CONFIG = {
    "base_url": "https://newsapi.org/v2/everything",
    "page_size": 10,
    "sort_by": "publishedAt"
}

# International Market Configuration
INTERNATIONAL_MARKETS = {
    "US": {
        "suffix": "",
        "examples": ["AAPL", "TSLA", "MSFT", "GOOGL"],
        "currency": "USD",
        "news_country": "us"
    },
    "UK": {
        "suffix": ".L",
        "examples": ["HSBA.L", "VOD.L", "BP.L", "TSCO.L"],
        "currency": "GBP",
        "news_country": "gb"
    },
    "Germany": {
        "suffix": ".DE",
        "examples": ["SAP.DE", "BMW.DE", "SIE.DE", "ALV.DE"],
        "currency": "EUR",
        "news_country": "de"
    },
    "France": {
        "suffix": ".PA",
        "examples": ["AIR.PA", "MC.PA", "SAN.PA", "OR.PA"],
        "currency": "EUR"
    },
    "Japan": {
        "suffix": ".T",
        "examples": ["7203.T", "6758.T", "9433.T", "9984.T"],
        "currency": "JPY"
    },
    "Canada": {
        "suffix": ".TO",
        "examples": ["RY.TO", "TD.TO", "SHOP.TO", "ENB.TO"],
        "currency": "CAD"
    },
    "Australia": {
        "suffix": ".AX",
        "examples": ["CBA.AX", "BHP.AX", "WBC.AX", "NAB.AX"],
        "currency": "AUD"
    },
    "Hong Kong": {
        "suffix": ".HK",
        "examples": ["0700.HK", "0941.HK", "0005.HK", "1299.HK"],
        "currency": "HKD"
    },
    "India": {
        "suffix": ".NS",
        "examples": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"],
        "currency": "INR"
    },
    "Brazil": {
        "suffix": ".SA",
        "examples": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA"],
        "currency": "BRL"
    }
}
# Country codes for news API
COUNTRY_CODES = {
    "US": "us",
    "UK": "gb",
    "Germany": "de",
    "France": "fr",
    "Japan": "jp",
    "Canada": "ca",
    "Australia": "au",
    "Hong Kong": "hk",
    "India": "in",
    "Brazil": "br"
}