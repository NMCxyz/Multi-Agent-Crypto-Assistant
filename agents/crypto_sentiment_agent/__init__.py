from .crypto_sentiment_inference import CryptoSentimentAgent as _CryptoSentimentAgent

class CryptoSentimentAgent:
    """Crypto sentiment analysis agent using CryptoBERT model via Azure ML."""

    def __init__(self, azure_endpoint="https://rg-crypto-bert.eastasia.inference.ml.azure.com/score"):
        self.model = _CryptoSentimentAgent(azure_endpoint=azure_endpoint)

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyze sentiment of crypto/financial text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment label and confidence
        """
        return self.model.predict(text)
