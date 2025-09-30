import os
import logging
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CryptoSentimentAgent:
    """
    Crypto sentiment analysis agent using CryptoBERT model.
    """

    def __init__(self, azure_endpoint="https://rg-crypto-bert.eastasia.inference.ml.azure.com/score"):
        self.logger = logging.getLogger(__name__)
        self.azure_endpoint = azure_endpoint
        self.api_key = os.getenv("CRYPTO_BERT_API_KEY")

        # API key is required for Azure ML endpoint

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of crypto/financial text using Azure ML endpoint.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment label and confidence
        """
        import urllib.request
        import json

        # Prepare request data
        data = {"text": text}

        # Create request
        body = json.dumps(data).encode('utf-8')
        url = self.azure_endpoint
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        req = urllib.request.Request(url, body, headers)

        # Make request and get response
        with urllib.request.urlopen(req) as response:
            result = response.read()
            prediction_result = json.loads(result.decode("utf8"))

        # Parse response - handle different response formats
        sentiment_label = None
        confidence = None

        # Check if response has the expected direct format
        if "label" in prediction_result and "score" in prediction_result:
            sentiment_label = prediction_result["label"]
            confidence = prediction_result["score"]
            probabilities = prediction_result.get("probabilities", {})

        # Check if response has nested format (Azure ML result array)
        elif "result" in prediction_result and isinstance(prediction_result["result"], list) and len(prediction_result["result"]) > 0:
            result_item = prediction_result["result"][0]
            if "label" in result_item and "score" in result_item:
                sentiment_label = result_item["label"]
                confidence = result_item["score"]
                probabilities = result_item.get("probabilities", {})

        # Convert label to standard format
        label_mapping = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"
        }
        sentiment_label = label_mapping.get(sentiment_label, sentiment_label)

        return {
            "sentiment": sentiment_label,
            "confidence": confidence,
            "probabilities": probabilities
        }
