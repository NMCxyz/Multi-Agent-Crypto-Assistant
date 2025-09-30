import os
import json
import logging
import urllib.request
from typing import Dict, Any

class ForwardLookingStatementAgent:
    """
    Forward-looking statement detection agent using FinBERT-FLS model via Azure ML.
    """

    def __init__(self, azure_endpoint="https://rg-finbert-cls.eastasia.inference.ml.azure.com/score"):
        self.logger = logging.getLogger(__name__)
        self.azure_endpoint = azure_endpoint
        self.api_key = os.getenv("AZURE_ML_API_KEY")

        # API key is required for Azure ML endpoint

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Detect forward-looking statements in financial/crypto text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with FLS detection result and reasoning
        """
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
        is_fls = None
        reason = "Analysis completed"
        confidence = 0.5

        # Check if response has the expected direct format
        if "is_fls" in prediction_result:
            is_fls = prediction_result["is_fls"]
            reason = prediction_result.get("reason", "Analysis completed")
            confidence = prediction_result.get("confidence", 0.5)

        # Check if response has nested format (Azure ML result array)
        elif "result" in prediction_result and isinstance(prediction_result["result"], list) and len(prediction_result["result"]) > 0:
            result_item = prediction_result["result"][0]
            if "is_fls" in result_item:
                is_fls = result_item["is_fls"]
                reason = result_item.get("reason", "Analysis completed")
                confidence = result_item.get("confidence", 0.5)

        return {
            "is_fls": is_fls,
            "reason": reason,
            "confidence": confidence
        }
