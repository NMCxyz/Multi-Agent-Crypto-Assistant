from .fls_inference import ForwardLookingStatementAgent as _ForwardLookingStatementAgent

class ForwardLookingStatementAgent:
    """Forward-looking statement detection agent using FinBERT-FLS model via Azure ML."""

    def __init__(self, azure_endpoint="https://rg-finbert-cls.eastasia.inference.ml.azure.com/score"):
        self.model = _ForwardLookingStatementAgent(azure_endpoint)

    def detect_fls(self, text: str) -> dict:
        """
        Detect forward-looking statements in financial/crypto text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with FLS detection result and reasoning
        """
        return self.model.predict(text)
