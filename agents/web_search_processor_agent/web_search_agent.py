import requests
from typing import Dict

from .tavily_search import TavilySearchAgent

class WebSearchAgent:
    """
    Agent responsible for retrieving real-time crypto/financial information from web sources.
    """
    
    def __init__(self, config):
        self.tavily_search_agent = TavilySearchAgent()
    
    def search(self, query: str) -> str:
        """
        Perform general web search for crypto/financial information.
        """
        tavily_results = self.tavily_search_agent.search_tavily(query=query)

        return f"Tavily Results:\n{tavily_results}\n"
