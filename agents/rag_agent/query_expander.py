import logging
from typing import List, Dict, Any

class QueryExpander:
    """
    Expands user queries with crypto/financial terminology to improve retrieval.
    """
    def __init__(self, config):
        self.logger = logging.getLogger(f"{self.__module__}")
        self.config = config
        self.model = config.rag.llm
        
    def expand_query(self, original_query: str) -> Dict[str, Any]:
        """
        Decompose the original query into optimized sub-queries for better retrieval.

        Args:
            original_query: The user's original query

        Returns:
            Dictionary with original query and decomposed sub-queries
        """
        self.logger.info(f"Decomposing query: {original_query}")
        
        # Generate sub-queries for better retrieval
        sub_queries = self._decompose_query(original_query)
        
        return {
            "original_query": original_query,
            "sub_queries": sub_queries,
            "expanded_query": original_query  # Keep for backward compatibility
        }
    
    def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose user query into 2-4 optimized sub-queries for better retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            List of 2-4 sub-queries optimized for vector search
        """
        prompt = f"""
        As a crypto/financial expert, decompose the following user query into 2-4 specific, focused sub-queries that would be optimal for semantic search in a crypto/financial database.

        User Query: {query}

        Guidelines:
        1. Create 2-4 sub-queries maximum
        2. Each sub-query should target specific aspects or keywords from the original query
        3. Use precise crypto/financial terminology that would match document content
        4. Focus on entities, concepts, or data points the user is seeking
        5. Make queries concise and search-friendly (3-8 words each)
        6. IMPORTANT: Keep key terms unchanged if they appear to be specific document sections or proper nouns (e.g., "RISK FACTORS SUMMARY" should remain "RISK FACTORS SUMMARY")

        Examples:
        - If user asks "show data from coingecko annual report" → ["coingecko annual report data", "cryptocurrency market statistics", "crypto industry report findings"]
        - If user asks "risk factors summary" → ["risk factors summary", "business risks crypto", "operating results volatility"]
        - If user asks "revenue information from RISK FACTORS SUMMARY" → ["RISK FACTORS SUMMARY revenue", "revenue financial data", "revenue implications business"]

        Respond with only the sub-queries, one per line, no explanations:
        """
        
        response = self.model.invoke(prompt)
        sub_queries = [q.strip() for q in response.content.split('\n') if q.strip()]
        
        self.logger.info(f"Raw LLM response: {response.content}")
        self.logger.info(f"Parsed sub_queries: {sub_queries}")
        
        # Ensure we have 2-4 queries
        if len(sub_queries) < 2:
            self.logger.warning(f"Query decomposition produced only {len(sub_queries)} sub-queries, adding original query")
            sub_queries.append(query)  # Add original query as fallback
        if len(sub_queries) > 4:
            sub_queries = sub_queries[:4]
            
        self.logger.info(f"Final decomposed sub-queries: {sub_queries}")
        return sub_queries
    
    def _generate_expansions(self, query: str) -> str:
        """Use LLM to expand query with crypto/financial terminology (legacy method)."""
        prompt = f"""
        As a crypto/financial expert, expand the following query with relevant cryptocurrency, financial, and market terminology,
        synonyms, and related concepts that would help in retrieving relevant crypto/financial information:

        User Query: {query}

        Expand the query only if you feel like it is required, otherwise keep the user query intact.
        Be specific to the crypto/financial domain mentioned in the user query, do not add other domains.
        If the user query asks about answering in tabular format, include that in the expanded query and do not answer in tabular format yourself.
        Provide only the expanded query without explanations.
        """
        expansion = self.model.invoke(prompt)
        
        return expansion