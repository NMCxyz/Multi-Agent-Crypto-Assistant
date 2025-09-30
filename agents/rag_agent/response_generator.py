import logging
from typing import List, Dict, Any, Optional, Union
from utils.redis_cache import get_cache_manager

class ResponseGenerator:
    """
    Generates responses based on retrieved context and user query.
    """
    def __init__(self, config):
        """
        Initialize the response generator.

        Args:
            config: Configuration object
            llm: Large language model for response generation
        """
        self.logger = logging.getLogger(__name__)
        self.config = config  # Store config for later use
        self.response_generator_model = config.rag.response_generator_model
        self.include_sources = getattr(config.rag, "include_sources", True)
        self.cache_manager = get_cache_manager() if config.cache.enable_caching else None

    def _build_prompt(
            self,
            query: str, 
            context: str,
            chat_history: Optional[List[Dict[str, str]]] = None
        ) -> str:
        """
        Build the prompt for the language model.
        
        Args:
            query: User query
            context: Formatted context from retrieved documents
            chat_history: Optional chat history
            
        Returns:
            Complete prompt string
        """

        table_instructions = """
        Some of the retrieved information is presented in table format. When using information from tables:
        1. Present tabular data using proper markdown table formatting with headers, like this:
            | Column1 | Column2 | Column3 |
            |---------|---------|---------|
            | Value1  | Value2  | Value3  |
        2. Re-format the table structure to make it easier to read and understand
        3. If any new component is introduced during re-formatting of the table, mention it explicitly
        4. Clearly interpret the tabular data in your response
        5. Reference the relevant table when presenting specific data points
        6. If appropriate, summarize trends or patterns shown in the tables
        7. If only reference numbers are mentioned and you can fetch the corresponding values like research paper title or authors from the context, replace the reference numbers with the actual values
        """

        response_format_instructions = """Instructions:
        1. Answer the query based ONLY on the information provided in the context.
        2. If the context doesn't contain relevant information to answer the query, state: "I don't have enough information to answer this question based on the provided context."
        3. Do not use prior knowledge not contained in the context.
        5. Be concise and accurate.
        6. Provide a well-structured response with heading, sub-headings and tabular structure if required in markdown format based on retrieved knowledge. Keep the headings and sub-headings small sized.
        7. Only provide sections that are meaningful to have in a chatbot reply. For example, do not explicitly mention references.
        8. If values are involved, make sure to respond with perfect values present in context. Do not make up values.
        9. Do not repeat the question in the answer or response."""
            
        # Build the prompt
        prompt = f"""You are a crypto/financial assistant providing accurate information based on verified crypto and financial sources.

        Here are the last few messages from our conversation:

        {chat_history}

        The user has asked the following question:
        {query}

        I've retrieved the following information to help answer this question:

        {context}

        {table_instructions}

        {response_format_instructions}

        Based on the provided information, please answer the user's question thoroughly but concisely.
        If the information doesn't contain the answer, acknowledge the limitations of the available information.

        Do not provide any source link that is not present in the context. Do not make up any source link.

        Crypto/Financial Assistant Response:"""

        return prompt

    def validate_context_sufficiency(
            self,
            query: str,
            retrieved_docs: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
        """
        Validate if the retrieved context is sufficient to answer the user query.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved document chunks
            
        Returns:
            Dict with validation result and reasoning
        """
        if not retrieved_docs:
            return {
                "sufficient": False,
                "reasoning": "No relevant documents found",
                "confidence": 0.0
            }
        
        # Extract content from documents for validation
        doc_texts = [doc["content"] for doc in retrieved_docs]
        context = "\n\n---\n\n".join(doc_texts)

        # Validation debug info removed for cleaner output

        validation_prompt = f"""
        As a crypto/financial expert, determine if the provided context contains enough information to answer the user's query.

        User Query: {query}

        Retrieved Context:
        {context}

        Analysis Requirements:
        1. Does the context directly address the user's query?
        2. Are there specific data points, facts, or information that answer the question?
        3. Is the information complete enough for a satisfactory response?

        Respond with ONLY one of these formats:
        - SUFFICIENT: [brief reason why context is adequate]
        - INSUFFICIENT: [brief reason why context is inadequate]

        Response:
        """

        validation_response = self.response_generator_model.invoke(validation_prompt)
        response_text = validation_response.content.strip()

        if response_text.startswith("SUFFICIENT:"):
            result = {
                "sufficient": True,
                "reasoning": response_text.replace("SUFFICIENT:", "").strip(),
                "confidence": self._calculate_confidence(retrieved_docs)
            }
        else:
            result = {
                "sufficient": False,
                "reasoning": response_text.replace("INSUFFICIENT:", "").strip(),
                "confidence": self._calculate_confidence(retrieved_docs)
            }

        return result

    def generate_response(
            self,
            query: str,
            retrieved_docs: List[Dict[str, Any]],
            picture_paths: List[str],
            chat_history: Optional[List[Dict[str, str]]] = None,
            session_id: Optional[str] = None,
        ) -> Dict[str, Any]:
        """
        Generate a response based on retrieved documents.

        Args:
            query: User query
            retrieved_docs: List of retrieved document dictionaries
            chat_history: Optional chat history

        Returns:
            Dict containing response text and source information
        """
        # Extract content from documents for context
        doc_texts = [doc["content"] for doc in retrieved_docs]

        # Combine retrieved documents into a single context
        context = "\n\n===DOCUMENT SECTION===\n\n".join(doc_texts)

        # Check cache first
        final_response = ""
        if self.cache_manager:
            cached_response = self.cache_manager.get_prompt_response_cache(
                query, context, chat_history, session_id
            )
            if cached_response:
                final_response = cached_response
                # Still extract sources for cached responses
                sources = self._extract_sources(retrieved_docs) if self.include_sources else []
                confidence = self._calculate_confidence(retrieved_docs)

                # Add picture paths to cached response
                response_with_pictures = final_response + "\n\n##### Reference images:"
                for picture_path in picture_paths:
                    response_with_pictures += f"\n- [{picture_path.split('/')[-1]}]({picture_path})"

                return {
                    "response": response_with_pictures,
                    "sources": sources,
                    "confidence": confidence,
                    "cached": True  # Flag to indicate cached response
                }

        # Build the prompt
        prompt = self._build_prompt(query, context, chat_history)

        # Generate response
        response = self.response_generator_model.invoke(prompt)

        # Set final response for return
        if not final_response:
            final_response = response.content

        # Cache the response if enabled
        if self.cache_manager:
            self.cache_manager.set_prompt_response_cache(
                query, context, final_response, chat_history, session_id,
                ttl_hours=self.config.cache.cache_ttl_hours
            )
        
        # Extract sources for citation
        sources = self._extract_sources(retrieved_docs) if self.include_sources else []

        # Calculate confidence
        confidence = self._calculate_confidence(retrieved_docs)

        # Add sources to response
        if self.include_sources:
            response_with_source = final_response + "\n\n##### Source documents:"
            for current_source in sources:
                source_path = current_source['path']
                source_title = current_source['title']
                response_with_source += f"\n- [{source_title}]({source_path})"
        else:
            response_with_source = final_response

        # Add picture paths to response
        response_with_source_and_picture_paths = response_with_source + "\n\n##### Reference images:"
        for picture_path in picture_paths:
            response_with_source_and_picture_paths += f"\n- [{picture_path.split('/')[-1]}]({picture_path})"

        # Format final response
        result = {
            "response": response_with_source_and_picture_paths,
            "sources": sources,
            "confidence": confidence,
            "cached": False  # Flag to indicate fresh response
        }

        return result

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract source information from retrieved documents for citation.
        
        Args:
            documents: List of retrieved document dictionaries
            
        Returns:
            List of source information dictionaries
        """
        sources = []
        seen_sources = set()  # Track unique sources to avoid duplicates
        
        for doc in documents:
            # Extract source and source_path
            source = doc.get("source")
            source_path = doc.get("source_path")
            
            # Skip if no source information is available
            if not source:
                continue
                
            # Create a unique identifier for this source
            source_id = f"{source}|{source_path}"
            
            # Skip if we've already included this source
            if source_id in seen_sources:
                continue
                
            # Add to our sources list
            source_info = {
                "title": source,
                "path": source_path,
                "score": doc.get("combined_score", doc.get("rerank_score", doc.get("score", 0.0)))
            }
            
            sources.append(source_info)
            seen_sources.add(source_id)
        
        # Sort sources by score from highest to lowest
        sources.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Format the final sources list, removing the scores which were just used for sorting
        formatted_sources = []
        for source in sources:
            formatted_source = {
                "title": source["title"],
                "path": source["path"]
            }
            formatted_sources.append(formatted_source)
            
        return formatted_sources

    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on retrieved documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Confidence score between 0 and 1
        """
        if not documents:
            return 0.0
            
        # Use combined score (both reranker and cosine similarity) if available, otherwise use original score
        if "combined_score" in documents[0]:
            scores = [doc.get("combined_score", 0) for doc in documents[:3]]
        elif "rerank_score" in documents[0]:
            scores = [doc.get("rerank_score", 0) for doc in documents[:3]]
        else:
            scores = [doc.get("score", 0) for doc in documents[:3]]
            
        # Average of top 3 document scores or fewer if less than 3
        return sum(scores) / len(scores) if scores else 0.0