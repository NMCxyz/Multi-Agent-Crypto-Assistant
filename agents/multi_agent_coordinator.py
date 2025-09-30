"""
Multi-Agent Coordinator System for handling complex queries requiring multiple agents.

This module enables orchestration of multiple specialized agents in a coordinated pipeline
to handle complex user requests that require combined capabilities from different agents.
"""

import json
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
import os, getpass, uuid
from dotenv import load_dotenv

from config import Config

load_dotenv()

# Load configuration
config = Config()


class AgentRequirement(TypedDict):
    """Structure for defining agent requirements in a pipeline."""
    agent_name: str
    priority: int  # Priority order (lower numbers run first)
    input_from: Optional[str]  # Which agent's output to use as input (None for original query)
    parameters: Optional[Dict[str, Any]]  # Additional parameters for the agent


class PipelineConfig(TypedDict):
    """Configuration for a multi-agent pipeline."""
    pipeline_id: str
    requirements: List[AgentRequirement]
    execution_mode: Literal["sequential", "parallel"]  # sequential or parallel execution
    output_combination_strategy: Literal["concatenate", "summarize", "merge"]


class MultiAgentCoordinator:
    """
    Coordinator for orchestrating multiple agents in response to complex queries.
    """

    def __init__(self, config: Config):
        """Initialize the Multi-Agent Coordinator."""
        self.config = config
        self.llm = config.agent_decision.llm

        # Initialize available agents
        self.available_agents = {
            "RAG_AGENT": "For specific crypto/financial knowledge questions from reports",
            "WEB_SEARCH_PROCESSOR_AGENT": "For recent crypto developments and current market trends",
            "CRYPTO_SENTIMENT_AGENT": "For analyzing sentiment in crypto/financial text content",
            "FORWARD_LOOKING_STATEMENT_AGENT": "For detecting forward-looking statements in financial reports",
            "CONVERSATION_AGENT": "For general chat and conversation"
        }

        # Initialize output parser
        self.json_parser = JsonOutputParser()

        # Create the coordination prompt
        self.coordination_prompt = self._create_coordination_prompt()

    def _create_coordination_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for multi-agent coordination decisions."""

        system_prompt = """You are an intelligent Multi-Agent Coordinator that analyzes complex user queries and determines the optimal combination of specialized agents needed to provide a comprehensive response.

Available Agents:
1. RAG_AGENT - For specific crypto/financial knowledge questions that can be answered from the CoinGecko 2024 Annual Crypto Industry Report and related financial literature
2. WEB_SEARCH_PROCESSOR_AGENT - For questions about recent crypto developments, current market trends, or time-sensitive financial information
3. CRYPTO_SENTIMENT_AGENT - For analyzing sentiment and tone in crypto/financial TEXT ONLY (articles, tweets, news, reports) - NOT for numerical data or tables
4. FORWARD_LOOKING_STATEMENT_AGENT - For detecting and analyzing forward-looking statements, predictions, and expectations in financial/crypto reports
5. CONVERSATION_AGENT - For general chat, greetings, and non-crypto questions

Decision Framework:
You are an intelligent coordinator that analyzes user queries and determines the optimal combination of specialized agents needed for comprehensive responses. Your goal is to break down complex queries into logical steps and assign each step to the most appropriate agent.

Available Agents and Their Capabilities:
1. RAG_AGENT - Retrieves and analyzes information from the CoinGecko 2024 Annual Crypto Industry Report and related financial literature. Best for: specific crypto/financial knowledge, data analysis, structured information from reports.

2. WEB_SEARCH_PROCESSOR_AGENT - Searches and processes current web information, recent developments, and time-sensitive data. Best for: recent events, current market trends, real-time information not in static reports.

3. CRYPTO_SENTIMENT_AGENT - Analyzes sentiment and emotional tone in textual content (articles, tweets, news, reports). Best for: understanding market sentiment, public opinion, emotional analysis of text content.

4. FORWARD_LOOKING_STATEMENT_AGENT - Detects and analyzes predictions, expectations, and forward-looking statements in financial content. Best for: identifying future-oriented statements, risk analysis, predictive content.

5. CONVERSATION_AGENT - Handles general conversation, greetings, and basic queries. Best for: casual interaction, simple clarifications.

Coordination Logic:
1. **Query Analysis**: Break down the user's query into its core components and requirements. Identify what information is needed and what type of analysis is required.

2. **Information Sources**: Determine where the required information is most likely to be found:
   - Static/timeless crypto knowledge → RAG_AGENT
   - Recent/current events and trends → WEB_SEARCH_PROCESSOR_AGENT
   - Text sentiment analysis → CRYPTO_SENTIMENT_AGENT
   - Forward-looking statements → FORWARD_LOOKING_STATEMENT_AGENT

3. **Dependencies**: Map out logical dependencies between information gathering and analysis:
   - Content retrieval usually precedes analysis
   - Raw data often needs processing before sentiment analysis
   - Multiple data sources may need to be combined

4. **Pipeline Design**: Create an execution sequence that:
   - Gathers all necessary information first
   - Performs required analyses in logical order
   - Combines results into a coherent response

5. **Decision Rules**:
   - Use multiple agents when a query requires different types of processing (retrieval + analysis)
   - Prefer RAG_AGENT for established crypto/financial knowledge from reports
   - Use WEB_SEARCH when dealing with recent events or information not in static reports
   - Include sentiment analysis when emotional tone or public opinion is relevant
   - Use forward-looking analysis when predictions or expectations are involved
   - ALWAYS use WEB_SEARCH + SENTIMENT_AGENT for sentiment analysis of historical periods (pre-2024) since RAG only contains 2024 data
   - For queries mentioning specific years (2017-2018, etc.) with sentiment analysis, combine WEB_SEARCH (to find content) with CRYPTO_SENTIMENT_AGENT (to analyze it)

Response Format (JSON):
{{
    "needs_multiple_agents": true,
    "primary_agent": "AGENT_NAME",
    "pipeline_requirements": [
        {{
            "agent_name": "AGENT_NAME",
            "priority": 1,
            "input_from": null,
            "parameters": {{}}
        }}
    ],
    "execution_mode": "sequential",
    "reasoning": "Step-by-step reasoning explaining the coordination decision and pipeline design"
}}

Decision Process:
1. Analyze the query to understand what the user is asking for
2. Identify which agents have the capabilities needed to fulfill different aspects of the request
3. Design a logical execution sequence based on information flow and dependencies
4. Specify how outputs from one agent become inputs for subsequent agents
5. Ensure the final output provides a complete, well-synthesized response to the original query
"""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "User Query: {query}\n\nConversation Context: {context}\n\nDetermine the optimal agent pipeline for this query:")
        ])

    def analyze_query_complexity(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze a query to determine if it requires multiple agents and plan the pipeline.

        Args:
            query: The user query to analyze
            context: Recent conversation context

        Returns:
            Dict containing coordination decision and pipeline configuration
        """
        # Create the coordination chain
        coordination_chain = self.coordination_prompt | self.llm | self.json_parser

        # Prepare input
        coordination_input = {
            "query": query,
            "context": context
        }

        # Get coordination decision
        coordination_result = coordination_chain.invoke(coordination_input)

        return coordination_result

    def create_pipeline_config(self, coordination_result: Dict[str, Any]) -> PipelineConfig:
        """
        Convert coordination result into a structured pipeline configuration.

        Args:
            coordination_result: Result from analyze_query_complexity

        Returns:
            Structured pipeline configuration
        """
        pipeline_id = str(uuid.uuid4())

        return PipelineConfig(
            pipeline_id=pipeline_id,
            requirements=coordination_result["pipeline_requirements"],
            execution_mode=coordination_result["execution_mode"],
            output_combination_strategy="intelligent"
        )

    def is_multi_agent_query(self, query: str, context: str = "") -> bool:
        """
        Use LLM-based decision making to determine if a query needs multiple agents.

        This method relies entirely on intelligent analysis by a decision agent rather than
        hard-coded keyword patterns, making it completely generalizable and adaptable.

        Args:
            query: The user query to analyze
            context: Conversation context for additional insights

        Returns:
            Boolean indicating if multiple agents are needed
        """
        # Create a decision prompt for multi-agent detection
        decision_prompt = f"""
        You are an expert system that determines whether a user query requires multiple specialized agents to answer effectively.

        Available agent types:
        1. RAG_AGENT - For specific crypto/financial knowledge from reports
        2. WEB_SEARCH_AGENT - For recent information and current events
        3. SENTIMENT_AGENT - For analyzing sentiment and tone in text
        4. FLS_AGENT - For detecting forward-looking statements
        5. CONVERSATION_AGENT - For general chat and simple queries

        Query to analyze: "{query}"

        Context: "{context[:500]}"  # Limit context length

        Decision criteria:
        - Single agent queries: Simple factual questions, greetings, basic explanations
        - Multi-agent queries: Complex analysis requiring multiple data sources, sentiment analysis of historical data, comparative studies, predictions requiring multiple perspectives

        Specific indicators for multi-agent:
        - Queries requiring information from multiple time periods or sources
        - Queries involving sentiment analysis of content that needs to be retrieved first
        - Queries comparing current vs historical data
        - Queries requiring both factual retrieval and analysis

        Respond with ONLY "MULTI_AGENT" or "SINGLE_AGENT":

        Decision:
        """

        # Use LLM to make the decision
        decision_response = self.llm.invoke(decision_prompt)

        # Extract decision from response
        decision_text = decision_response.content if hasattr(decision_response, 'content') else str(decision_response)
        decision_text = decision_text.strip().upper()

        # Parse the decision
        if "MULTI_AGENT" in decision_text:
            return True
        elif "SINGLE_AGENT" in decision_text:
            return False
        else:
            # If LLM doesn't give clear response, default to single agent for safety
            print(f"⚠️ Unclear decision from LLM: '{decision_text}', defaulting to single agent")
            return False


class AgentPipeline:
    """
    Executes a pipeline of multiple agents based on a pipeline configuration.
    """

    def __init__(self, config: Config, pipeline_config: PipelineConfig):
        """
        Initialize the agent pipeline.

        Args:
            config: Application configuration
            pipeline_config: Configuration defining the agent pipeline
        """
        self.config = config
        self.pipeline_config = pipeline_config
        self.agent_results = {}

        # Import agents dynamically to avoid circular imports
        from agents.rag_agent import CryptoFinancialRAG
        from agents.web_search_processor_agent import WebSearchProcessorAgent
        from agents.crypto_sentiment_agent import CryptoSentimentAgent
        from agents.forward_looking_statement_agent import ForwardLookingStatementAgent

        # Initialize agent instances
        self.agents = {
            "RAG_AGENT": CryptoFinancialRAG(config),
            "WEB_SEARCH_PROCESSOR_AGENT": WebSearchProcessorAgent(config),
            "CRYPTO_SENTIMENT_AGENT": CryptoSentimentAgent(config.crypto_analysis.crypto_sentiment_endpoint),
            "FORWARD_LOOKING_STATEMENT_AGENT": ForwardLookingStatementAgent(config.crypto_analysis.fls_azure_endpoint),
            "CONVERSATION_AGENT": None  # Will be handled separately
        }

    def execute_pipeline(self, query: str, conversation_history: List[BaseMessage], session_id: str) -> Dict[str, Any]:
        """
        Execute the configured agent pipeline.

        Args:
            query: Original user query
            conversation_history: Conversation history
            session_id: Session identifier

        Returns:
            Dict containing results from all agents in the pipeline
        """
        requirements = self.pipeline_config["requirements"]
        execution_mode = self.pipeline_config["execution_mode"]

        if execution_mode == "sequential":
            return self._execute_sequential(query, conversation_history, session_id, requirements)
        else:
            return self._execute_parallel(query, conversation_history, session_id, requirements)

    def _execute_sequential(self, query: str, conversation_history: List[BaseMessage], session_id: str, requirements: List[AgentRequirement]) -> Dict[str, Any]:
        """
        Execute agents sequentially based on priority order.

        Args:
            query: Original user query
            conversation_history: Conversation history
            session_id: Session identifier
            requirements: List of agent requirements

        Returns:
            Dict containing results from all executed agents
        """
        # Sort requirements by priority
        sorted_requirements = sorted(requirements, key=lambda x: x["priority"])

        current_input = query
        self.agent_results = {}

        for requirement in sorted_requirements:
            agent_name = requirement["agent_name"]
            input_from = requirement["input_from"]

            # Determine input for this agent
            if input_from and input_from in self.agent_results:
                # Use output from previous agent as input
                previous_result = self.agent_results[input_from]
                if previous_result.get("success", False):
                    current_input = previous_result["output"]
                    print(f"📥 Using previous agent output as input for {agent_name}")
                else:
                    print(f"❌ Previous agent {input_from} failed, using original query")
                    current_input = query
            else:
                # Use original query
                current_input = query

            # Execute the agent
            result = self._execute_single_agent(
                agent_name, current_input, conversation_history, session_id, requirement["parameters"]
            )

            self.agent_results[agent_name] = result

        return self.agent_results

    def _execute_parallel(self, query: str, conversation_history: List[BaseMessage], session_id: str, requirements: List[AgentRequirement]) -> Dict[str, Any]:
        """
        Execute agents in parallel (simplified - run sequentially for now).

        Args:
            query: Original user query
            conversation_history: Conversation history
            session_id: Session identifier
            requirements: List of agent requirements

        Returns:
            Dict containing results from all executed agents
        """
        # For simplicity, execute sequentially even in "parallel" mode
        # In a real implementation, this could use threading or async execution
        return self._execute_sequential(query, conversation_history, session_id, requirements)

    def _execute_single_agent(self, agent_name: str, input_text, conversation_history: List[BaseMessage], session_id: str, parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a single agent with the given input.

        Args:
            agent_name: Name of the agent to execute
            input_text: Input text for the agent
            conversation_history: Conversation history
            session_id: Session identifier
            parameters: Additional parameters for the agent

        Returns:
            Dict containing agent execution results
        """
        # Route to appropriate agent execution method based on agent name
        execution_methods = {
            "RAG_AGENT": self._execute_rag_agent,
            "WEB_SEARCH_PROCESSOR_AGENT": self._execute_web_search_agent,
            "CRYPTO_SENTIMENT_AGENT": self._execute_sentiment_agent,
            "FORWARD_LOOKING_STATEMENT_AGENT": self._execute_fls_agent,
            "CONVERSATION_AGENT": self._execute_conversation_agent
        }

        execution_method = execution_methods[agent_name]

        # Execute the agent with appropriate parameters
        if agent_name == "RAG_AGENT":
            result = execution_method(input_text, conversation_history, session_id)
        elif agent_name == "WEB_SEARCH_PROCESSOR_AGENT":
            result = execution_method(input_text, conversation_history)
        elif agent_name in ["CRYPTO_SENTIMENT_AGENT", "FORWARD_LOOKING_STATEMENT_AGENT"]:
            result = execution_method(input_text)
        elif agent_name == "CONVERSATION_AGENT":
            result = execution_method(input_text, conversation_history)
        else:
            result = execution_method(input_text, conversation_history, session_id)

        return result

    def _execute_rag_agent(self, query: str, conversation_history: str, session_id: str) -> Dict[str, Any]:
        """Execute RAG agent."""
        rag_agent = self.agents["RAG_AGENT"]

        # Execute RAG query
        response = rag_agent.process_query(query, chat_history=conversation_history, session_id=session_id)

        return {
            "agent": "RAG_AGENT",
            "success": True,
            "output": response.get("response", ""),
            "sources": response.get("sources", []),
            "confidence": response.get("confidence", 0.0)
        }

    def _execute_web_search_agent(self, query: str, conversation_history: str) -> Dict[str, Any]:
        """Execute web search agent."""
        web_search_agent = self.agents["WEB_SEARCH_PROCESSOR_AGENT"]

        # Execute web search
        response = web_search_agent.process_web_search_results(query=query, chat_history=conversation_history)

        # Extract text content for sentiment analysis if response is structured
        text_content = response
        if isinstance(response, dict) and "response" in response:
            text_content = response["response"]

        # Ensure we return text content suitable for sentiment analysis
        return {
            "agent": "WEB_SEARCH_PROCESSOR_AGENT",
            "success": True,
            "output": text_content,
            "raw_response": response,
            "sources": response.get("sources", []) if isinstance(response, dict) else []
        }

    def _execute_sentiment_agent(self, text) -> Dict[str, Any]:
        """Execute crypto sentiment agent."""
        sentiment_agent = self.agents["CRYPTO_SENTIMENT_AGENT"]

        # Handle different input types (string or AIMessage)
        if hasattr(text, 'content'):
            # It's an AIMessage object
            text_content = text.content
        else:
            # It's already a string
            text_content = str(text)

        # If text is very long (likely from web search), truncate it
        if len(text_content) > 2000:
            print(f"📝 Sentiment input too long ({len(text_content)} chars), using first 2000 chars")
            text_content = text_content[:2000] + "..."

        # Execute sentiment analysis
        result = sentiment_agent.analyze_sentiment(text_content)

        return {
            "agent": "CRYPTO_SENTIMENT_AGENT",
            "success": True,
            "output": f"Sentiment Analysis Result:\n- Sentiment: {result['sentiment']}\n- Confidence: {result['confidence']:.3f}\n- Text Length: {len(text_content)} characters",
            "sentiment": result['sentiment'],
            "confidence": result['confidence'],
            "analyzed_text": text_content
        }

    def _execute_fls_agent(self, text: str) -> Dict[str, Any]:
        """Execute forward-looking statement agent."""
        fls_agent = self.agents["FORWARD_LOOKING_STATEMENT_AGENT"]

        text_content = str(text) if text else ""

        # Execute FLS detection
        result = fls_agent.detect_fls(text_content)

        return {
            "agent": "FORWARD_LOOKING_STATEMENT_AGENT",
            "success": True,
            "output": f"Is FLS: {result['is_fls']}, Confidence: {result['confidence']:.3f}",
            "is_fls": result['is_fls'],
            "confidence": result['confidence'],
            "reasoning": result['reason']
        }

    def _execute_conversation_agent(self, query: str, conversation_history: List[BaseMessage]) -> Dict[str, Any]:
        """Execute conversation agent."""
        # Generate conversation response
        conversation_prompt = f"User query: {query}"
        response = self.config.conversation.llm.invoke(conversation_prompt)

        # Extract response content
        if hasattr(response, 'content'):
            response_content = response.content
        else:
            response_content = str(response)

        return {
            "agent": "CONVERSATION_AGENT",
            "success": True,
            "output": response_content
        }


class OutputCombiner:
    """
    Combines outputs from multiple agents into a coherent final response.
    """

    def __init__(self, config: Config):
        """Initialize the output combiner."""
        self.config = config
        self.llm = config.conversation.llm

    def combine_outputs(self, agent_results: Dict[str, Any], original_query: str, strategy: str = "intelligent") -> str:
        """
        Intelligently combine outputs from multiple agents based on content analysis.

        Args:
            agent_results: Results from executed agents
            original_query: The original user query
            strategy: Combination strategy (uses intelligent analysis)

        Returns:
            Intelligently combined final response
        """
        # Analyze agent results to determine best combination approach
        successful_agents = {name: result for name, result in agent_results.items() if result.get("success", False)}

        # Use intelligent combination based on agent types and content
        return self._intelligently_combine_outputs(successful_agents, original_query)

    def _intelligently_combine_outputs(self, agent_results: Dict[str, Any], original_query: str) -> str:
        """
        Intelligently combine agent outputs into a cohesive, natural response.

        Uses LLM to create a unified response that flows naturally without
        relying on hardcoded structures or agent-specific formatting.
        """
        # Extract all successful agent outputs
        successful_results = {name: result for name, result in agent_results.items() if result.get("success", False)}

        # Create a comprehensive synthesis prompt for natural response generation
        synthesis_prompt = self._create_natural_synthesis_prompt(successful_results, original_query)

        # Use LLM to generate a unified, natural response
        response = self.llm.invoke(synthesis_prompt)
        return response.content if hasattr(response, 'content') else str(response)

    def _create_natural_synthesis_prompt(self, agent_results: Dict[str, Any], original_query: str) -> str:
        """Create a prompt for LLM to synthesize agent outputs into a natural response."""

        # Collect all agent outputs in a simple format
        all_outputs = []
        for agent_name, result in agent_results.items():
            output = result.get("output", "")
            if output:
                # Convert to string if it's an AIMessage or other object
                output_str = str(output.content) if hasattr(output, 'content') else str(output)
                all_outputs.append(output_str)

        combined_outputs = "\n\n".join(all_outputs)

        prompt = f"""
You are an expert financial analyst providing a comprehensive response to: "{original_query}"

I have gathered relevant information from multiple sources. Please create a single, cohesive, professionally written response that:

1. **Flows naturally** - Write as if you're providing expert analysis in one voice
2. **Is well-structured** - Organize information logically with clear, descriptive headers
3. **Presents information objectively** - Combine insights from different sources seamlessly
4. **Is comprehensive but concise** - Include all key information without repetition
5. **Uses professional, accessible language** - Explain concepts clearly for the reader

Information available:
{combined_outputs}

Create a response that reads like a single expert analysis, naturally organized into logical sections based on the content. Let the information itself determine the best structure - don't force it into predetermined sections.
"""

        return prompt

