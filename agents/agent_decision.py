"""
Agent Decision System for Multi-Agent Crypto/Financial Assistant

This module handles the orchestration of different agents using LangGraph.
It dynamically routes user queries to the appropriate agent based on content and context.
"""

import json
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
import os, getpass, uuid
from dotenv import load_dotenv
from agents.rag_agent import CryptoFinancialRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.crypto_sentiment_agent import CryptoSentimentAgent
from agents.forward_looking_statement_agent import ForwardLookingStatementAgent
from agents.guardrails.local_guardrails import LocalGuardrails
from agents.multi_agent_coordinator import MultiAgentCoordinator, AgentPipeline, OutputCombiner
from utils.redis_cache import get_cache_manager
from utils.mongodb_manager import get_mongodb_manager
from utils.memory_tools import get_memory_manager, save_conversation_summary, get_conversation_history

from langgraph.checkpoint.memory import MemorySaver

import cv2
import numpy as np

from config import Config

load_dotenv()

# Load configuration
config = Config()

# Initialize memory
memory = MemorySaver()

# Specify a thread
thread_config = {"configurable": {"thread_id": "1"}}


    # Agent that takes the decision of routing the request further to correct task specific agent
class AgentConfig:
    """Configuration settings for the agent decision system."""

    # Decision model
    DECISION_MODEL = "gpt-4o"  # or whichever model you prefer

    # Vision model for image analysis (now used for crypto sentiment analysis)
    VISION_MODEL = "gpt-4o"

    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85

    # System instructions for the decision agent
    DECISION_SYSTEM_PROMPT = """You are an intelligent crypto and financial analysis triage system that routes user queries to
    the appropriate specialized agent. Your job is to analyze the user's request and determine which agent
    is best suited to handle it based on the query content, tone analysis needs, and conversation context.

    Available agents:
    1. CONVERSATION_AGENT - For general chat, greetings, and non-crypto questions.
    2. RAG_AGENT - For specific crypto/financial knowledge questions that can be answered from the CoinGecko 2024 Annual Crypto Industry Report and related financial literature.
    3. WEB_SEARCH_PROCESSOR_AGENT - For questions about recent crypto developments, current market trends, or time-sensitive financial information.
    4. CRYPTO_SENTIMENT_AGENT - For analyzing sentiment and tone in crypto/financial TEXT ONLY (NOT numerical data or tables), classifying as Positive, Neutral, or Negative.
    5. FORWARD_LOOKING_STATEMENT_AGENT - For detecting and analyzing forward-looking statements, predictions, and expectations in financial/crypto reports.

    Make your decision based on these guidelines:
    - If the user asks about sentiment analysis of TEXTUAL CONTENT (articles, tweets, news, reports), route to CRYPTO_SENTIMENT_AGENT.
      * DO NOT route to CRYPTO_SENTIMENT_AGENT if the query mentions: "table data", "numbers", "revenue figures", "numerical data", or any structured data.
      * CRYPTO_SENTIMENT_AGENT only analyzes text sentiment, not numerical/tabular data.
    - If the user asks about predictions, forecasts, expectations, or forward-looking statements in crypto/financial reports, route to FORWARD_LOOKING_STATEMENT_AGENT.
    - If the user asks about recent crypto developments, market trends, or current financial situations, use the web search processor agent.
    - If the user asks specific questions about crypto/financial data, tables, numbers, or information from reports like CoinGecko, use the RAG agent.
    - For general conversation, greetings, or non-crypto questions, use the conversation agent.
    - If uncertain about which agent to use, default to CONVERSATION_AGENT.

    You must provide your answer in JSON format with the following structure:
    {{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }}
    """

    crypto_sentiment_agent = CryptoSentimentAgent(azure_endpoint=config.crypto_analysis.crypto_sentiment_endpoint)
    fls_agent = ForwardLookingStatementAgent(azure_endpoint=config.crypto_analysis.fls_azure_endpoint)


class AgentState(MessagesState):
    """State maintained across the workflow."""
    # messages: List[BaseMessage]  # Conversation history
    agent_name: Optional[str]  # Current active agent
    current_input: Optional[Union[str, Dict]]  # Input to be processed
    has_image: bool  # Whether the current input contains an image
    image_type: Optional[str]  # Type of crypto/financial image if present
    output: Optional[str]  # Final output to user
    needs_human_validation: bool  # Whether human validation is required
    retrieval_confidence: float  # Confidence in retrieval (for RAG agent)
    bypass_routing: bool  # Flag to bypass agent routing for guardrails
    insufficient_info: bool  # Flag indicating RAG response has insufficient information
    session_id: str  # Unique session identifier for caching
    # Multi-agent support
    is_multi_agent: bool  # Whether this query requires multiple agents
    multi_agent_results: Optional[Dict[str, Any]]  # Results from multi-agent pipeline
    primary_agent: Optional[str]  # Primary agent for multi-agent coordination
    pipeline_config: Optional[Dict[str, Any]]  # Pipeline configuration for multi-agent execution


class AgentDecision(TypedDict):
    """Output structure for the decision agent."""
    agent: str
    reasoning: str
    confidence: float


def create_agent_graph():
    """Create and configure the LangGraph for agent orchestration."""

    # Initialize guardrails with the same LLM used elsewhere
    guardrails = LocalGuardrails(config.rag.llm)

    # Initialize MongoDB memory system
    mongodb_manager = get_mongodb_manager()
    memory_store = mongodb_manager.get_store()
    checkpointer = mongodb_manager.get_saver()
    print("✅ MongoDB memory system enabled")

    # LLM
    decision_model = config.agent_decision.llm

    # Initialize the output parser
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)

    # Create the decision prompt
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])

    # Create the decision chain
    decision_chain = decision_prompt | decision_model | json_parser
    
    # Define graph state transformations
    def analyze_input(state: AgentState) -> AgentState:
        """Analyze the input to detect images and determine input type."""
        current_input = state["current_input"]
        has_image = False
        image_type = None
        
        # Get the text from the input
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Check input through guardrails if text is present
        if input_text:
            is_allowed, message = guardrails.check_input(input_text)
            if not is_allowed:
                # If input is blocked, return early with guardrail message
                print(f"Selected agent: INPUT GUARDRAILS, Message: ", message)
                return {
                    **state,
                    "messages": message,
                    "agent_name": "INPUT_GUARDRAILS",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True  # flag to end flow
                }
        
        # Original image processing code
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image", None)
            image_type_response = AgentConfig.image_analyzer.analyze_image(image_path)
            image_type = image_type_response['image_type']
            print("ANALYZED IMAGE TYPE: ", image_type)
        
        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,
            "bypass_routing": False  # Explicitly set to False for normal flow
        }
    
    def check_if_bypassing(state: AgentState) -> str:
        """Check if we should bypass normal routing due to guardrails."""
        if state.get("bypass_routing", False):
            return "apply_guardrails"
        return "route_to_agent"
    
    def route_to_agent(state: AgentState) -> AgentState:
        """Make decision about which agent should handle the query or if multi-agent coordination is needed."""
        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state["has_image"]
        image_type = state["image_type"]

        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        print(f"🔍 DECISION INPUT: '{input_text}'")
        print(f"🖼️ Has image: {has_image}, Type: {image_type}")

        # Create context from recent conversation history (last 3 messages)
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges (6 messages)  # Not provided control from config
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"

        # Initialize multi-agent coordinator
        coordinator = MultiAgentCoordinator(config)

        # Check if this query needs multiple agents
        is_multi_agent = coordinator.is_multi_agent_query(input_text, recent_context)

        print(f"🔍 MULTI-AGENT CHECK: {is_multi_agent}")

        if is_multi_agent:
            print("🔍 MULTI-AGENT QUERY DETECTED - Using Multi-Agent Coordinator")

            # Analyze query complexity and get pipeline configuration
            coordination_result = coordinator.analyze_query_complexity(input_text, recent_context)

            print(f"🔍 COORDINATION RESULT: {coordination_result}")

            # Check if coordination suggests multiple agents
            if coordination_result.get("needs_multiple_agents", False):
                # Create pipeline configuration
                pipeline_config = coordinator.create_pipeline_config(coordination_result)

                print(f"🔍 MULTI-AGENT PIPELINE CREATED: {[req['agent_name'] for req in pipeline_config['requirements']]}")

                # Update state for multi-agent processing
                updated_state = {
                    **state,
                    "agent_name": "run_multi_agent_pipeline",
                    "is_multi_agent": True,
                    "primary_agent": coordination_result.get("primary_agent"),
                    "multi_agent_results": None,
                    "pipeline_config": pipeline_config  # Store pipeline config in state
                }

                return updated_state
            else:
                print("❌ COORDINATION FAILED - Falling back to single agent decision")
                is_multi_agent = False

        # Single agent decision (original logic)
        if not is_multi_agent:
            print("🔍 SINGLE AGENT QUERY - Using standard decision logic")

            # Combine everything for the decision input
            decision_input = f"""
            User query: {input_text}

            Recent conversation context:
            {recent_context}

            Has image: {has_image}
            Image type: {image_type if has_image else 'None'}

            Based on this information, which agent should handle this query?
            """

            # Make the decision
            decision = decision_chain.invoke({"input": decision_input})

            # Decided agent
            print(f"Decision: {decision['agent']}")
            print(f"Reasoning: {decision['reasoning']}")
            print(f"Confidence: {decision['confidence']}")

            # Update state with decision
            updated_state = {
                **state,
                "agent_name": decision["agent"],
                "is_multi_agent": False,
                "primary_agent": None,
                "multi_agent_results": None
            }

            # Route directly to selected agent (no confidence check)
            return updated_state

        # This should not be reached, but handle gracefully
        return state

    # Define agent execution functions (these will be implemented in their respective modules)
    def run_conversation_agent(state: AgentState) -> AgentState:
        """Handle general conversation."""

        print(f"Selected agent: CONVERSATION_AGENT")

        messages = state["messages"]
        current_input = state["current_input"]

        # Get MongoDB store
        mongodb_manager = get_mongodb_manager()
        memory_store = mongodb_manager.get_store()

        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        # Create context from recent conversation history
        recent_context = ""
        for msg in messages:#[-20:]:  # Get last 10 exchanges (20 messages)  # currently considering complete history - limit control from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"
        
        # Get relevant memories from MongoDB store
        memory_results = memory_store.search(
            ("user_memories",),  # namespace_prefix as positional argument
            query=f"{input_text} {recent_context[-500:]}",  # Use query + recent context
            limit=3
        )
        relevant_memories = "\n".join([
            f"- {memory.content}" for memory in memory_results
        ]) if memory_results else ""

        # Combine everything for the decision input
        memories_section = f"Relevant memories from previous conversations: {relevant_memories}\n" if relevant_memories else ""

        conversation_prompt = (
            f"User query: {input_text}\n\n"
            f"Recent conversation context: {recent_context}\n\n"
            f"{memories_section}\n"
            "You are an AI-powered Crypto/Financial Conversation Assistant. Your goal is to facilitate smooth and informative conversations with users, handling both casual and crypto/financial-related queries. You must respond naturally while ensuring financial accuracy and clarity.\n\n"
            "### Role & Capabilities\n"
            "- Engage in **general conversation** while maintaining professionalism.\n"
            "- Answer **crypto/financial questions** using verified knowledge from CoinGecko reports and market data.\n"
            "- Route **complex queries** to RAG (retrieval-augmented generation) or web search if needed.\n"
            "- Handle **follow-up questions** while keeping track of conversation context.\n"
            "- Redirect **sentiment analysis** or **forward-looking statement detection** requests to appropriate specialized agents.\n\n"
            "### Guidelines for Responding:\n"
            "1. **General Conversations:**\n"
            "- If the user engages in casual talk (e.g., greetings, small talk), respond in a friendly, engaging manner.\n"
            "- Keep responses **concise and engaging**, unless a detailed answer is needed.\n\n"
            "2. **Crypto/Financial Questions:**\n"
            "- If you have **high confidence** in answering, provide accurate crypto/financial information.\n"
            "- Ensure responses are **clear, concise, and factual**.\n"
            "- Always include appropriate disclaimers for financial information.\n\n"
            "3. **Follow-Up & Clarifications:**\n"
            "- Maintain conversation history for better responses.\n"
            "- If a query is unclear, ask **follow-up questions** before answering.\n\n"
            "4. **Handling Specialized Analysis:**\n"
            "- Do **not** attempt to perform sentiment analysis or detect forward-looking statements yourself.\n"
            "- If user asks for **sentiment analysis** of crypto/financial text, route to CRYPTO_SENTIMENT_AGENT.\n"
            "- If user asks for **forward-looking statement detection** in financial reports, route to FORWARD_LOOKING_STATEMENT_AGENT.\n"
            "- After analysis, **help the user interpret the results**.\n\n"
            "5. **Uncertainty & Ethical Considerations:**\n"
            "- If unsure, **never assume** financial facts or provide investment advice.\n"
            "- Recommend consulting a **licensed financial advisor** for investment decisions.\n"
            "- Avoid providing **specific investment recommendations** or **financial advice**.\n"
            "- Always include disclaimers: \"This is not financial advice. Please consult with a qualified financial professional.\"\n\n"
            "### Response Format:\n"
            "- Maintain a **conversational yet professional tone**.\n"
            "- Use **bullet points or numbered lists** for clarity when needed.\n"
            "- If pulling from external sources (RAG/Web Search), mention **where the information is from** (e.g., \"According to CoinGecko 2024 Annual Report...\").\n"
            "- Always include appropriate financial disclaimers.\n\n"
            "### Example User Queries & Responses:\n\n"
            "**User:** \"Hey, how's your day going?\"\n"
            "**You:** \"I'm here and ready to help! How can I assist you with crypto or financial questions today?\"\n\n"
            "**User:** \"What's the current sentiment around Bitcoin?\"\n"
            "**You:** \"I'll route this to our sentiment analysis agent for accurate assessment. Please note this is not financial advice.\"\n\n"
            "**User:** \"Should I invest in Ethereum?\"\n"
            "**You:** \"I'm not qualified to provide investment advice. Please consult with a licensed financial advisor for personalized investment decisions.\"\n\n"
            "Conversational LLM Response:"
        )

        # print("Conversation Prompt:", conversation_prompt)

        response = config.conversation.llm.invoke(conversation_prompt)

        # print("Conversation respone:", response)

        # Get memory manager
        memory_manager = get_memory_manager()

        # Save conversation summary
        if len(messages) > 2:  # Only save if there's a real conversation
            # Create a summary of the conversation
            conversation_summary = f"User asked about: {input_text}. Assistant provided information about crypto/financial topics."
            # Call the function directly instead of as a tool
            save_conversation_summary.func(conversation_summary, ["crypto", "financial"])

        # response = AIMessage(content="This would be handled by the conversation agent.")

        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT"
        }
    
    def run_rag_agent(state: AgentState) -> AgentState:
        """Handle crypto/financial knowledge queries using RAG."""
        # Initialize the RAG agent

        print(f"🎯 RUNNING RAG_AGENT")
        print(f"Query: {state['current_input']}")

        rag_agent = CryptoFinancialRAG(config)
        
        messages = state["messages"]
        query = state["current_input"]
        rag_context_limit = config.rag.context_limit

        recent_context = ""
        for msg in messages[-rag_context_limit:]:# limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        response = rag_agent.process_query(query, chat_history=recent_context, session_id=state.get("session_id"))
        retrieval_confidence = response.get("confidence", 0.0)  # Default to 0.0 if not provided

        print(f"Retrieval Confidence: {retrieval_confidence}")
        print(f"Sources: {len(response['sources'])}")

        # Check if response indicates insufficient information
        # New logic: Use validation_failed flag from enhanced RAG agent
        insufficient_info = response.get("validation_failed", False)
        
        response_content = response["response"]
        
        # Extract the content properly based on type
        if isinstance(response_content, dict) and hasattr(response_content, 'content'):
            # If it's an AIMessage or similar object with a content attribute
            response_text = response_content.content
        else:
            # If it's already a string
            response_text = response_content
            
        print(f"Response text type: {type(response_text)}")
        print(f"Response text full content: {response_text}")

        # Removed fallback text checking - rely only on validation_failed flag from RAG agent

        if insufficient_info:
            print("RAG response indicates insufficient information (validation_failed flag)")
            print(f"Validation failed flag: {response.get('validation_failed', False)}")
            if 'validation_reasoning' in response:
                print(f"Validation reasoning: {response['validation_reasoning']}")
            print(f"Response text full content: {response_text}")

        print(f"Insufficient info flag set to: {insufficient_info}")

        # Store RAG output based on context validation (not threshold)
        if not insufficient_info:
            # Context validation passed - use the response
            response_output = AIMessage(content=response_text)
        else:
            # Context validation failed - empty response to trigger web search
            response_output = AIMessage(content="")
        
        return {
            **state,
            "output": response_output,
            "needs_human_validation": False,  # Assuming no validation needed for RAG responses
            "retrieval_confidence": retrieval_confidence,
            "agent_name": "RAG_AGENT",
            "insufficient_info": insufficient_info
        }

    # Web Search Processor Node
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        """Handles web search results, processes them with LLM, and generates a refined response."""

        print(f"Selected agent: WEB_SEARCH_PROCESSOR_AGENT")
        print("[WEB_SEARCH_PROCESSOR_AGENT] Processing Web Search Results...")
        
        messages = state["messages"]
        web_search_context_limit = config.web_search.context_limit

        recent_context = ""
        for msg in messages[-web_search_context_limit:]: # limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        web_search_processor = WebSearchProcessorAgent(config)

        processed_response = web_search_processor.process_web_search_results(query=state["current_input"], chat_history=recent_context)

        # print("######### DEBUG WEB SEARCH:", processed_response)
        
        if state['agent_name'] != None:
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"
        else:
            involved_agents = "WEB_SEARCH_PROCESSOR_AGENT"

        # Overwrite any previous output with the processed Web Search response
        return {
            **state,
            # "output": "This would be handled by the web search agent, finding the latest information.",
            "output": processed_response,
            "agent_name": involved_agents
        }

    # Define Routing Logic  
    def validation_based_routing(state: AgentState) -> Dict[str, str]:
        """Route based on RAG context validation (not confidence threshold)."""
        # Debug prints
        print(f"🔀 NEW ROUTING - Insufficient info flag: {state.get('insufficient_info', False)}")
        print(f"🔀 NEW ROUTING - Context validation result from RAG agent")
        
        # Redirect ONLY if context validation failed
        if state.get("insufficient_info", False):
            print("🔀 NEW ROUTING - Re-routed to Web Search Agent due to LLM context validation failure!")
            return "WEB_SEARCH_PROCESSOR_AGENT"
        else:
            print("🔀 NEW ROUTING - Context validation passed - proceeding with RAG response!")
            return "check_validation"
    
    def run_crypto_sentiment_agent(state: AgentState) -> AgentState:
        """Handle crypto sentiment analysis."""

        print(f"Selected agent: CRYPTO_SENTIMENT_AGENT")

        # Get text content for analysis
        current_input = state["current_input"]
        if isinstance(current_input, str):
            text_to_analyze = current_input
        elif isinstance(current_input, dict):
            text_to_analyze = current_input.get("text", "")
        else:
            text_to_analyze = str(current_input)

        # Use LLM to intelligently determine if this query is appropriate for sentiment analysis
        validation_prompt = f"""
        Analyze if this query is appropriate for sentiment analysis. Sentiment analysis is for analyzing emotional tone and sentiment in textual content like articles, tweets, news, or reports.

        Query: {text_to_analyze}

        Decision criteria:
        - APPROPRIATE: Queries about sentiment, tone, emotional analysis of text content
        - INAPPROPRIATE: Queries about numerical data, tables, financial figures, statistics, or structured data

        Respond with only "APPROPRIATE" or "INAPPROPRIATE":
        """

        # Use LLM to validate query appropriateness
        validation_response = config.conversation.llm.invoke(validation_prompt)
        validation_result = validation_response.content if hasattr(validation_response, 'content') else str(validation_response)
        validation_result = validation_result.strip().upper()

        if "INAPPROPRIATE" in validation_result:
            # Use LLM to generate contextual response for inappropriate query
            sentiment_prompt = f"""
            User query: {text_to_analyze}

            This query appears to be about numerical or tabular data rather than textual content for sentiment analysis.
            Please provide a helpful response explaining that sentiment analysis is for text content like articles, tweets, or news,
            and suggest alternatives for numerical data queries.

            Provide a professional, helpful response.
            """
        else:
            # Analyze sentiment using the model
            sentiment_result = AgentConfig.crypto_sentiment_agent.analyze_sentiment(text_to_analyze)

            # Create prompt for LLM with sentiment analysis results
            sentiment_prompt = f"""
            Original Query: {text_to_analyze}

            Sentiment Analysis Results:
            - Sentiment: {sentiment_result['sentiment']}
            - Confidence: {sentiment_result['confidence']:.3f}
            - Probabilities: {sentiment_result.get('probabilities', {})}

            Please provide a comprehensive, professional response about this sentiment analysis.
            Include:
            1. A clear summary of the sentiment analysis results
            2. Interpretation of what this sentiment means in crypto/financial context
            3. Any relevant insights or observations
            4. Professional disclaimers about investment decisions

            Make the response informative, balanced, and actionable.
            """

        # Use LLM to generate the final response
        response = config.conversation.llm.invoke(sentiment_prompt)

        return {
            **state,
            "output": response,
            "needs_human_validation": False,
            "agent_name": "CRYPTO_SENTIMENT_AGENT"
        }
    
    def run_forward_looking_statement_agent(state: AgentState) -> AgentState:
        """Handle forward-looking statement detection."""

        print(f"Selected agent: FORWARD_LOOKING_STATEMENT_AGENT")

        # Get text content for analysis
        current_input = state["current_input"]
        if isinstance(current_input, str):
            text_to_analyze = current_input
        elif isinstance(current_input, dict):
            text_to_analyze = current_input.get("text", "")
        else:
            text_to_analyze = str(current_input)

        # Detect forward-looking statements using the model
        fls_result = AgentConfig.fls_agent.detect_fls(text_to_analyze)

        # Create prompt for LLM with FLS analysis results
        fls_prompt = f"""
        Original Query: {text_to_analyze}

        Forward-Looking Statement Analysis Results:
        - Is Forward-Looking Statement: {fls_result['is_fls']}
        - Confidence: {fls_result['confidence']:.3f}
        - Reasoning: {fls_result['reason']}

        Please provide a comprehensive, professional response about this forward-looking statement analysis.
        Include:
        1. A clear summary of the FLS detection results
        2. Explanation of what forward-looking statements mean in financial/crypto context
        3. Analysis of the confidence level and reasoning provided
        4. Practical implications for investors and analysts
        5. Professional disclaimers about the nature of forward-looking information

        Make the response informative, balanced, and actionable for financial analysis.
        """

        # Use LLM to generate the final response
        response = config.conversation.llm.invoke(fls_prompt)

        return {
            **state,
            "output": response,
            "needs_human_validation": False,
            "agent_name": "FORWARD_LOOKING_STATEMENT_AGENT"
        }

    def run_multi_agent_pipeline(state: AgentState) -> AgentState:
        """Execute multi-agent pipeline for complex queries."""
        print(f"Selected agent: MULTI_AGENT_PIPELINE")

        # Debug: Check what's in state
        print(f"🔍 STATE DEBUG: is_multi_agent={state.get('is_multi_agent')}, pipeline_config exists={state.get('pipeline_config') is not None}")

        # Get pipeline configuration from state
        pipeline_config = state.get("pipeline_config")

        if not pipeline_config:
            print("❌ No pipeline configuration found in state")
            print(f"🔍 STATE KEYS: {list(state.keys())}")
            # Return error state instead of falling back to conversation agent
            error_response = "Unable to execute multi-agent pipeline due to missing configuration. Please try again."
            return {
                **state,
                "output": error_response,
                "agent_name": "MULTI_AGENT_PIPELINE_ERROR",
                "is_multi_agent": True,
                "multi_agent_results": None,
                "primary_agent": state.get("primary_agent")
            }

        # Initialize pipeline executor
        pipeline = AgentPipeline(config, pipeline_config)

        # Execute the pipeline
        print(f"🚀 EXECUTING MULTI-AGENT PIPELINE: {pipeline_config['pipeline_id']}")
        print(f"📋 REQUIREMENTS: {[req['agent_name'] for req in pipeline_config['requirements']]}")

        agent_results = pipeline.execute_pipeline(
            query=state["current_input"],
            conversation_history=state["messages"],
            session_id=state["session_id"]
        )

        print(f"✅ PIPELINE EXECUTION COMPLETE")
        print(f"📊 RESULTS: {list(agent_results.keys())}")

        # Initialize output combiner
        combiner = OutputCombiner(config)

        # Combine outputs
        combined_response = combiner.combine_outputs(
            agent_results=agent_results,
            original_query=state["current_input"],
            strategy=pipeline_config["output_combination_strategy"]
        )

        print(f"🔄 OUTPUT COMBINED - Length: {len(combined_response)}")

        # Update state with results
        return {
            **state,
            "output": combined_response,
            "agent_name": f"MULTI_AGENT_PIPELINE:{','.join(agent_results.keys())}",
            "is_multi_agent": True,
            "multi_agent_results": agent_results,
            "primary_agent": state.get("primary_agent")
        }

    # Note: Skin lesion agent removed as we're now focused on crypto/financial analysis
    # def run_skin_lesion_agent(state: AgentState) -> AgentState:
    #     """Handle skin lesion image analysis."""
    #     # This agent has been replaced with crypto sentiment and FLS agents
    
    def handle_human_validation(state: AgentState) -> Dict:
        """Prepare for human validation if needed."""
        if state.get("needs_human_validation", False):
            return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
        return {"agent_state": state, "next": END}
    
    def perform_human_validation(state: AgentState) -> AgentState:
        """Handle human validation process."""
        print(f"Selected agent: HUMAN_VALIDATION")

        # Append validation request to the existing output
        validation_prompt = f"{state['output'].content}\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."

        # Create an AI message with the validation prompt
        validation_message = AIMessage(content=validation_prompt)

        return {
            **state,
            "output": validation_message,
            "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"
        }

    # Check output through guardrails
    def apply_output_guardrails(state: AgentState) -> AgentState:
        """Apply output guardrails to the generated response."""
        output = state["output"]
        current_input = state["current_input"]

        # Check if output is valid
        if not output or not isinstance(output, (str, AIMessage)):
            return state

        output_text = output if isinstance(output, str) else output.content
        
        # If the last message was a human validation message
        if "Human Validation Required" in output_text:
            # Check if the current input is a human validation response
            validation_input = ""
            if isinstance(current_input, str):
                validation_input = current_input
            elif isinstance(current_input, dict):
                validation_input = current_input.get("text", "")
            
            # If validation input exists
            if validation_input.lower().startswith(('yes', 'no')):
                # Add the validation result to the conversation history
                validation_response = HumanMessage(content=f"Validation Result: {validation_input}")
                
                # If validation is 'No', modify the output
                if validation_input.lower().startswith('no'):
                    fallback_message = AIMessage(content="The previous crypto/financial analysis requires further review. A financial professional has flagged potential inaccuracies.")
                    return {
                        **state,
                        "messages": [validation_response, fallback_message],
                        "output": fallback_message
                    }
                
                return {
                    **state,
                    "messages": validation_response
                }
        
        # Get the original input text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Apply output sanitization
        sanitized_output = guardrails.check_output(output_text, input_text)
        # sanitized_output = output_text
        
        # For non-validation cases, add the sanitized output to messages
        sanitized_message = AIMessage(content=sanitized_output) if isinstance(output, AIMessage) else sanitized_output
        
        return {
            **state,
            "messages": sanitized_message,
            "output": sanitized_message
        }

    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("run_multi_agent_pipeline", run_multi_agent_pipeline)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("CRYPTO_SENTIMENT_AGENT", run_crypto_sentiment_agent)
    workflow.add_node("FORWARD_LOOKING_STATEMENT_AGENT", run_forward_looking_statement_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)
    
    # Define the edges (workflow connections)
    workflow.set_entry_point("analyze_input")
    # workflow.add_edge("analyze_input", "route_to_agent")
    # Add conditional routing for guardrails bypass
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "route_to_agent": "route_to_agent"
        }
    )
    
    # Connect decision router to agents
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x.get("next", x.get("agent_name", "CONVERSATION_AGENT")),
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "CRYPTO_SENTIMENT_AGENT": "CRYPTO_SENTIMENT_AGENT",
            "FORWARD_LOOKING_STATEMENT_AGENT": "FORWARD_LOOKING_STATEMENT_AGENT",
            "run_multi_agent_pipeline": "run_multi_agent_pipeline",
            "needs_validation": "RAG_AGENT"  # Default to RAG if confidence is low
        }
    )
    
    # Connect agent outputs to validation check
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_conditional_edges("RAG_AGENT", validation_based_routing)
    workflow.add_edge("CRYPTO_SENTIMENT_AGENT", "check_validation")
    workflow.add_edge("FORWARD_LOOKING_STATEMENT_AGENT", "check_validation")
    workflow.add_edge("run_multi_agent_pipeline", "check_validation")

    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)
    
    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: "apply_guardrails"  # Route to guardrails instead of END
        }
    )
    
    # workflow.add_edge("human_validation", END)

    # Compile the graph with appropriate checkpointer
    return workflow.compile(checkpointer=checkpointer, store=memory_store)


def init_agent_state(session_id: Optional[str] = None) -> AgentState:
    """Initialize the agent state with default values."""
    if session_id is None:
        session_id = str(uuid.uuid4())

    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False,
        "session_id": session_id,
        # Multi-agent support
        "is_multi_agent": False,
        "multi_agent_results": None,
        "primary_agent": None,
        "pipeline_config": None
    }


def process_query(query: Union[str, Dict], conversation_history: List[BaseMessage] = None, session_id: Optional[str] = None) -> str:
    """
    Process a user query through the agent decision system.

    Args:
        query: User input (text string or dict with text and image)
        conversation_history: Optional list of previous messages, NOT NEEDED ANYMORE since the state saves the conversation history now
        session_id: Optional session identifier for caching

    Returns:
        Response from the appropriate agent
    """
    # Initialize the graph
    graph = create_agent_graph()

    # # Save Graph Flowchart
    # image_bytes = graph.get_graph().draw_mermaid_png()
    # decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    # cv2.imwrite("./assets/graph.png", decoded)
    # print("Graph flowchart saved in assets.")

    # Initialize state
    state = init_agent_state(session_id)

    # Get cache manager
    cache_manager = get_cache_manager()

    # Load session memory from cache if available
    if state["session_id"]:
        cached_memory = cache_manager.get_session_memory(state["session_id"])
        if cached_memory:
            # Convert cached memory to LangChain message format
            messages = []
            for msg in cached_memory:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            state["messages"] = messages

    # if conversation_history:
    #     state["messages"] = conversation_history

    # Add the current query
    state["current_input"] = query

    # To handle image upload case
    if isinstance(query, dict):
        query_text = query.get("text", "") + ", user uploaded an image for diagnosis."
    else:
        query_text = query

    state["messages"].append(HumanMessage(content=query_text))

    # result = graph.invoke(state, thread_config)
    result = graph.invoke(state, thread_config)
    # print("######### DEBUG 4:", result)
    # state["messages"] = [result["messages"][-1].content]

    # Update session memory cache
    if result.get("session_id"):
        # Convert LangChain messages to cache format
        memory_data = []
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                memory_data.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                memory_data.append({"role": "assistant", "content": msg.content})

        # Cache the updated session memory
        cache_manager.set_session_memory(
            result["session_id"], memory_data,
            ttl_hours=config.cache.session_memory_ttl_hours
        )

    # Keep history to reasonable size (ANOTHER OPTION: summarize and store before truncating history)
    if len(result["messages"]) > config.max_conversation_history:  # Keep last config.max_conversation_history messages
        result["messages"] = result["messages"][-config.max_conversation_history:]

    # visualize conversation history in console
    for m in result["messages"]:
        m.pretty_print()

    # Add the response to conversation history
    return result