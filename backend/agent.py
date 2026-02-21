import logging
import operator
from typing import Annotated, TypedDict, Sequence, Literal, List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from backend.config import config
from backend.pmc_tool import search_pmc

logger = logging.getLogger(__name__)

# --- State Schema ---
class AgentState(TypedDict):
    """The state of the agentic graph."""
    # List of messages. We use operator.add to append to the list rather than overwrite
    messages: Annotated[Sequence[BaseMessage], operator.add]
    pmc_queries_count: int
    draft_response: str
    gathered_literature: Annotated[List[str], operator.add]

# --- LLM Setup ---
# Initialize the OpenAI compatible LLM using the current config (this allows hitting local vllm/medgemma too)
llm = ChatOpenAI(
    api_key=config.OPENAI_API_KEY if config.MODEL_PROVIDER == "openai" else config.MEDGEMMA_API_KEY,
    base_url=config.MEDGEMMA_BASE_URL if config.MODEL_PROVIDER != "openai" else None,
    model=config.OPENAI_MODEL if config.MODEL_PROVIDER == "openai" else config.MEDGEMMA_MODEL,
    max_tokens=config.OPENAI_MAX_TOKENS,
    temperature=config.OPENAI_TEMPERATURE
)

# Bind the tool to the LLM for the research phase
tools = [search_pmc]
llm_with_tools = llm.bind_tools(tools)

# --- Nodes ---
def researcher_node(state: AgentState):
    """Decides whether to search for more literature or finalize research."""
    logger.info("--- RESEARCHER NODE ---")
    messages = state.get("messages", [])
    query_count = state.get("pmc_queries_count", 0)
    
    # Simple system prompt for the tool-calling researcher
    sys_msg = SystemMessage(content="""You are a medical researcher. 
    Your goal is to gather relevant literature from PubMed Central using the `search_pmc` tool to answer the user's latest query.
    If you feel you have enough information to answer the question, output a final thought summarizing what you found without calling the tool.
    If you do not have enough info, use the search_pmc tool. DO NOT ATTEMPT TO ANSWER YET. Just gather data.""")
    
    # Pass system message + history to LLM
    response = llm_with_tools.invoke([sys_msg] + list(messages))
    
    return {
        "messages": [response], 
        "pmc_queries_count": query_count
    }

def tool_executor(state: AgentState):
    """Executes the tool call and updates query count"""
    logger.info("--- TOOL EXECUTOR NODE ---")
    # Wrap standard ToolNode lightly just to increment the counter and save the context
    tool_node = ToolNode(tools)
    result = tool_node.invoke(state)
    
    # Extract just what the tool returned
    new_messages = result["messages"]
    gathered = []
    for msg in new_messages:
        gathered.append(msg.content)
        
    return {
        "messages": new_messages,
        "pmc_queries_count": state["pmc_queries_count"] + 1,
        "gathered_literature": gathered
    }

def draft_node(state: AgentState):
    """Drafts the final response to the user based on gathered literature."""
    logger.info("--- DRAFT NODE ---")
    
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    last_user_query = user_msgs[-1].content if user_msgs else "No query found."
    
    literature = "\n\n".join(state.get("gathered_literature", []))
    
    sys_prompt = f"""You are a helpful medical chatbot. 
    Your task is to answer the user's query based on the provided literature.
    You must ground all medical advice and statements in the provided literature. You may summarize the literature into easily understandable terms.
    If the provided literature completely lacks relevant information to answer the core of the question, explicitly state that you do not have enough information to answer definitively.
    
    -- GATHERED LITERATURE --
    {literature}
    -------------------------
    """
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=last_user_query)])
    return {"draft_response": response.content}

def verify_node(state: AgentState):
    """Validates that the drafted response is grounded in the literature."""
    logger.info("--- VERIFY NODE ---")
    
    draft = state["draft_response"]
    literature = "\n\n".join(state.get("gathered_literature", []))
    
    sys_prompt = f"""You are a strict medical validator.
    Evaluate the following DRAFT RESPONSE against the GATHERED LITERATURE.
    Rules:
    1. Any specific medical advice or clinical statements in the draft MUST be grounded in the gathered literature.
    2. The draft can summarize concepts found in the literature into layman's terms. 
    3. If the draft makes up treatments or specific medical facts not found in the text, reject it.
    
    Output exactly 'GROUNDED' if it passes.
    Otherwise, output 'NOT GROUNDED: [Explain what claims are floating/hallucinated]'.
    
    -- GATHERED LITERATURE --
    {literature}
    
    -- DRAFT RESPONSE --
    {draft}
    """
    response = llm.invoke([SystemMessage(content=sys_prompt)])
    content = response.content.strip()
    
    logger.info(f"Validation result: {content}")
    
    if content.startswith("GROUNDED"):
        return {"messages": [AIMessage(content=draft)]} # Final output to user
    else:
        # Pass the feedback back as a system message to the drafter
        feedback = AIMessage(content=f"Your previous draft was rejected by the medical validator. Reason: {content}. Please rewrite the draft to clearly adhere only to the gathered literature, or state you don't know if the literature lacks the details.")
        return {"messages": [feedback]}

# --- Routing Logic ---
def route_research(state: AgentState) -> Literal["tools", "draft"]:
    """Route from researcher to tool node or draft node."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM called a tool
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        if state["pmc_queries_count"] >= 5:
            logger.warning("Max PMC queries reached (5). Forcing transition to draft.")
            return "draft"
        return "tools"
    
    return "draft"

def route_validation(state: AgentState) -> Literal["__end__", "draft"]:
    """Route from validation to end or back to draft."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message is an AI message containing the final drafted response (meaning it passed), end.
    # The verify_node appends an AIMessage only when 'GROUNDED'. When rejected, it passes feedback but we don't end.
    if isinstance(last_message, AIMessage) and "rejected by the medical validator" not in last_message.content:
         return "__end__"
         
    return "draft"

# --- Build Graph ---
def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("tools", tool_executor)
    workflow.add_node("draft", draft_node)
    workflow.add_node("verify", verify_node)
    
    # Graph entry and topology
    workflow.set_entry_point("researcher")
    
    workflow.add_conditional_edges(
        "researcher",
        route_research,
        {
            "tools": "tools",
            "draft": "draft"
        }
    )
    workflow.add_edge("tools", "researcher")
    workflow.add_edge("draft", "verify")
    
    workflow.add_conditional_edges(
        "verify",
        route_validation,
        {
            "draft": "draft",
            "__end__": END
        }
    )
    
    return workflow.compile()

# Global compiled graph
agent_graph = build_agent_graph()

def generate_agentic_response(history: List[Dict[str, str]]) -> Dict[str, str]:
    """Execute the agent graph given a message history."""
    
    # Convert dict history to Langchain Messages
    lc_messages = []
    for msg in history:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
            
    try:
        final_state = agent_graph.invoke(
            {
                "messages": lc_messages, 
                "pmc_queries_count": 0,
                "gathered_literature": [],
                "draft_response": ""
            }
        )
        # The final state's last message is the approved draft from the verify node
        final_msg = final_state["messages"][-1]
        
        return {
            "response": final_msg.content,
            "finish_reason": "stop"
        }
    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        return {
             "response": f"Sorry, the agentic workflow encountered an internal error: {e}",
             "finish_reason": "error"
        }
