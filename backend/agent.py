# Copyright (c) 2026 Anush Agarwal. All rights reserved.
# This code is proprietary and provided for public review and educational purposes.
# Unauthorized use, reproduction, or distribution is strictly prohibited.

import logging
import operator
from typing import Annotated, TypedDict, Sequence, Literal, List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

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
    extracted_claims: List[str]
    validation_feedback: str
    safety_feedback: str
    gathered_literature: Annotated[List[str], operator.add]
    draft_attempts: int
    thought_logs: Annotated[List[str], operator.add]

# --- LLM Setup ---
import time
import httpx

_GCP_TOKEN_CACHE = {"token": None, "expires_at": 0}

class GCPAuth(httpx.Auth):
    def __init__(self, base_url: str):
        self.base_url = base_url

    def auth_flow(self, request: httpx.Request):
        global _GCP_TOKEN_CACHE
        current_time = time.time()
        
        if not _GCP_TOKEN_CACHE["token"] or current_time > _GCP_TOKEN_CACHE["expires_at"]:
            token = self._fetch_token()
            if token:
                _GCP_TOKEN_CACHE["token"] = token
                _GCP_TOKEN_CACHE["expires_at"] = current_time + 3000
        
        if _GCP_TOKEN_CACHE["token"]:
            request.headers["Authorization"] = f"Bearer {_GCP_TOKEN_CACHE['token']}"
        yield request

    def _fetch_token(self):
        try:
            import google.auth
            from google.auth.transport.requests import Request
            from google.oauth2 import id_token
            
            target_audience = self.base_url.split("/v1")[0] if "/v1" in self.base_url else self.base_url
            req = Request()
            token = id_token.fetch_id_token(req, target_audience)
            if token:
                return token
        except Exception as e:
            logger.debug(f"Direct ID token fetch failed: {e}")

        try:
            import subprocess
            logger.info("Falling back to gcloud CLI for Identity Token...")
            result = subprocess.run(
                ["gcloud", "auth", "print-identity-token"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception as e:
            logger.error(f"Could not fetch GCP ID token: {e}")
            return None

http_client = None
if config.MODEL_PROVIDER != "openai" and config.MEDGEMMA_BASE_URL and "run.app" in config.MEDGEMMA_BASE_URL:
    http_client = httpx.Client(auth=GCPAuth(config.MEDGEMMA_BASE_URL), timeout=60.0)

# Initialize the OpenAI compatible LLM using the current config (this allows hitting local vllm/medgemma too)
llm_kwargs = {
    "api_key": config.OPENAI_API_KEY if config.MODEL_PROVIDER == "openai" else config.MEDGEMMA_API_KEY,
    "model": config.OPENAI_MODEL if config.MODEL_PROVIDER == "openai" else config.MEDGEMMA_MODEL,
    "max_tokens": config.OPENAI_MAX_TOKENS,
    "temperature": config.OPENAI_TEMPERATURE,
}

if config.MODEL_PROVIDER != "openai" and config.MEDGEMMA_BASE_URL:
    base_url = config.MEDGEMMA_BASE_URL
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    llm_kwargs["base_url"] = base_url

if http_client:
    llm_kwargs["http_client"] = http_client

llm = ChatOpenAI(**llm_kwargs)

def draft_node(state: AgentState):
    logger.info("--- DRAFT NODE ---")
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    last_user_query = user_msgs[-1].content if user_msgs else "No query found."
    
    val_feedback = state.get("validation_feedback", "")
    safety_feedback = state.get("safety_feedback", "")
    
    if "UNSAFE" in safety_feedback.upper(): # Refinement for safety
        literature = "\n\n".join(state.get("gathered_literature", []))
        sys_prompt = f"""You are a medical author correcting your previous draft based on a safety evaluation.
        
        SAFETY FEEDBACK: {safety_feedback}
        
        Your task is to rewrite your previous draft to incorporate the safety feedback:
        1. Remove or modify any advice that the safety feedback identified as potentially harmful.
        2. Where claims are supported by the literature, inject a superscript HTML citation (e.g., <sup>1</sup>, <sup>2</sup>) immediately after it.
        3. Append a "References" section at the end if citations are used. Do NOT hallucinate PMC IDs. Only use the PMC IDs from the literature provided below.
        4. Maintain the approachable, educational tone from your original draft.
        
        -- GATHERED LITERATURE --
        {literature}
        """
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=f"Previous Draft:\n{state['draft_response']}")])
        thoughts = ["✏️ Rewrote draft to incorporate safety feedback."]
        return {"draft_response": response.content, "thought_logs": thoughts}

    elif val_feedback and "GROUNDED" not in val_feedback.upper(): # Refinement pass for validation
        literature = "\n\n".join(state.get("gathered_literature", []))
        sys_prompt = f"""You are a medical author correcting your previous draft based on validation feedback and peer-reviewed literature.
        
        FEEDBACK: {val_feedback}
        
        Your task is to rewrite your previous draft to incorporate the feedback:
        1. Correct or remove any unvalidated claims that contradict or are not found in the literature.
        2. Where claims are supported by the literature, inject a superscript HTML citation (e.g., <sup>1</sup>, <sup>2</sup>) immediately after it.
        3. Append a "References" section at the end if citations are used. Do NOT hallucinate PMC IDs. Only use the PMC IDs from the literature provided below.
        4. Maintain the approachable, educational tone from your original draft.
        
        -- GATHERED LITERATURE --
        {literature}
        """
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=f"Previous Draft:\n{state['draft_response']}")])
        thoughts = ["✏️ Rewrote draft to incorporate feedback and add citations."]
        return {"draft_response": response.content, "thought_logs": thoughts}
        
    else: # First pass
        sys_prompt = """You are an expert medical educator and clinician communicating with the general public.
        Your goal is to provide a comprehensive, educational, and approachable answer to the user's question.
        Take an educating role: explain concepts clearly, venture into relevant detail, and ensure the user gets a solid understanding.
        Draft the response based on your internal knowledge. Do not worry about citing literature yet.
        If the question is not medical, just respond in a friendly, conversational manner.
        """
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=last_user_query)])
        thoughts = ["✍️ Drafted initial educational response directly using internal knowledge."]
        return {"draft_response": response.content, "thought_logs": thoughts}

def safety_node(state: AgentState):
    logger.info("--- SAFETY NODE ---")
    draft = state["draft_response"]
    
    sys_prompt = """You are a medical safety agent. Your priority is to ensure the response provided to the user does not indicate anything that is likely to cause the user or others harm.
    
    For example, if someone says they have a deep splinter and it needs to be digged out, the response should tell them to go to a doctor, NOT to try to dig it out at home.
    However, there is nuance: if someone has a shallow splinter, it is ok to suggest carefully removing it at home.
    This applies to all medical advice: do not tell the user to do something that could cause them harm.
    
    Evaluate the following DRAFT RESPONSE.
    Rules:
    1. If the draft constitutes safe advice or is simply informational and does not pose a safety risk, output EXACTLY 'SAFE'.
    2. If the draft contains harmful advice, output 'UNSAFE: [Explain exactly what the harmful advice is and how it should be corrected]'.
    """
    
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=f"DRAFT RESPONSE:\n{draft}")])
    content = response.content.strip()
    
    if "UNSAFE" in content.upper():
        feedback = content.replace("UNSAFE:", "").strip()
        if not feedback:
            feedback = content.replace("UNSAFE", "").strip()
        thoughts = [f"🛑 Safety checker REJECTED draft: {feedback}"]
        
        attempts = state.get("draft_attempts", 0) + 1
        return_payload = {
            "safety_feedback": f"UNSAFE: {feedback}",
            "thought_logs": thoughts,
            "draft_attempts": attempts
        }
        if attempts >= 3:
            thoughts.append("⚠️ Max attempts reached in safety check. Releasing best-effort draft with a safety warning.")
            return_payload["messages"] = [AIMessage(content=f"{draft}\n\n*Note: This response could not be fully verified for safety.*")]
            
        return return_payload
    else:
        thoughts = ["Safety checker APPROVED draft - claims and advice does not pose safety risk"]
        return {
            "safety_feedback": "SAFE",
            "thought_logs": thoughts
        }

def extract_claims_node(state: AgentState):
    logger.info("--- EXTRACT CLAIMS NODE ---")
    draft = state["draft_response"]
    
    sys_prompt = """You are a medical researcher identifying claims to validate.
    Analyze the provided draft response and extract the core medical facts, treatments, or assertions that need grounding in peer-reviewed literature.
    Output each distinct concept as a short, keyword-based PubMed Central (PMC) search query on a NEW LINE.
    Do NOT output full sentences. Use AND/OR operators if helpful.
    If the draft contains no testable medical claims (e.g., just conversational), output exactly: NONE
    
    Example Output:
    "acute myeloid leukemia" AND ("treatments" OR "therapy")
    "flu complications" AND "respiratory"
    """
    
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=f"Draft:\n{draft}")])
    content = response.content.strip()
    
    if content == "NONE" or not content or "NONE" in content.upper():
        thoughts = ["🧠 Analyzed draft. Decision: No medical claims to validate, skipping PMC search."]
        return {"extracted_claims": [], "thought_logs": thoughts}
        
    queries = [q.strip() for q in content.split("\n") if q.strip()]
    
    # Cap slightly so we don't spam PMC
    queries = queries[:3]
    thoughts = [f"🔬 Extracted {len(queries)} specific medical queries to validate: {', '.join(queries)}"]
    
    return {"extracted_claims": queries, "thought_logs": thoughts}

def retrieve_node(state: AgentState):
    logger.info("--- RETRIEVE NODE ---")
    queries = state.get("extracted_claims", [])
    if not queries:
        return {"pmc_queries_count": 0, "gathered_literature": []}
        
    all_literature = []
    thoughts = []
    
    for q in queries:
        result_str = search_pmc.invoke({"query": q})
        all_literature.append(result_str)
        
        import re
        titles = re.findall(r"--- Source: (PMC\d+) \(Title: (.*?)\) ---", result_str)
        if titles:
            parsed = ", ".join([f"[{t[0]}] {t[1]}" for t in titles])
            thoughts.append(f"📚 Retrieved articles for '{q}': {parsed}")
        else:
            thoughts.append(f"⚠️ No readable articles found for '{q}'.")
            
    return {
        "pmc_queries_count": len(queries),
        "gathered_literature": all_literature,
        "thought_logs": thoughts
    }

def verify_node(state: AgentState):
    logger.info("--- VERIFY NODE ---")
    draft = state["draft_response"]
    literature = "\n\n".join(state.get("gathered_literature", []))
    
    if not state.get("extracted_claims"):
         # No claims were extracted initially, or it's non-medical
         thoughts = ["✅ No medical validation required."]
         return {
             "validation_feedback": "GROUNDED",
             "messages": [AIMessage(content=draft)],
             "thought_logs": thoughts
         }

    sys_prompt = f"""You are a strict and unforgiving medical validator.
    Evaluate the following DRAFT RESPONSE against the GATHERED LITERATURE.
    Rules:
    1. If the DRAFT RESPONSE already contains citations pointing to valid PMC IDs below and correctly reflects the literature with NO unvalidated claims, output EXACTLY 'GROUNDED'.
    2. Any specific medical claims in the draft MUST be grounded in the gathered literature.
    3. If the draft contains unvalidated claims that contradict or are not found in the literature at all, output 'NOT GROUNDED: [Explain which specific claims are invalid or missing from literature]'.
    
    -- GATHERED LITERATURE --
    {literature}
    
    -- DRAFT RESPONSE --
    {draft}
    """
    response = llm.invoke([SystemMessage(content=sys_prompt)])
    content = response.content.strip()
    
    if "NOT GROUNDED" in content.upper():
        feedback = content.replace("NOT GROUNDED:", "").strip()
        if not feedback:
            feedback = content.replace("NOT GROUNDED", "").strip()
        thoughts = [f"❌ Validator REJECTED draft: {feedback}"]
        attempts = state.get("draft_attempts", 0) + 1
        
        return_payload = {
            "validation_feedback": feedback,
            "draft_attempts": attempts,
            "thought_logs": thoughts
        }
        
        if attempts >= 3:
            thoughts.append("⚠️ Max attempts reached. Releasing best-effort draft.")
            return_payload["messages"] = [AIMessage(content=f"{draft}\n\n*Note: This response could not be fully validated against retrieved literature.*")]
            
        return return_payload
    else:
        thoughts = ["Medical validator APPROVED draft - all claims are supported by retrieved literature"]
        return {
            "validation_feedback": "GROUNDED",
            "messages": [AIMessage(content=draft)],
            "draft_attempts": state.get("draft_attempts", 0) + 1,
            "thought_logs": thoughts
        }

def route_after_safety(state: AgentState) -> str:
    safety_feedback = state.get("safety_feedback", "")
    attempts = state.get("draft_attempts", 0)
    
    if "UNSAFE" in safety_feedback.upper():
        if attempts >= 3:
            return "__end__"
        return "draft"
        
    val_feedback = state.get("validation_feedback", "")
    if val_feedback and "GROUNDED" not in val_feedback.upper():
        return "verify"
    return "extract"

def route_after_verify(state: AgentState) -> Literal["__end__", "draft"]:
    attempts = state.get("draft_attempts", 0)
    feedback = state.get("validation_feedback", "")
    
    if "GROUNDED" in feedback.upper() or attempts >= 3:
        return "__end__"
    return "draft"

def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("draft", draft_node)
    workflow.add_node("safety", safety_node)
    workflow.add_node("extract", extract_claims_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("verify", verify_node)
    
    workflow.set_entry_point("draft")
    
    workflow.add_edge("draft", "safety")
    
    workflow.add_conditional_edges(
        "safety",
        route_after_safety,
        {
            "draft": "draft",
            "extract": "extract",
            "verify": "verify",
            "__end__": END
        }
    )
    
    workflow.add_edge("extract", "retrieve")
    workflow.add_edge("retrieve", "verify")
    
    workflow.add_conditional_edges(
        "verify",
        route_after_verify,
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
        content = msg["content"]
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=content))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=content))
            
    # Context Window Management: Retain only the last 3 conversational turns (6 messages)
    if len(lc_messages) > 6:
        lc_messages = lc_messages[-6:]
        # Ensure the slice starts with a HumanMessage if we cut halfway through a turn
        if isinstance(lc_messages[0], AIMessage):
            lc_messages = lc_messages[1:]
            
    try:
        final_state = agent_graph.invoke(
            {
                "messages": lc_messages, 
                "pmc_queries_count": 0,
                "draft_response": "",
                "extracted_claims": [],
                "validation_feedback": "",
                "safety_feedback": "",
                "gathered_literature": [],
                "draft_attempts": 0,
                "thought_logs": []
            }
        )
        # The final state's last message is the approved draft from the verify node
        final_msg = final_state["messages"][-1]
        
        # Extract thoughts
        thought_process = final_state.get("thought_logs", [])
        
        return {
            "response": final_msg.content,
            "finish_reason": "stop",
            "intermediate_steps": thought_process,
            "safety_status": final_state.get("safety_feedback"),
            "validation_status": final_state.get("validation_feedback")
        }
    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        return {
             "response": f"Sorry, the agentic workflow encountered an internal error: {e}",
             "finish_reason": "error",
             "intermediate_steps": [f"Error occurred: {e}"],
             "safety_status": None,
             "validation_status": None
        }
