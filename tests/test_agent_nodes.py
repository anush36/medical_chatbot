# Copyright (c) 2026 Anush Agarwal. All rights reserved.
# This code is proprietary and provided for public review and educational purposes.
# Unauthorized use, reproduction, or distribution is strictly prohibited.

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from backend.agent import safety_node, verify_node, AgentState

def mock_llm_response(content: str):
    mock_response = MagicMock()
    mock_response.content = content
    return mock_response

# --- SAFETY NODE TESTS ---

@patch("backend.agent.llm")
def test_safety_node_safe(mock_llm):
    # Mock LLM returning 'SAFE'
    mock_llm.invoke.return_value = mock_llm_response("SAFE")
    
    state = {
        "draft_response": "I recommend you wash that cut with soap.",
        "draft_attempts": 0
    }
    
    result = safety_node(state)
    
    assert result["safety_feedback"] == "SAFE"
    assert "Safety checker APPROVED" in result["thought_logs"][0]

@patch("backend.agent.llm")
def test_safety_node_unsafe(mock_llm):
    # Mock LLM returning 'UNSAFE'
    mock_llm.invoke.return_value = mock_llm_response("UNSAFE: Do not try to perform surgery at home.")
    
    state = {
        "draft_response": "You should cut the splinter out yourself.",
        "draft_attempts": 0
    }
    
    result = safety_node(state)
    
    assert "UNSAFE: Do not try to perform surgery at home." in result["safety_feedback"]
    assert "🛑 Safety checker REJECTED" in result["thought_logs"][0]
    assert result["draft_attempts"] == 1
    assert "messages" not in result # Shouldn't output a final message unless attempts >= 3

@patch("backend.agent.llm")
def test_safety_node_unsafe_max_attempts(mock_llm):
    mock_llm.invoke.return_value = mock_llm_response("UNSAFE: Still dangerous.")
    
    state = {
        "draft_response": "Take a high dose of this unverified supplement.",
        "draft_attempts": 2  # next attempt will be 3
    }
    
    result = safety_node(state)
    
    assert result["draft_attempts"] == 3
    assert "messages" in result
    assert "could not be fully verified for safety" in result["messages"][0].content

# --- VERIFY NODE TESTS ---

@patch("backend.agent.llm")
def test_verify_node_grounded(mock_llm):
    # Mock LLM returning 'GROUNDED'
    mock_llm.invoke.return_value = mock_llm_response("GROUNDED")
    
    state = {
        "draft_response": "Flu vaccines reduce the risk of illness.",
        "extracted_claims": ["flu vaccine efficacy"],
        "gathered_literature": ["--- Source: PMC123 (Title: Flu Vaccines) ---"],
        "draft_attempts": 0
    }
    
    result = verify_node(state)
    
    assert result["validation_feedback"] == "GROUNDED"
    assert "Medical validator APPROVED" in result["thought_logs"][0]
    assert "messages" in result # Passes verify step

@patch("backend.agent.llm")
def test_verify_node_not_grounded(mock_llm):
    # Mock LLM returning 'NOT GROUNDED'
    mock_llm.invoke.return_value = mock_llm_response("NOT GROUNDED: The claim about preventing 100% of cases is missing.")
    
    state = {
        "draft_response": "Flu vaccines prevent 100% of illness.",
        "extracted_claims": ["flu vaccine efficacy"],
        "gathered_literature": ["--- Source: PMC123 (Title: Flu Vaccines) ---"],
        "draft_attempts": 0
    }
    
    result = verify_node(state)
    
    assert "The claim about preventing 100% of cases is missing" in result["validation_feedback"]
    assert "❌ Validator REJECTED" in result["thought_logs"][0]
    assert result["draft_attempts"] == 1
    assert "messages" not in result

@patch("backend.agent.llm")
def test_verify_node_not_grounded_max_attempts(mock_llm):
    mock_llm.invoke.return_value = mock_llm_response("NOT GROUNDED: Still unverified.")
    
    state = {
        "draft_response": "Flu vaccines prevent 100% of illness.",
        "extracted_claims": ["flu vaccine efficacy"],
        "gathered_literature": ["--- Source: PMC123 (Title: Flu Vaccines) ---"],
        "draft_attempts": 2 # next attempt will be 3
    }
    
    result = verify_node(state)
    
    assert result["draft_attempts"] == 3
    assert "messages" in result
    assert "could not be fully validated against retrieved literature." in result["messages"][0].content
