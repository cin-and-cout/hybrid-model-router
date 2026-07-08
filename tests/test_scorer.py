import pytest
from routing_agent.scorer import (
    clean_string,
    exact_match_scorer,
    json_format_scorer,
    token_overlap_scorer,
    evaluate_task
)

def test_clean_string():
    assert clean_string("  Hello World!  ") == "hello world"
    assert clean_string("```json\n{\"test\": 123}\n```") == "test 123"
    assert clean_string("Yes.") == "yes"

def test_exact_match_scorer():
    assert exact_match_scorer("  Yes  ", "yes.") == 1.0
    assert exact_match_scorer("No", "Yes") == 0.0

def test_json_format_scorer():
    # Valid JSON
    assert json_format_scorer('{"name": "test"}') == 1.0
    # Valid JSON with expected keys
    assert json_format_scorer('{"name": "test", "age": 20}', ["name", "age"]) == 1.0
    # Missing required key
    assert json_format_scorer('{"name": "test"}', ["name", "age"]) == 0.0
    # Invalid JSON
    assert json_format_scorer('{"name": "test"') == 0.0
    # Markdown wrapped JSON
    assert json_format_scorer('```json\n{"name": "test"}\n```', ["name"]) == 1.0

def test_token_overlap_scorer():
    # Complete match
    assert token_overlap_scorer("hello world", "hello world") == 1.0
    # Partial match
    # words of s1: {"hello", "world"}, words of s2: {"hello"}
    # intersection: {"hello"} (size 1)
    # union: {"hello", "world"} (size 2)
    # Jaccard: 1/2 = 0.5
    assert token_overlap_scorer("hello world", "hello") == 0.5
    # No match
    assert token_overlap_scorer("hello", "world") == 0.0
    # Both empty strings
    assert token_overlap_scorer("   ", "") == 1.0

def test_evaluate_task():
    # Math routing
    assert evaluate_task("math", " 42 ", "42.") == 1.0
    # Structured output routing (JSON format pass + overlap)
    assert evaluate_task(
        "structured_output", 
        '{"capital": "Paris"}', 
        '{"capital": "Paris"}', 
        ["capital"]
    ) == 1.0
    # Structured output routing with invalid JSON
    assert evaluate_task(
        "structured_output", 
        'invalid', 
        'Paris', 
        ["capital"]
    ) == 0.0
