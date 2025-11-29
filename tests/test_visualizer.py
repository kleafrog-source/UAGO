import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import plotly.graph_objects as go
import numpy as np

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualizer.uago_viz import render_model, mistral_transform

# --- Test Data ---
EXAMPLE_JSON_IFS = {
    "phases": { "phase5": {
        "model": "IFS (Iterated Function System)",
        "formula": "f(x,y)=(0.5*x, 0.5*y + 0.5), f(x,y)=(0.5*x + 0.5, 0.5*y)",
        "parameters": { "dimensionality": 1.0, "scale": 0.5 }
    }}
}

EXAMPLE_JSON_UNPARSABLE_IFS = {
    "phases": { "phase5": {
        "model": "IFS (Iterated Function System)",
        "formula": "this is not a valid formula",
        "parameters": { "dimensionality": 1.67, "complexity": 42 }
    }}
}

EXAMPLE_JSON_NON_IFS = {
    "phases": { "phase5": {
        "model": "L-System",
        "parameters": { "dimensionality": 1.26, "complexity": 10 }
    }}
}

# --- Tests for render_model (Bug Fix Verification) ---

def test_render_model_prioritizes_phase5_ifs():
    """Verify render_model creates a scatter plot for a valid IFS model."""
    fig = render_model(EXAMPLE_JSON_IFS)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scattergl)

def test_render_model_fallback_for_unparsable_ifs():
    """Verify the new fallback for unparsable IFS creates a line plot of invariants."""
    fig = render_model(EXAMPLE_JSON_UNPARSABLE_IFS)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scatter) # Fallback is a line plot (Scatter)
    assert "dimensionality" in fig.data[0].x
    assert fig.layout.title.text == "Numerical Invariants for IFS (Iterated Function System)"

def test_render_model_fallback_for_non_ifs():
    """Verify fallback for other models is still a line plot of invariants."""
    fig = render_model(EXAMPLE_JSON_NON_IFS)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scatter)
    assert "complexity" in fig.data[0].x

# --- Tests for mistral_transform (New Feature) ---

MISTRAL_API_KEY = "test_api_key"

@patch('visualizer.uago_viz.Mistral')
def test_mistral_transform_success(mock_mistral):
    """Test the successful path of mistral_transform: API call -> code extraction -> exec -> figure."""
    # Mock the Mistral API response structure explicitly
    mock_message = MagicMock()
    mock_message.content = (
        "```python\n"
        "fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))\n"
        "```"
    )
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [mock_choice]
    mock_mistral.return_value.chat.complete.return_value = mock_chat_response

    fig, log = mistral_transform(EXAMPLE_JSON_IFS, MISTRAL_API_KEY)

    # Verify that a figure was returned and is the one from the exec'd code
    assert isinstance(fig, go.Figure)
    assert np.array_equal(fig.data[0].x, [1, 2])

    # Verify the log entry
    assert "prompt" in log
    assert "response" in log
    assert "error" not in log

@patch('visualizer.uago_viz.Mistral')
def test_mistral_transform_api_failure_returns_fallback(mock_mistral):
    """Test that mistral_transform returns a fallback figure on API error."""
    # Mock an API failure
    mock_mistral.return_value.chat.complete.side_effect = Exception("API Error")

    fig, log = mistral_transform(EXAMPLE_JSON_IFS, MISTRAL_API_KEY)

    # Verify it returned the fallback figure
    assert isinstance(fig.data[0], go.Scatter)
    assert fig.layout.title.text == "Numerical Invariants for IFS (Iterated Function System)"

    # Verify the error was logged
    assert "error" in log
    assert "Mistral API call failed" in log["error"]

def test_mistral_transform_no_api_key_returns_none():
    """Test that mistral_transform returns None if no API key is provided."""
    fig, log = mistral_transform(EXAMPLE_JSON_IFS, api_key="")

    # Verify it returned None
    assert fig is None

    # Verify the reason was logged
    assert "error" in log
    assert "Mistral API key not provided" in log["error"]

@patch('visualizer.uago_viz.Mistral')
def test_mistral_transform_bad_code_returns_fallback(mock_mistral):
    """Test that mistral_transform returns a fallback when response has no code block."""
    # Mock a response with no runnable code
    mock_message = MagicMock()
    mock_message.content = "I cannot generate this visualization."
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [mock_choice]
    mock_mistral.return_value.chat.complete.return_value = mock_chat_response

    fig, log = mistral_transform(EXAMPLE_JSON_IFS, MISTRAL_API_KEY)

    # Verify it returned the fallback figure
    assert isinstance(fig.data[0], go.Scatter)

    # Verify the error was logged
    assert "error" in log
    assert "No Python code block found" in log["error"]

@patch('visualizer.uago_viz.Mistral')
def test_mistral_transform_with_safe_builtins_succeeds(mock_mistral):
    """Test that code using whitelisted built-ins (e.g., range) executes successfully."""
    mock_message = MagicMock()
    mock_message.content = (
        "```python\n"
        "x_vals = list(range(5))\n"
        "fig = go.Figure(data=go.Scatter(x=x_vals, y=[i**2 for i in x_vals]))\n"
        "```"
    )
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [mock_choice]
    mock_mistral.return_value.chat.complete.return_value = mock_chat_response

    fig, log = mistral_transform(EXAMPLE_JSON_IFS, MISTRAL_API_KEY)

    assert isinstance(fig, go.Figure)
    assert np.array_equal(fig.data[0].x, [0, 1, 2, 3, 4])
    assert "error" not in log

@patch('visualizer.uago_viz.Mistral')
def test_mistral_transform_with_unsafe_builtin_falls_back(mock_mistral):
    """Test that using a non-whitelisted built-in (e.g., open) fails and triggers a fallback."""
    mock_message = MagicMock()
    mock_message.content = (
        "```python\n"
        "with open('some_file.txt', 'w') as f:\n"
        "    f.write('malicious content')\n"
        "fig = go.Figure()\n"
        "```"
    )
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [mock_choice]
    mock_mistral.return_value.chat.complete.return_value = mock_chat_response

    fig, log = mistral_transform(EXAMPLE_JSON_IFS, MISTRAL_API_KEY)

    # Verify that it returned the fallback figure (a line plot of invariants)
    assert isinstance(fig.data[0], go.Scatter)
    assert fig.layout.title.text == "Numerical Invariants for IFS (Iterated Function System)"

    # Verify the error was logged
    assert "error" in log
    assert "Execution failed" in log["error"]
