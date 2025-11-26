import json
import pytest
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualizer.uago_viz import render_model

# The example JSON data provided in the task description
EXAMPLE_JSON = {
    "timestamp": "2025-11-26T12:35:22.303587",
    "input_shape": [898, 1200, 3],
    "phases": {
        "phase5": {
            "model": "IFS (Iterated Function System)",
            "formula": "The IFS consists of the following affine transformations:\n1. f1(x, y) = (0.5x, 0.5y + 0.5)\n2. f2(x, y) = (0.5x + 0.5, 0.5y)\n3. f3(x, y) = (0.5x, 0.5y)\n4. f4(x, y) = (0.5x + 0.5, 0.5y + 0.5)",
            "parameters": {
                "dimensionality": 1.6740240726199072,
                "scales": [0.5, 0.33, 0.25],
                "symmetry_group": "C4",
                "rotation_angles": [0, 90, 180, 270],
                "repetition": True,
                "connectivity": "high"
            }
        }
    }
}

NON_IFS_JSON = {
    "phases": {
        "phase5": {
            "model": "L-system",
            "parameters": { "dimensionality": 1.2 }
        }
    }
}

def test_render_model_with_ifs_data():
    """
    Tests that render_model generates a non-empty HTML string for IFS models.
    """
    html_output = render_model(EXAMPLE_JSON)
    assert html_output is not None
    assert isinstance(html_output, str)
    assert "p5.js" in html_output
    assert "canvas" in html_output
    assert "Controls" in html_output
    assert "<strong>Model:</strong> IFS (Iterated Function System)" in html_output

def test_render_model_with_non_ifs_data():
    """
    Tests that render_model returns a placeholder for non-IFS models.
    """
    html_output = render_model(NON_IFS_JSON)
    assert "Visualization for this model is currently in development" in html_output
    assert "Model: L-system" in html_output

def test_parser_fallback_mechanism():
    """
    Tests that the fallback mechanism is triggered when the formula is unparsable.
    """
    bad_formula_json = EXAMPLE_JSON.copy()
    bad_formula_json["phases"]["phase5"]["formula"] = "This is not a valid formula"
    html_output = render_model(bad_formula_json)
    assert "Could not parse formula" in html_output
    # Check if it still renders the main template
    assert "p5.js" in html_output
