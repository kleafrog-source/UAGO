import pytest
import sys
import os
import plotly.graph_objects as go

# Add the root directory to the Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualizer.uago_viz import render_model, _parse_ifs_formula_with_sympy

# Example JSON data for testing
EXAMPLE_JSON_IFS = {
    "phases": {
        "phase5": {
            "model": "IFS (Iterated Function System)",
            "formula": "f(x,y)=(0.5*x, 0.5*y + 0.5), f(x,y)=(0.5*x + 0.5, 0.5*y)",
            "parameters": { "dimensionality": 1.0 }
        }
    }
}

EXAMPLE_JSON_NON_IFS = {
    "phases": {
        "phase5": {
            "model": "L-System",
            "parameters": {
                "dimensionality": 1.26,
                "complexity": 10
            }
        }
    }
}

def test_render_model_returns_plotly_figure():
    """
    Tests that render_model always returns a Plotly Figure object.
    """
    fig_ifs = render_model(EXAMPLE_JSON_IFS)
    assert isinstance(fig_ifs, go.Figure)

    fig_non_ifs = render_model(EXAMPLE_JSON_NON_IFS)
    assert isinstance(fig_non_ifs, go.Figure)

def test_ifs_visualization_has_scatter_trace():
    """
    Tests that a successfully rendered IFS model contains a Scattergl trace.
    """
    fig = render_model(EXAMPLE_JSON_IFS)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scattergl)
    assert len(fig.data[0].x) > 100 # Should have generated points

def test_non_ifs_visualization_has_bar_trace():
    """
    Tests that a non-IFS model renders a bar chart of its invariants.
    """
    fig = render_model(EXAMPLE_JSON_NON_IFS)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Bar)
    assert "dimensionality" in fig.data[0].x

def test_sympy_parser_correctness():
    """
    Tests that the SymPy parser correctly interprets a valid formula string.
    """
    formula = "f(x,y) = (0.5*x, 0.5*y + 0.5), f(x,y) = (0.5*x + 0.5, 0.5*y)"
    transforms = _parse_ifs_formula_with_sympy(formula)

    assert transforms is not None
    assert len(transforms) == 2

    import numpy as np
    point = np.array([1.0, 1.0])

    # Test first transform: (0.5*1.0, 0.5*1.0 + 0.5) -> (0.5, 1.0)
    transformed_point1 = transforms[0](point)
    assert np.allclose(transformed_point1, [0.5, 1.0])

    # Test second transform: (0.5*1.0 + 0.5, 0.5*1.0) -> (1.0, 0.5)
    transformed_point2 = transforms[1](point)
    assert np.allclose(transformed_point2, [1.0, 0.5])

def test_sympy_parser_handles_invalid_formula():
    """
    Tests that the SymPy parser returns None for an unparsable formula.
    """
    formula = "this is not a valid formula"
    transforms = _parse_ifs_formula_with_sympy(formula)
    assert transforms is None

def test_render_model_handles_parser_failure():
    """
    Tests that render_model shows an error message if the IFS formula is invalid.
    """
    invalid_ifs_json = {
        "phases": {"phase5": {"model": "IFS (Iterated Function System)", "formula": "invalid"}}
    }
    fig = render_model(invalid_ifs_json)
    assert len(fig.data) == 0
    assert len(fig.layout.annotations) == 1
    assert "Could not parse" in fig.layout.annotations[0].text
