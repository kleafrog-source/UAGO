import pytest
import sys
import os
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualizer.uago_viz import render_model, save_visualization_html
from uago_utils import get_project_path as original_get_project_path

# --- Test Data ---

CYCLE_DATA_IFS = {
    "phases": {"phase5": {
        "model": "IFS",
        "viz_type": "scatter_iterative",
        "parameters": {
            "transforms": [{"scale_x": 0.5, "scale_y": 0.5, "rot_deg": 0.0, "trans_x": 0.0, "trans_y": 0.5}],
            "iterations": 5, "points": 1000
        }
    }}
}

CYCLE_DATA_LSYSTEM = {
    "phases": {"phase5": {
        "model": "L-System",
        "viz_type": "branching_tree",
        "parameters": { "axiom": "F", "rules": {"F": "F[+F]F[-F]F"}, "angle": 20, "iterations": 3 }
    }}
}

CYCLE_DATA_FALLBACK_UNKNOWN = {
    "phases": {"phase5": {
        "model": "Unknown",
        "viz_type": "unknown_viz",
        "parameters": {}
    }}
}

CYCLE_DATA_FALLBACK_INVARIANTS = {
    "phases": {"phase5": {
        "model": "InvariantsOnly",
        "viz_type": "point_cloud", # viz_type is point_cloud, but params suggest invariants plot
        "parameters": {"invariants": {"dim": 1.8, "complexity": 0.4}}
    }}
}

# --- Tests for the New Deterministic Renderers ---

def test_render_model_dispatcher_selects_ifs():
    fig = render_model(CYCLE_DATA_IFS)
    assert isinstance(fig.data[0], go.Scattergl)
    assert "IFS Visualization" in fig.layout.title.text

def test_render_model_dispatcher_selects_lsystem():
    fig = render_model(CYCLE_DATA_LSYSTEM)
    assert isinstance(fig.data[0], go.Scatter)
    assert "L-System Visualization" in fig.layout.title.text

def test_render_model_fallback_for_unknown_viz_type():
    """Verify fallback to a random point cloud for an unrecognized viz_type."""
    fig = render_model(CYCLE_DATA_FALLBACK_UNKNOWN)
    assert isinstance(fig.data[0], go.Scattergl) # Random point cloud is Scattergl
    assert "Fallback: Point Cloud" in fig.layout.title.text

def test_render_model_fallback_with_invariants():
    """Verify fallback plots numerical invariants if they are present."""
    fig = render_model(CYCLE_DATA_FALLBACK_INVARIANTS)
    assert isinstance(fig.data[0], go.Scatter)
    assert fig.data[0].mode == 'markers+text' # Invariants plot uses markers+text
    assert "Fallback: Numerical Invariants" in fig.layout.title.text

def test_render_ifs_with_no_transforms_falls_back():
    """An IFS model with no transforms should fall back to a random point cloud."""
    bad_ifs_data = {"phases": {"phase5": {
        "model": "IFS", "viz_type": "scatter_iterative", "parameters": {}
    }}}
    fig = render_model(bad_ifs_data)
    assert isinstance(fig.data[0], go.Scattergl)
    assert "Fallback: Point Cloud" in fig.layout.title.text

def test_save_visualization_html(monkeypatch, tmp_path):
    """Test saving the visualization HTML to the correct project directory."""
    project_name = "test_save_project"

    def mock_get_project_path(name):
        return tmp_path / "projects" / name

    monkeypatch.setattr("visualizer.uago_viz.get_project_path", mock_get_project_path)

    fig = go.Figure(data=go.Scatter(x=[1], y=[1]))
    save_path_str = save_visualization_html(project_name, fig)

    assert save_path_str is not None
    save_path = Path(save_path_str)
    assert save_path.exists()
    assert f"projects/{project_name}/output/viz" in str(save_path).replace("\\", "/")
