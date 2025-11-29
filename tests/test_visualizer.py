import pytest
import sys
import os
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualizer.uago_viz import render_model, save_visualization_html

# --- Test Data ---

INVARIANTS = {"dimensionality": 1.67, "complexity": 0.8}

CYCLE_DATA_IFS = {
    "phases": {"phase2": INVARIANTS, "phase5": {
        "model": "IFS", "viz_type": "scatter_iterative",
        "parameters": {"transforms": [
            {"scale_x": 0.5, "scale_y": 0.5, "rot_deg": 0.0, "trans_x": 0.0, "trans_y": 0.5},
            {"scale_x": 0.5, "scale_y": 0.5, "rot_deg": 90.0, "trans_x": 0.5, "trans_y": 0.5}
        ]}
    }}
}

CYCLE_DATA_LSYSTEM = {
    "phases": {"phase2": INVARIANTS, "phase5": {
        "model": "L-System", "viz_type": "branching_tree",
        "parameters": {"axiom": "F", "rules": {"F": "F[+F]F"}, "angle": 30}
    }}
}

CYCLE_DATA_3D = {
    "phases": {"phase2": INVARIANTS, "phase5": {
        "model": "AlgebraicVariety",
        "parameters": {"formula": "np.sin(x*y)"}
    }}
}

CYCLE_DATA_FALLBACK = {
    "phases": {"phase2": INVARIANTS, "phase5": {
        "model": "UnknownModel", "viz_type": "unsupported_type", "parameters": {}
    }}
}

# --- Tests ---

def test_render_model_ifs():
    fig = render_model(CYCLE_DATA_IFS)
    assert isinstance(fig.data[0], go.Scattergl)
    assert "IFS Attractor" in fig.layout.title.text
    assert "1.67" in fig.layout.title.text # Invariant in title
    assert "Formula" in fig.data[0].hovertemplate

def test_render_model_lsystem():
    fig = render_model(CYCLE_DATA_LSYSTEM)
    assert isinstance(fig.data[0], go.Scatter)
    assert "L-System Branching" in fig.layout.title.text
    assert "Rule" in fig.data[0].hovertemplate

def test_render_model_3d_surface():
    fig = render_model(CYCLE_DATA_3D)
    assert isinstance(fig.data[0], go.Surface)
    assert "3D Variety" in fig.layout.title.text
    assert "z = f(x,y)" in fig.data[0].hovertemplate

def test_render_model_fallback():
    fig = render_model(CYCLE_DATA_FALLBACK)
    assert isinstance(fig.data[0], go.Scatter)
    assert "Discovered Structure" in fig.layout.title.text
    assert "UnknownModel" in fig.layout.title.text

def test_save_visualization_html(monkeypatch, tmp_path):
    project_name = "test_save"
    monkeypatch.setattr("visualizer.uago_viz.get_project_path", lambda name: tmp_path / "projects" / name)

    fig = go.Figure(data=go.Scatter(x=[1], y=[1]))
    save_path = Path(save_visualization_html(project_name, fig))

    assert save_path.exists()
    assert f"projects/{project_name}/output/viz" in str(save_path).replace("\\", "/")
