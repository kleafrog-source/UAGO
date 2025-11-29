import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import math

from uago_utils import get_project_path

def render_model(cycle_data: Dict[str, Any]) -> Optional[go.Figure]:
    phase5 = cycle_data.get("phases", {}).get("phase5", {})
    if not phase5: return None

    viz_type = phase5.get("viz_type")
    params = phase5.get("parameters", {})
    model_name = phase5.get("model", "Unknown")

    renderers = {
        "scatter_iterative": _render_ifs,
        "branching_tree": _render_lsystem,
        "parametric_curve": _render_parametric_curve,
        "point_cloud": _render_point_cloud_fallback,
    }
    renderer = renderers.get(viz_type, _render_point_cloud_fallback)

    fig = renderer(params, model_name)
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleratio=1),
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False
    )
    return fig

def _render_ifs(params: Dict[str, Any], model_name: str) -> go.Figure:
    """Generates an Iterated Function System (IFS) fractal with enhanced visuals."""
    transforms = params.get("transforms", [])
    num_points = params.get("points", 15000)

    if not transforms:
        return _render_point_cloud_fallback({}, "IFS (no transforms)")

    points = np.zeros((num_points, 2))
    custom_data = np.zeros((num_points, 2))
    current_point = np.array([0.5, 0.5])

    for i in range(num_points):
        transform_id = np.random.randint(0, len(transforms))
        t_info = transforms[transform_id]

        rot = math.radians(t_info.get("rot_deg", 0))
        sx, sy = t_info.get("scale_x", 0.5), t_info.get("scale_y", 0.5)
        tx, ty = t_info.get("trans_x", 0.0), t_info.get("trans_y", 0.0)

        new_x = current_point[0] * sx * math.cos(rot) - current_point[1] * sy * math.sin(rot) + tx
        new_y = current_point[0] * sx * math.sin(rot) + current_point[1] * sy * math.cos(rot) + ty

        current_point = np.array([new_x, new_y])
        points[i] = current_point
        custom_data[i] = [i, transform_id]

    points_to_plot = points[100:]
    custom_data_to_plot = custom_data[100:]
    iterations_for_color = custom_data_to_plot[:, 0]

    fig = go.Figure(data=go.Scattergl(
        x=points_to_plot[:, 0], y=points_to_plot[:, 1],
        mode='markers',
        customdata=custom_data_to_plot,
        hovertemplate="Iter: %{customdata[0]}<br>Transform: %{customdata[1]}<extra></extra>",
        marker=dict(
            size=3, opacity=0.7,
            color=iterations_for_color, colorscale='RdYlBu',
            colorbar=dict(title="Iteration"), showscale=True
        )
    ))
    fig.update_layout(title=f"Enhanced IFS Visualization: {model_name}")
    return fig

def _render_lsystem(params: Dict[str, Any], model_name: str) -> go.Figure:
    axiom = params.get("axiom", "F")
    rules = params.get("rules", {})
    angle = params.get("angle", 25.7)
    iterations = params.get("iterations", 3)
    current_string = axiom
    for _ in range(iterations):
        current_string = "".join(rules.get(c, c) for c in current_string)

    x, y, theta, stack, points = 0.0, 0.0, 90.0, [], [(0.0, 0.0)]
    for char in current_string:
        if char == 'F':
            x += math.cos(math.radians(theta))
            y += math.sin(math.radians(theta))
            points.append((x, y))
        elif char == '+': theta += angle
        elif char == '-': theta -= angle
        elif char == '[': stack.append((x, y, theta))
        elif char == ']' and stack:
            x, y, theta = stack.pop()
            points.extend([None, (x, y)])

    x_coords, y_coords = ([p[0] if p else None for p in points], [p[1] if p else None for p in points])
    fig = go.Figure(data=go.Scatter(x=x_coords, y=y_coords, mode='lines', line=dict(color='green')))
    fig.update_layout(title=f"{model_name} Visualization (branching_tree)")
    return fig

def _render_parametric_curve(params: Dict[str, Any], model_name: str) -> go.Figure:
    r_theta_formula = params.get("r_theta", "a * (1 - np.cos(theta))")
    theta = np.linspace(0, 2 * np.pi * params.get("revolutions", 1), 1000)
    safe_dict = {"np": np, "theta": theta, "a": params.get("a", 1.0), "b": params.get("b", 1.0)}
    try:
        r = eval(r_theta_formula, {"__builtins__": {}}, safe_dict)
        x, y = r * np.cos(theta), r * np.sin(theta)
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', line=dict(color='purple')))
        fig.update_layout(title=f"{model_name} Visualization (parametric_curve)")
    except Exception:
        fig = _render_point_cloud_fallback({}, f"{model_name} (Invalid Formula)")
    return fig

def _render_point_cloud_fallback(params: Dict[str, Any], model_name: str) -> go.Figure:
    invariants = params.get("invariants", {})
    if invariants and isinstance(invariants, dict):
        fig = go.Figure(data=go.Scatter(
            x=list(invariants.keys()), y=list(invariants.values()),
            mode='markers+text', text=[f"{v:.2f}" for v in invariants.values()]
        ))
        fig.update_layout(title="Fallback: Numerical Invariants")
    else:
        points = np.random.rand(params.get("points", 1000), 2)
        fig = go.Figure(data=go.Scattergl(
            x=points[:, 0], y=points[:, 1],
            mode='markers', marker=dict(color='grey', size=2)
        ))
        fig.update_layout(title=f"Fallback: Point Cloud for {model_name}")
    return fig

def save_visualization_html(project_name: str, fig: go.Figure) -> Optional[str]:
    project_path = get_project_path(project_name)
    if not project_path: return None
    viz_dir = project_path / "output" / "viz"
    viz_dir.mkdir(exist_ok=True, parents=True)
    filepath = viz_dir / f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    try:
        fig.write_html(str(filepath))
        return str(filepath)
    except Exception as e:
        print(f"Error saving visualization: {e}")
        return None
