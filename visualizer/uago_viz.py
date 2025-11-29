import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import math
import plotly.colors as pcolors

from uago_utils import get_project_path

def render_model(cycle_data: Dict[str, Any]) -> Optional[go.Figure]:
    phase2 = cycle_data.get("phases", {}).get("phase2", {})
    phase5 = cycle_data.get("phases", {}).get("phase5", {})
    if not phase5: return None

    viz_type = phase5.get("viz_type")
    params = phase5.get("parameters", {})
    model_name = phase5.get("model", "Unknown")

    renderers = {
        "scatter_iterative": _render_ifs,
        "branching_tree": _render_lsystem,
        "grid_tiling": _render_tiling,
        "parametric_curve": _render_parametric_curve,
    }

    # Route special models to the 3D renderer
    if model_name in ["LieGroup", "AlgebraicVariety"]:
        renderer = _render_3d_surface
    else:
        renderer = renderers.get(viz_type, _render_point_cloud_fallback)

    fig = renderer(params, model_name, phase2) # Pass invariants
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    return fig

def _render_ifs(params: Dict[str, Any], model_name: str, invariants: Dict) -> go.Figure:
    """Generates an Iterated Function System (IFS) fractal with a single connected trace."""
    transforms = params.get("transforms", [])
    if not transforms: return _render_point_cloud_fallback({}, "IFS (no transforms)", invariants)

    num_points = 20000
    points = np.zeros((num_points, 2))
    custom_data = []
    colors = []

    current_point = np.array([0.5, 0.5])
    color_palette = pcolors.qualitative.Set1

    for i in range(num_points):
        transform_id = np.random.randint(0, len(transforms))
        t = transforms[transform_id]
        rot, sx, sy, tx, ty = math.radians(t.get("rot_deg",0)), t.get("scale_x",0.5), t.get("scale_y",0.5), t.get("trans_x",0), t.get("trans_y",0)

        current_point = np.array([
            current_point[0]*sx*math.cos(rot) - current_point[1]*sy*math.sin(rot) + tx,
            current_point[0]*sx*math.sin(rot) + current_point[1]*sy*math.cos(rot) + ty
        ])
        points[i] = current_point

        formula = f"f_{transform_id}(x,y) = {sx:.3f}x + {tx:.3f}, {sy:.3f}y + {ty:.3f} | θ={t.get('rot_deg',0):.1f}°"
        custom_data.append([formula, i])
        colors.append(color_palette[transform_id % len(color_palette)])

    fig = go.Figure(data=go.Scattergl(
        x=points[:, 0], y=points[:, 1],
        mode='lines+markers',
        line=dict(width=1.5, color='rgba(0,0,0,0.1)'), # Faint connecting lines
        marker=dict(size=2, opacity=0.8, color=colors),
        customdata=custom_data,
        hovertemplate="<b>Formula</b>: %{customdata[0]}<br><b>Point Index</b>: %{customdata[1]}<extra></extra>"
    ))

    dim = invariants.get('dimensionality', 2.0)
    fig.update_layout(title_text=f"IFS Attractor — {len(transforms)} Transforms (dim≈{dim:.2f})")
    return fig

def _render_lsystem(params: Dict[str, Any], model_name: str, invariants: Dict) -> go.Figure:
    """Generates a branching structure from an L-system with color-by-depth."""
    axiom = params.get("axiom", "F")
    rules = params.get("rules", {"F": "F+F--F+F"})
    angle = params.get("angle", 60)
    iterations = params.get("iterations", 4)

    current_string = axiom
    for _ in range(iterations):
        current_string = "".join(rules.get(c, c) for c in current_string)

    x, y, theta, depth = 0.0, 0.0, 90.0, 0
    stack = []
    segments = []

    for char in current_string:
        if char == 'F':
            x_new = x + math.cos(math.radians(theta))
            y_new = y + math.sin(math.radians(theta))
            segments.append({'x0': x, 'y0': y, 'x1': x_new, 'y1': y_new, 'depth': depth})
            x, y = x_new, y_new
        elif char == '+':
            theta += angle
        elif char == '-':
            theta -= angle
        elif char == '[':
            stack.append((x, y, theta, depth))
            depth += 1
        elif char == ']':
            if stack:
                x, y, theta, depth = stack.pop()

    fig = go.Figure()
    max_depth = max(s['depth'] for s in segments) if segments else 1
    colorscale = pcolors.get_colorscale('Viridis')

    for seg in segments:
        norm_depth = seg['depth'] / max_depth if max_depth > 0 else 0
        color = pcolors.sample_colorscale(colorscale, norm_depth)[0]
        fig.add_trace(go.Scatter(
            x=[seg['x0'], seg['x1']], y=[seg['y0'], seg['y1']], mode='lines',
            line=dict(width=1.5, color=color),
            customdata=[[seg['depth'], angle]],
            hovertemplate=f"Rule: F<br>Angle: {angle}°<br>Depth: {seg['depth']}<extra></extra>"
        ))

    fig.update_layout(title_text=f"L-System Branching — Axiom: {axiom}, Rules: {str(rules)[:30]}...")
    return fig

def _render_tiling(params: Dict[str, Any], model_name: str, invariants: Dict) -> go.Figure:
    """Generates a grid tiling based on symmetry parameters."""
    fig = go.Figure()
    symmetry = params.get('symmetry', 'p1')
    period = params.get('period', 50)

    # Define a simple fundamental domain (a square)
    domain = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) * period

    # Define symmetry operations (simplified for demo)
    ops = {
        'p1': [np.array([1, 0, 0, 1])], # Identity
        'p4m': [ # Rotations and reflections for a square
            np.array([1, 0, 0, 1]), np.array([0, -1, 1, 0]),
            np.array([-1, 0, 0, -1]), np.array([0, 1, -1, 0]),
            np.array([-1, 0, 0, 1]), np.array([1, 0, 0, -1]),
        ]
    }

    op_matrices = ops.get(symmetry, ops['p1'])
    colors = pcolors.qualitative.Plotly

    for i in range(-2, 3):
        for j in range(-2, 3):
            for op_index, op in enumerate(op_matrices):
                rotation = op.reshape(2, 2)
                transformed_domain = (domain @ rotation) + np.array([i * period * 2, j * period * 2])

                fig.add_trace(go.Scatter(
                    x=transformed_domain[:, 0], y=transformed_domain[:, 1],
                    mode='lines',
                    line=dict(width=2, color=colors[op_index % len(colors)]),
                    customdata=[[symmetry, period]],
                    hovertemplate="Symmetry: %{customdata[0]}<br>Period: %{customdata[1]}px<extra></extra>"
                ))

    fig.update_layout(title_text=f"Aperiodic Tiling — {symmetry} (coverage=1.0)")
    return fig

def _render_parametric_curve(params: Dict[str, Any], model_name: str, invariants: Dict) -> go.Figure:
    formula = params.get("r_theta", "a * (1 - np.cos(theta))")
    a = params.get("a", 1.0)
    b = params.get("b", 1.0)
    theta = np.linspace(0, 2 * np.pi, 1000)
    try:
        r = eval(formula, {"__builtins__":{}}, {"np": np, "theta": theta, "a": a, "b": b})
        x, y = r * np.cos(theta), r * np.sin(theta)
        customdata = np.stack((theta, r), axis=-1)
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', line_width=3, fill='toself',
                                       customdata=customdata, hovertemplate="θ = %{customdata[0]:.2f}, r = %{customdata[1]:.3f}"))
    except Exception:
        return _render_point_cloud_fallback({}, "Invalid Curve", invariants)
    fig.update_layout(title_text=f"Polar Curve — r(θ) = {formula} (a={a:.3f}, b={b:.3f})")
    return fig

def _render_3d_surface(params: Dict[str, Any], model_name: str, invariants: Dict) -> go.Figure:
    formula = params.get("formula", "np.sin(x) * np.cos(y)")
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    x, y = np.meshgrid(x, y)
    try:
        z = eval(formula, {"__builtins__":{}}, {"np": np, "x": x, "y": y})
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis',
                                        hovertemplate="z = f(x,y)<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>")])
    except Exception:
        return _render_point_cloud_fallback({}, "Invalid 3D Formula", invariants)
    fig.update_layout(title_text=f"3D Variety — {model_name}")
    return fig

def _render_point_cloud_fallback(params: Dict[str, Any], model_name: str, invariants: Dict) -> go.Figure:
    fig = go.Figure(data=go.Scatter(
        x=[k for k,v in invariants.items() if isinstance(v, (int, float))],
        y=[v for k,v in invariants.items() if isinstance(v, (int, float))],
        mode='markers+text', text=[f"{v:.2f}" for v in invariants.values() if isinstance(v, (int, float))]
    ))
    fig.update_layout(title=f"Discovered Structure — Model: {model_name} (in development)")
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
        print(f"Error saving viz: {e}")
        return None
