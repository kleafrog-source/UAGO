import re
import numpy as np
import plotly.graph_objects as go
from sympy import sympify, symbols, lambdify
from typing import List, Dict, Any, Optional, Callable

# Define a type for the affine transformation functions
TransformFunc = Callable[[np.ndarray], np.ndarray]

def _parse_ifs_formula_with_sympy(formula: str) -> Optional[List[TransformFunc]]:
    """
    Parses an IFS formula string using a two-stage approach:
    1. Find all coordinate pairs with a simple regex.
    2. Validate and convert valid mathematical expressions using SymPy.
    """
    x, y = symbols('x y')
    transformations = []

    pattern = re.compile(r"\(([^,]+),([^)]+)\)")
    matches = pattern.findall(formula)

    if not matches:
        return None

    for expr_x_str, expr_y_str in matches:
        try:
            expr_x = sympify(expr_x_str.strip(), locals={'x': x, 'y': y})
            expr_y = sympify(expr_y_str.strip(), locals={'x': x, 'y': y})

            # Stage 2: Validate. A valid transformation is not just a single variable.
            # It must be a constant, or contain 'x' or 'y' in a more complex expression.
            is_expr_x_valid = expr_x.is_constant() or 'x' in str(expr_x) or 'y' in str(expr_x)
            is_expr_y_valid = expr_y.is_constant() or 'x' in str(expr_y) or 'y' in str(expr_y)

            # Filter out simple '(x,y)' declarations
            if (str(expr_x) == 'x' and str(expr_y) == 'y') or not (is_expr_x_valid and is_expr_y_valid) :
                continue

            func_x = lambdify((x, y), expr_x, 'numpy')
            func_y = lambdify((x, y), expr_y, 'numpy')

            def make_transform(fx, fy):
                return lambda p: np.array([fx(p[0], p[1]), fy(p[0], p[1])])

            transformations.append(make_transform(func_x, func_y))

        except Exception:
            continue

    return transformations if transformations else None

def _generate_ifs_points(transforms: List[TransformFunc], iterations: int, num_points: int) -> np.ndarray:
    """
    Generates the point cloud for an IFS fractal using the chaos game algorithm.
    """
    points = np.zeros((num_points, 2))
    current_point = np.random.rand(2)

    for i in range(num_points):
        transform = transforms[np.random.randint(0, len(transforms))]
        current_point = transform(current_point)
        points[i] = current_point

    return points[100:]

def render_model(
    json_data: Dict[str, Any],
    iterations: int = 5,
    scale: float = 1.0
) -> go.Figure:
    """
    Generates an interactive Plotly visualization for a UAGO model.
    """
    phase5 = json_data.get("phases", {}).get("phase5", {})
    model_name = phase5.get("model", "Unknown")

    fig = go.Figure()

    if model_name == "IFS (Iterated Function System)":
        formula = phase5.get("formula", "")
        transforms = _parse_ifs_formula_with_sympy(formula)

        if transforms:
            num_points = 2000 * iterations
            points = _generate_ifs_points(transforms, iterations, num_points)
            points *= scale

            fig.add_trace(go.Scattergl(
                x=points[:, 0],
                y=points[:, 1],
                mode='markers',
                marker=dict(color='blue', size=1, opacity=0.7)
            ))
            fig.update_layout(title="IFS Visualization", showlegend=False)
        else:
            fig.add_annotation(text="Could not parse IFS formula.", showarrow=False)
            fig.update_layout(title="IFS Model - Parsing Error")

    else:
        parameters = phase5.get("parameters", {})
        invariants = {k: v for k, v in parameters.items() if isinstance(v, (int, float))}

        if invariants:
            fig.add_trace(go.Bar(x=list(invariants.keys()), y=list(invariants.values())))
            fig.update_layout(title=f"Model Invariants: {model_name}")
        else:
            fig.add_annotation(text=f"No numerical invariants for model: {model_name}", showarrow=False)
            fig.update_layout(title="Unsupported Model")

    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig
