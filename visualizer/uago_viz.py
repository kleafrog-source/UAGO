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
    
    # Get the input image data if available
    input_image = None
    if "input_image" in json_data:
        input_image = json_data["input_image"]
    
    fig = go.Figure()

    # Try to visualize based on available data
    if "measured_values" in phase5:
        # Visualize measured values
        measured = phase5["measured_values"]
        
        if "scale_ratios" in measured:
            # Create a bar chart for scale ratios
            scales = measured["scale_ratios"]
            fig.add_trace(go.Bar(
                x=[f"Scale {i+1}" for i in range(len(scales))],
                y=scales,
                name="Scale Ratios"
            ))
            fig.update_layout(title="Scale Ratios")
            
        elif "rotation_angles" in measured:
            # Create a polar plot for rotation angles
            angles = measured["rotation_angles"]
            fig = go.Figure(go.Barpolar(
                r=[1] * len(angles),
                theta=angles,
                name="Rotation Angles",
                marker_color='blue',
                opacity=0.7
            ))
            fig.update_layout(title="Rotation Angles Distribution")
            
        else:
            # Fallback: show all numerical measurements
            numeric_data = {k: v for k, v in measured.items() 
                          if isinstance(v, (int, float)) and not k.startswith('_')}
            
            if numeric_data:
                fig.add_trace(go.Bar(
                    x=list(numeric_data.keys()),
                    y=list(numeric_data.values()),
                    name="Measurements"
                ))
                fig.update_layout(title="Numerical Measurements")
            
    # If no measurements, try to show hypotheses
    if not fig.data and "hypotheses" in json_data.get("phases", {}).get("phase3", {}):
        hypotheses = json_data["phases"]["phase3"]["hypotheses"]
        fig.add_trace(go.Bar(
            x=[h["id"] for h in hypotheses],
            y=[h["priority"] for h in hypotheses],
            text=[h["desc"] for h in hypotheses],
            name="Hypotheses"
        ))
        fig.update_layout(title="Hypotheses by Priority")
    
    # If still no data, show a message
    if not fig.data:
        fig.add_annotation(
            text="No visualizable data found in the model.\nTry processing an image first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
    
    # Add image if available
    if input_image is not None:
        # Add a small image in the corner
        fig.add_layout_image(
            dict(
                source=input_image,
                xref="paper", yref="paper",
                x=1, y=1,
                sizex=0.2, sizey=0.2,
                xanchor="right", yanchor="top"
            )
        )

    # Update layout
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True
    )

    return fig
