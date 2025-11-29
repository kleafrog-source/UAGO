import re
import json
import numpy as np
import plotly.graph_objects as go
from sympy import sympify, symbols, lambdify
from typing import List, Dict, Any, Optional, Callable, Tuple

import logging
from mistralai import Mistral

# Define a type for the affine transformation functions
TransformFunc = Callable[[np.ndarray], np.ndarray]

def _parse_ifs_formula_with_sympy(formula: str) -> Optional[List[TransformFunc]]:
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

            is_expr_x_valid = expr_x.is_constant() or 'x' in str(expr_x) or 'y' in str(expr_x)
            is_expr_y_valid = expr_y.is_constant() or 'x' in str(expr_y) or 'y' in str(expr_y)

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
    points = np.zeros((num_points, 2))
    current_point = np.random.rand(2)

    for i in range(num_points):
        transform = transforms[np.random.randint(0, len(transforms))]
        current_point = transform(current_point)
        points[i] = current_point

    return points[100:]

def _create_fallback_fig(parameters: Dict[str, Any], model_name: str) -> go.Figure:
    fig = go.Figure()
    invariants = {k: v for k, v in parameters.items() if isinstance(v, (int, float))}

    if invariants:
        fig.add_trace(go.Scatter(
            x=list(invariants.keys()),
            y=list(invariants.values()),
            mode='lines+markers'
        ))
        fig.update_layout(title=f"Numerical Invariants for {model_name}")
    else:
        fig.add_annotation(text=f"No numerical invariants to display for model: {model_name}", showarrow=False)
        fig.update_layout(title="Unsupported or Unparsable Model")
    return fig

def render_model(
    json_data: Dict[str, Any],
    iterations: int = 5,
    scale: float = 1.0
) -> go.Figure:
    phases = json_data.get("phases", {})
    if "phase5" not in phases:
        fig = go.Figure()
        fig.add_annotation(text="Phase 5 data is missing.", showarrow=False)
        return fig

    phase5 = phases["phase5"]
    model_name = phase5.get("model", "Unknown")
    parameters = phase5.get("parameters", {})

    fig = go.Figure()

    if model_name == "IFS (Iterated Function System)":
        formula = phase5.get("formula", "")
        transforms = _parse_ifs_formula_with_sympy(formula)

        if transforms:
            num_points = 2000 * iterations
            points = _generate_ifs_points(transforms, iterations, num_points)
            points *= scale

            fig.add_trace(go.Scattergl(x=points[:, 0], y=points[:, 1], mode='markers', marker=dict(color='blue', size=1, opacity=0.7)))
            fig.update_layout(title="IFS Visualization", showlegend=False)
        else:
            fig = _create_fallback_fig(parameters, model_name)
    else:
        fig = _create_fallback_fig(parameters, model_name)

    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def mistral_transform(json_data: Dict[str, Any], api_key: str) -> Tuple[Optional[go.Figure], Dict[str, Any]]:
    log_entry = {"phase": "mistral_visualization"}

    if not api_key:
        log_entry["error"] = "Mistral API key not provided."
        return None, log_entry

    phases = json_data.get("phases", {})
    phase5 = phases.get("phase5", {})
    if not phase5:
        log_entry["error"] = "Phase 5 data is missing."
        return None, log_entry

    model = phase5.get("model", "N/A")
    formula = phase5.get("formula", "N/A")
    parameters = json.dumps(phase5.get("parameters", {}))

    # Whitelist of safe built-in functions
    safe_builtins = {
        "range": range, "len": len, "list": list, "dict": dict,
        "float": float, "int": int, "str": str, "abs": abs,
        "min": min, "max": max, "sum": sum, "zip": zip,
    }

    prompt = (
        f"Generate Plotly Python code for the following mathematical model:\n"
        f"Model: {model}\n"
        f"Formula: {formula}\n"
        f"Parameters: {parameters}\n"
        f"Create a Plotly figure object named 'fig'. Do not include `import` statements. "
        f"You have access to `go` (plotly.graph_objects), `np` (numpy), `json`, and the following safe built-ins: "
        f"{', '.join(safe_builtins.keys())}. "
        f"Prefer list comprehensions over complex loops where possible. "
        f"Output ONLY a single Python code block enclosed in ```python...```."
    )
    log_entry["prompt"] = prompt

    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        content = response.choices[0].message.content
        log_entry["response"] = content

        code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
        if not code_match:
            log_entry["error"] = "No Python code block found in Mistral's response."
            return _create_fallback_fig(phase5.get("parameters", {}), model), log_entry

        code = code_match.group(1).strip()

        # Execute the code in a restricted environment
        try:
            restricted_globals = {
                "__builtins__": safe_builtins,
                "go": go, "np": np, "json": json,
            }
            local_vars = {}
            exec(code, restricted_globals, local_vars)

            fig = local_vars.get("fig")
            if isinstance(fig, go.Figure):
                return fig, log_entry
            else:
                log_entry["error"] = "Executed code did not produce a valid Plotly Figure."
                return _create_fallback_fig(phase5.get("parameters", {}), model), log_entry
        except Exception as e:
            logging.error(f"Exec of Mistral code failed: {e}")
            log_entry["error"] = f"Execution failed: {e}. Using fallback."
            return _create_fallback_fig(phase5.get("parameters", {}), model), log_entry

    except Exception as e:
        logging.error(f"Mistral API call failed: {e}")
        log_entry["error"] = f"Mistral API call failed: {e}. Using fallback."
        return _create_fallback_fig(phase5.get("parameters", {}), model), log_entry
