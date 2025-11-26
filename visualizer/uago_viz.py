import re
import json
from jinja2 import Environment, FileSystemLoader, select_autoescape

def _parse_formula_string(formula_str: str):
    """
    A smart parser to extract affine transformations from a human-readable string.
    Looks for patterns like f(x, y) = (ax + by + e, cx + dy + f).
    Returns a list of transform dictionaries or None if parsing fails.
    """
    transforms = []
    # Regex to find coefficients for an affine transformation
    # e.g., (0.5x, 0.5y + 0.5)
    pattern = re.compile(
        r"\(\s*"
        r"(?P<a>-?\d*\.?\d*)\s*\*?\s*x\s*"  # a*x
        r"(?P<b>[+\-]?\s*\d*\.?\d*)\s*\*?\s*y\s*"  # + b*y
        r"(?P<e>[+\-]?\s*\d*\.?\d*)\s*,"  # + e
        r"\s*"
        r"(?P<c>-?\d*\.?\d*)\s*\*?\s*x\s*"  # c*x
        r"(?P<d>[+\-]?\s*\d*\.?\d*)\s*\*?\s*y\s*"  # + d*y
        r"(?P<f>[+\-]?\s*\d*\.?\d*)\s*"  # + f
        r"\)"
    )

    for match in pattern.finditer(formula_str):
        d = {k: v.replace(' ', '') if v else '0' for k, v in match.groupdict().items()}

        # Handle implicit coefficients like 'x' or '-x'
        for k in 'abcd':
            if d[k] == '' or d[k] == '+':
                d[k] = '1'
            elif d[k] == '-':
                d[k] = '-1'

        try:
            transforms.append({
                'a': float(d['a']), 'b': float(d['b']), 'c': float(d['c']),
                'd': float(d['d']), 'e': float(d['e']), 'f': float(d['f']),
            })
        except (ValueError, TypeError):
            continue # Skip if a value is not a valid float

    return transforms if transforms else None

def _get_fallback_transforms(parameters: dict):
    """
    Generate a simple set of transformations based on phase5.parameters
    as a fallback. This is a simplified interpretation.
    """
    scale = parameters.get("scales", [0.5])[0]
    # A simple four-corner fractal similar to the example
    return [
        {'a': scale, 'b': 0, 'c': 0, 'd': scale, 'e': 0, 'f': 0.5},
        {'a': scale, 'b': 0, 'c': 0, 'd': scale, 'e': 0.5, 'f': 0},
        {'a': scale, 'b': 0, 'c': 0, 'd': scale, 'e': 0, 'f': 0},
        {'a': scale, 'b': 0, 'c': 0, 'd': scale, 'e': 0.5, 'f': 0.5},
    ]

def render_model(json_data: dict) -> str:
    """
    Generates an interactive HTML visualization for a UAGO model.

    Args:
        json_data: A dictionary containing the UAGO cycle data.

    Returns:
        An HTML string with the embedded p5.js visualization.
    """
    env = Environment(
        loader=FileSystemLoader('visualizer/templates'),
        autoescape=select_autoescape(['html', 'xml'])
    )
    env.filters['tojson'] = json.dumps

    phase5 = json_data.get("phases", {}).get("phase5", {})
    model_name = phase5.get("model", "Unknown")
    parameters = phase5.get("parameters", {})

    # If the model is not IFS, return a placeholder.
    if model_name != "IFS (Iterated Function System)":
        return f"""
        <div style="padding: 20px; font-family: sans-serif; background-color: #f9f9f9; border: 1px solid #ddd;">
            <h3>Model: {model_name}</h3>
            <p>Visualization for this model is currently in development.</p>
            <p>
                <strong>Dimension:</strong> {parameters.get('dimensionality', 'N/A')}<br>
                <strong>Symmetry:</strong> {parameters.get('symmetry_group', 'N/A')}
            </p>
        </div>
        """

    formula = phase5.get("formula", "")
    error_message = ""

    transforms = _parse_formula_string(formula)

    if not transforms:
        error_message = "Could not parse formula — using fallback IFS from parameters."
        transforms = _get_fallback_transforms(parameters)

    template = env.get_template('ifs_template.html')
    return template.render(
        model=model_name,
        parameters=parameters,
        transforms=transforms,
        error_message=error_message
    )
