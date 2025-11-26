"""
UAGO Visualization Module

This module handles the integration between UAGO and p5.js for interactive
visualization of geometric structures.
"""
import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import base64

logger = logging.getLogger("UAGO.Visualizer")

class UAGOVisualizer:
    def __init__(self, project_name: str = None, base_output_dir: str = "output"):
        """
        Initialize the visualizer with project-specific output directory.
        
        Args:
            project_name: Name of the current project (optional)
            base_output_dir: Base directory for outputs (default: "output")
        """
        if project_name:
            # If project name is provided, use project-specific directory
            self.output_dir = Path("projects") / project_name / "output"
        else:
            # Fallback to default output directory
            self.output_dir = Path(base_output_dir)
            
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_path = Path(__file__).parent / "templates" / "visualization.html"
        self.static_dir = Path(__file__).parent / "static"
        self.static_dir.mkdir(exist_ok=True)

    def generate_visualization(self, uago_data: Dict) -> str:
        """Generate and save visualization HTML file with a filesystem-safe name."""
        # Generate a filesystem-safe timestamp (no colons)
        safe_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visualization_{safe_timestamp}.html"
        output_path = self.output_dir / filename
    
        try:
            html_content = self._generate_html(uago_data)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return str(output_path)
        except Exception as e:
            logging.error(f"Error saving visualization: {e}")
            raise
    
    def _prepare_visualization_data(self, uago_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare UAGO data for visualization."""
        # Extract relevant data from UAGO output
        phases = uago_data.get('phases', {})
        
        # Get model information if available
        model_info = {}
        if 'phase5' in phases:
            model_info = {
                'type': phases['phase5'].get('model', ''),
                'formula': phases['phase5'].get('formula', ''),
                'parameters': phases['phase5'].get('parameters', {})
            }
        
        # Get hypotheses if available
        hypotheses = []
        if 'phase3' in phases and 'hypotheses' in phases['phase3']:
            hypotheses = [
                {
                    'id': h.get('id', ''),
                    'description': h.get('desc', ''),
                    'priority': h.get('priority', 0)
                }
                for h in phases['phase3']['hypotheses']
            ]
        
        return {
            'model': model_info,
            'hypotheses': hypotheses,
            'invariants': self._extract_invariants(phases),
            'metadata': {
                'timestamp': uago_data.get('timestamp', ''),
                'input_shape': uago_data.get('input_shape', [])
            }
        }
    
    def _extract_invariants(self, phases: Dict[str, Any]) -> Dict[str, Any]:
        """Extract invariant data from UAGO phases."""
        invariants = {}
        
        # Extract from phase 2 (coarse invariants)
        if 'phase2' in phases:
            phase2 = phases['phase2']
            invariants.update({
                'dimensionality': phase2.get('dimensionality'),
                'scales': phase2.get('scales', []),
                'symmetry': phase2.get('symmetry'),
                'connectivity': phase2.get('connectivity')
            })
        
        # Extract from phase 4 (adaptive measurements)
        if 'phase4' in phases and 'measured_values' in phases['phase4']:
            invariants.update(phases['phase4']['measured_values'])
        
        return invariants
    
    def _generate_html(self, vis_data: Dict[str, Any]) -> str:
        """Generate the HTML content for the visualization."""
        # Read the template
        template = self.template_path.read_text(encoding='utf-8')
        
        # Prepare the data for the template
        data_js = f"const uagoData = {json.dumps(vis_data, indent=2)};"
        
        # Replace placeholders in the template
        html = template.replace('/* {{UAGO_DATA }} */', data_js)
        
        return html
