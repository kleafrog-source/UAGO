"""
AI Interpreter for UAGO Visualizations

This module provides AI-powered interpretation of UAGO outputs to generate
p5.js visualization code.
"""
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
import re
from pathlib import Path

logger = logging.getLogger("UAGO.AIInterpreter")

class AIVisualizationInterpreter:
    """Interprets UAGO outputs and generates visualization code using AI."""
    
    def __init__(self, mistral_client=None, max_retries: int = 3):
        """Initialize the AI interpreter.
        
        Args:
            mistral_client: Optional Mistral API client
            max_retries: Maximum number of retries for API calls
        """
        self.mistral_client = mistral_client
        self.max_retries = max_retries
        self.template_dir = Path(__file__).parent / "templates"
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load prompt templates from files."""
        self.prompts = {
            'visualization': self._load_prompt('visualization_prompt.txt'),
            'refinement': self._load_prompt('refinement_prompt.txt'),
            'error_handling': self._load_prompt('error_handling_prompt.txt')
        }
    
    def _load_prompt(self, filename: str) -> str:
        """Load a prompt template from a file."""
        try:
            path = self.template_dir / filename
            return path.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"Failed to load prompt {filename}: {e}")
            return ""
    
    def generate_visualization_code(
        self,
        uago_data: Dict[str, Any],
        max_iterations: int = 5
    ) -> Tuple[str, List[Dict]]:
        """Generate p5.js visualization code from UAGO output.
        
        Args:
            uago_data: Dictionary containing UAGO analysis results
            max_iterations: Maximum number of refinement iterations
            
        Returns:
            Tuple of (generated_code, history) where history is a list of 
            interaction steps
        """
        if not self.mistral_client:
            return self._generate_fallback_code(uago_data), []
        
        # Prepare the context for the AI
        context = self._prepare_context(uago_data)
        history = []
        
        # Initial generation
        prompt = self.prompts['visualization'].format(
            model_info=json.dumps(context['model'], indent=2),
            hypotheses=json.dumps(context['hypotheses'], indent=2),
            invariants=json.dumps(context['invariants'], indent=2)
        )
        
        current_code = ""
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                # Generate or refine code
                if iteration == 1:
                    response = self._call_mistral(prompt)
                else:
                    # Use refinement prompt for subsequent iterations
                    refinement_prompt = self.prompts['refinement'].format(
                        current_code=current_code,
                        error=context.get('last_error', '')
                    )
                    response = self._call_mistral(refinement_prompt)
                
                # Extract code from response
                new_code = self._extract_code(response)
                if not new_code:
                    raise ValueError("No valid code found in AI response")
                
                # Validate the code
                is_valid, error = self._validate_code(new_code)
                
                # Log this iteration
                history.append({
                    'iteration': iteration,
                    'prompt': prompt,
                    'response': response,
                    'code': new_code,
                    'is_valid': is_valid,
                    'error': error if not is_valid else None
                })
                
                if is_valid:
                    current_code = new_code
                    break
                else:
                    context['last_error'] = error
                    prompt = self.prompts['error_handling'].format(
                        error=error,
                        current_code=current_code
                    )
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {str(e)}")
                if iteration >= max_iterations:
                    logger.warning("Max iterations reached, using fallback code")
                    return self._generate_fallback_code(uago_data), history
        
        return current_code, history
    
    def _prepare_context(self, uago_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the context for the AI prompt."""
        phases = uago_data.get('phases', {})
        
        # Extract model information
        model_info = {}
        if 'phase5' in phases:
            model_info = {
                'type': phases['phase5'].get('model', ''),
                'formula': phases['phase5'].get('formula', ''),
                'parameters': phases['phase5'].get('parameters', {})
            }
        
        # Extract hypotheses
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
        
        # Extract invariants
        invariants = {}
        if 'phase2' in phases:
            phase2 = phases['phase2']
            invariants.update({
                'dimensionality': phase2.get('dimensionality'),
                'scales': phase2.get('scales', []),
                'symmetry': phase2.get('symmetry'),
                'connectivity': phase2.get('connectivity')
            })
        
        if 'phase4' in phases and 'measured_values' in phases['phase4']:
            invariants.update(phases['phase4']['measured_values'])
        
        return {
            'model': model_info,
            'hypotheses': hypotheses,
            'invariants': invariants,
            'metadata': {
                'timestamp': uago_data.get('timestamp', ''),
                'input_shape': uago_data.get('input_shape', [])
            }
        }
    
    def _call_mistral(self, prompt: str) -> str:
        """Call the Mistral API with error handling and retries."""
        if not self.mistral_client:
            raise RuntimeError("Mistral client not initialized")
        
        for attempt in range(self.max_retries):
            try:
                response = self.mistral_client.chat.complete(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
    
    def _extract_code(self, text: str) -> str:
        """Extract code block from text response."""
        # Try to find code blocks first
        code_blocks = re.findall(r'```(?:javascript|js)?\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, look for function definitions
        functions = re.findall(r'function\s+\w+\s*\([^)]*\)\s*\{[^}]*\}', text, re.DOTALL)
        if functions:
            return '\n\n'.join(functions)
        
        # As a last resort, return the whole text
        return text.strip()
    
    def _validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate the generated JavaScript code."""
        # Basic validation - check for common issues
        if not code.strip():
            return False, "Empty code"
        
        # Check for required p5.js functions
        required_functions = ['setup', 'draw']
        for func in required_functions:
            if f'function {func}(' not in code and f'function{func}(' not in code:
                return False, f"Missing required function: {func}"
        
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            'eval\s*\(', 'Function\s*\(', 'setTimeout\s*\(', 'setInterval\s*\('
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return False, f"Potentially dangerous pattern detected: {pattern}"
        
        return True, None
    
    def _generate_fallback_code(self, uago_data: Dict[str, Any]) -> str:
        """Generate a simple fallback visualization when AI is not available."""
        return """
        function setup() {
            createCanvas(800, 600);
            noLoop();
            background(240);
            textAlign(CENTER, CENTER);
            textSize(20);
            fill(0);
            text("AI Visualization Unavailable", width/2, height/2);
            textSize(14);
            text("Please check your API key and try again", width/2, height/2 + 30);
        }
        
        function draw() {
            // Fallback visualization
            noFill();
            stroke(100, 100, 200);
            strokeWeight(2);
            
            // Simple grid
            for (let x = 0; x < width; x += 40) {
                line(x, 0, x, height);
            }
            for (let y = 0; y < height; y += 40) {
                line(0, y, width, y);
            }
        }
        """

# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example UAGO data
    example_data = {
        "timestamp": "2023-11-26T12:34:56.789012",
        "input_shape": [800, 600, 3],
        "phases": {
            "phase2": {
                "dimensionality": 1.8,
                "scales": [0.5, 0.25, 0.125],
                "symmetry": "radial",
                "connectivity": "high"
            },
            "phase3": {
                "hypotheses": [
                    {
                        "id": "H1",
                        "desc": "Fractal structure with self-similar branches",
                        "priority": 0.9
                    }
                ]
            },
            "phase5": {
                "model": "Fractal Tree",
                "formula": "Recursive branching with angle variation",
                "parameters": {
                    "branch_angle": 0.5,
                    "scale_factor": 0.67,
                    "levels": 5
                }
            }
        }
    }
    
    # Initialize interpreter (without Mistral client for this example)
    interpreter = AIVisualizationInterpreter()
    
    # Generate visualization code
    code, history = interpreter.generate_visualization_code(example_data)
    print("Generated code:")
    print(code)
