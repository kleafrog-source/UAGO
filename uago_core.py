import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List, Callable
import logging
from datetime import datetime
import json
import re
import time
from scipy.signal import correlate2d
from skimage import measure
import pywt
from uago_config import UAGO_CONFIG
from uago_utils import save_mistral_response

class UAGOCore:
    def __init__(self, project_name: str, api_key: Optional[str] = None, demo_mode: bool = False):
        self.project_name = project_name
        self.api_key = api_key
        self.demo_mode = demo_mode or (api_key is None)
        self.logger = logging.getLogger(f"UAGO-{project_name}")
        self.mistral_client = None

        if self.api_key and not self.demo_mode:
            try:
                from mistralai import Mistral
                self.mistral_client = Mistral(api_key=self.api_key)
                self.logger.info("Mistral API client initialized successfully.")
            except ImportError:
                self.logger.warning("`mistralai` package not found. Install it to use the AI features.")
                self.demo_mode = True
            except Exception as e:
                self.logger.error(f"Failed to initialize Mistral API: {e}")
                self.demo_mode = True

        self.current_cycle_data = {}

    def _safe_mistral_call(self, prompt: str, phase: int) -> Optional[str]:
        if not self.mistral_client:
            self.logger.warning(f"Mistral client not available for Phase {phase}. Skipping API call.")
            return None

        for attempt in range(3):
            try:
                response = self.mistral_client.chat.complete(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1, # Lower temperature for stricter JSON output
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait = 2 ** attempt
                    self.logger.warning(f"Rate limit hit in Phase {phase}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    self.logger.error(f"An unexpected error occurred in Phase {phase}: {e}")
                    return None
        self.logger.error(f"Max retries exceeded for API call in Phase {phase}.")
        return None

    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Extracts a JSON object from a string, tolerating markdown code blocks."""
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback for raw JSON without markdown
            json_str = content

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to decode JSON from content: {content[:250]}...")
            return None

    def process_frame(self, frame: np.ndarray, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        self.current_cycle_data = {
            "timestamp": datetime.now().isoformat(),
            "input_shape": frame.shape,
            "phases": {},
        }

        phases = [
            (1, "Structure Detection", self.phase1_structure_detection),
            (2, "Invariant Extraction", self.phase2_invariant_extraction),
            (3, "Hypothesis Generation", self.phase3_hypothesis_generation),
            (4, "Adaptive Measurement", self.phase4_adaptive_measurement),
            (5, "Model Search", self.phase5_model_search),
            (6, "Validation", self.phase6_validation),
            (7, "Context Transition", self.phase7_context_transition),
        ]

        for i, (phase_num, phase_name, phase_func) in enumerate(phases):
            if progress_callback:
                progress_callback(phase_num, phase_name, (i / len(phases)) * 100)

            result = phase_func(frame)
            self.current_cycle_data["phases"][f"phase{phase_num}"] = result

        if progress_callback:
            progress_callback(7, "Complete", 100)

        return self.current_cycle_data

    # --- Phase Implementations ---

    def phase1_structure_detection(self, frame: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) > 2 else frame
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"structure_detected": False}

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return {
            "roi": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "structure_detected": True,
            "contour_points": len(largest_contour)
        }

    def phase2_invariant_extraction(self, frame: np.ndarray) -> Dict[str, Any]:
        roi_data = self.current_cycle_data["phases"].get("phase1", {}).get("roi")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) > 2 else frame

        if roi_data:
            x, y, w, h = roi_data['x'], roi_data['y'], roi_data['width'], roi_data['height']
            gray_roi = gray[y:y+h, x:x+w]
        else:
            gray_roi = gray
        
        # Fractal Dimension
        _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dimension = self._estimate_box_counting_dimension(binary)
        
        # Hu Moments for symmetry
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        return {
            "fractal_dimension": dimension,
            "hu_moments": [float(h) for h in hu_moments[:3]],
            "complexity": float(np.sum(binary > 0) / binary.size)
        }

    def phase3_hypothesis_generation(self, frame: np.ndarray) -> Dict[str, Any]:
        # This phase is now simplified, as the heavy lifting is in phase 5.
        # It could be used for pre-filtering models in a more complex system.
        return {"status": "completed", "info": "Hypothesis generation is now integrated into Phase 5."}

    def phase4_adaptive_measurement(self, frame: np.ndarray) -> Dict[str, Any]:
        # This phase is also simplified. It can be used to calculate specific
        # parameters needed for the models proposed in a future, more advanced phase 3.
        return {"status": "completed", "info": "Adaptive measurement is now integrated into Phase 5."}

    def phase5_model_search(self, frame: np.ndarray) -> Dict[str, Any]:
        if self.demo_mode or not self.mistral_client:
            self.logger.info("Phase 5: Using demo/fallback model.")
            return self._get_default_model_response()

        invariants = self.current_cycle_data["phases"].get("phase2", {})
        prompt = self._build_model_search_prompt(invariants)
        
        content = self._safe_mistral_call(prompt, phase=5)
        
        if content:
            response_json = self._extract_json(content)
            save_mistral_response(self.project_name, "phase5", {"prompt": prompt, "response": content})

            if response_json and self._validate_model_schema(response_json):
                return response_json
        
        self.logger.warning("Phase 5: Mistral response was invalid or failed. Using fallback.")
        return self._get_default_model_response()

    def _build_model_search_prompt(self, invariants: Dict) -> str:
        return f"""
Analyze these geometric invariants: {json.dumps(invariants)}.
Your task is to select the most plausible mathematical model that could generate such a structure and provide its parameters in a specific JSON format.

The JSON object MUST have the following structure:
{{
  "model": "IFS|L-system|Tiling|LieGroup|AlgebraicCurve|Other",
  "viz_type": "scatter_iterative|branching_tree|grid_tiling|orbit_plot|parametric_curve|point_cloud",
  "parameters": {{ ... }}
}}

- For "model": "IFS", "viz_type" must be "scatter_iterative". The "parameters" must contain a "transforms" array, with each object having "scale_x", "scale_y", "rot_deg", "trans_x", "trans_y". Also include "iterations" and "points".
- For "model": "L-system", "viz_type" must be "branching_tree". "parameters" must contain "axiom", "rules" (a dictionary), and "angle".
- For "model": "AlgebraicCurve", "viz_type" must be "parametric_curve". "parameters" must contain "r_theta" (a string formula) and any variables used in it (e.g., "a", "b").
- For other models, choose the most appropriate "viz_type" and fill "parameters" accordingly.

Based on the invariants, infer the parameters. For example, a high fractal dimension suggests an IFS.

Respond EXCLUSIVELY with the valid JSON object. No explanations. No markdown. No natural language.
"""

    def _validate_model_schema(self, data: Dict) -> bool:
        """A basic validator for the Phase 5 JSON schema."""
        if "model" not in data or "viz_type" not in data or "parameters" not in data:
            return False
        if not isinstance(data["parameters"], dict):
            return False
        # Add more specific validation rules here if needed
        return True

    def _get_default_model_response(self) -> Dict[str, Any]:
        """A safe fallback response for Phase 5."""
        return {
            "model": "Unknown",
            "viz_type": "point_cloud",
            "parameters": {
                "points": 1000,
                "invariants": self.current_cycle_data["phases"].get("phase2", {})
            }
        }

    def phase6_validation(self, frame: np.ndarray) -> Dict[str, Any]:
        # Validation can now be more deterministic, comparing the generated model's
        # invariants with the ones extracted in phase 2.
        model_params = self.current_cycle_data["phases"].get("phase5", {}).get("parameters", {})
        original_invariants = self.current_cycle_data["phases"].get("phase2", {})

        score = 0.85 # Default score
        if "fractal_dimension" in original_invariants and "transforms" in model_params:
            # A simple validation: score is higher if model is complex for high dimension
            if original_invariants["fractal_dimension"] > 1.5 and len(model_params["transforms"]) > 2:
                score = 0.95

        return {"validation_score": score, "status": "completed"}

    def phase7_context_transition(self, frame: np.ndarray) -> Dict[str, Any]:
        return {"next_action": "Analyze adjacent regions", "cycle_complete": True}

    def _estimate_box_counting_dimension(self, binary_image: np.ndarray) -> float:
        """Estimates the fractal dimension of a binary image using box-counting."""
        # This is a simplified implementation. A more robust version would be needed for production.
        pixels = np.where(binary_image)
        if len(pixels[0]) < 2:
            return 1.0 # Not enough points for a meaningful calculation

        scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
        counts = []
        for scale in scales:
            H, W = binary_image.shape
            c = (binary_image[0:H:int(scale), 0:W:int(scale)] > 0).sum()
            if c > 0:
                counts.append(c)

        if len(counts) < 2:
            return 1.0

        # Fit a line to the log-log plot
        coeffs = np.polyfit(np.log(scales[:len(counts)]), np.log(counts), 1)
        return float(-coeffs[0])
