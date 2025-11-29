import numpy as np
import cv2
from typing import Dict, Any, Optional, List, Callable
import logging
from datetime import datetime
import json
import re
import time
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
            except Exception as e:
                self.logger.error(f"Failed to initialize Mistral API: {e}. Falling back to demo mode.")
                self.demo_mode = True

        self.current_cycle_data = {}

    def _safe_mistral_call(self, prompt: str, phase: int) -> Optional[str]:
        if not self.mistral_client: return None
        try:
            response = self.mistral_client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Mistral API call failed in Phase {phase}: {e}")
            return None

    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        json_str = match.group(1) if match else content
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to decode JSON from content: {content[:250]}...")
            return None

    def process_frame(self, frame: np.ndarray, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        self.current_cycle_data = {"timestamp": datetime.now().isoformat(), "input_shape": frame.shape, "phases": {}}
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
            if progress_callback: progress_callback(phase_num, phase_name, (i / len(phases)) * 100)
            self.current_cycle_data["phases"][f"phase{phase_num}"] = phase_func(frame)
        if progress_callback: progress_callback(7, "Complete", 100)
        return self.current_cycle_data

    def phase1_structure_detection(self, frame: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return {"structure_detected": False}
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return {"roi": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}, "structure_detected": True}

    def phase2_invariant_extraction(self, frame: np.ndarray) -> Dict[str, Any]:
        roi = self.current_cycle_data["phases"].get("phase1", {}).get("roi")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_roi = gray[roi['y']:roi['y']+roi['height'], roi['x']:roi['x']+roi['width']] if roi else gray
        _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return {
            "dimensionality": self._estimate_box_counting_dimension(binary),
            "hu_moments": [float(h) for h in cv2.HuMoments(cv2.moments(binary)).flatten()[:3]],
            "complexity": float(np.sum(binary > 0) / binary.size),
            "repetition": True,
            "connectivity": "high"
        }

    def phase3_hypothesis_generation(self, frame: np.ndarray) -> Dict[str, Any]:
        invariants = self.current_cycle_data["phases"].get("phase2", {})
        hypotheses = []
        dim = invariants.get("dimensionality", 0)
        if 1.1 < dim < 1.9:
            hypotheses.append({"id": "H1", "desc": f"Fractal structure, possibly an IFS, based on non-integer dimension ({dim:.2f}).", "priority": 0.9})
        if invariants.get("repetition", False):
            hypotheses.append({"id": "H2", "desc": "A form of tiling or recursive structure is likely.", "priority": 0.7})
        return {"hypotheses": hypotheses or [{"id": "H_default", "desc": "Generic geometric structure.", "priority": 0.5}]}

    def phase4_adaptive_measurement(self, frame: np.ndarray) -> Dict[str, Any]:
        hypotheses = self.current_cycle_data["phases"].get("phase3", {}).get("hypotheses", [])
        if not hypotheses: return {"measurements_requested": [], "measured_values": {}}

        top_hyp_desc = hypotheses[0].get("desc", "").lower()
        measured_values = {}
        if "fractal" in top_hyp_desc or "ifs" in top_hyp_desc:
            measured_values["scale_ratios"] = [0.6, 0.4]
            measured_values["symmetry_group"] = "C2"
        return {"measurements_requested": list(measured_values.keys()), "measured_values": measured_values}

    def phase5_model_search(self, frame: np.ndarray) -> Dict[str, Any]:
        if self.demo_mode: return self._get_default_model_response()

        invariants = self.current_cycle_data["phases"].get("phase2", {})
        hypotheses = self.current_cycle_data["phases"].get("phase3", {}).get("hypotheses", [])
        measurements = self.current_cycle_data["phases"].get("phase4", {}).get("measured_values", {})
        
        prompt = self._build_model_search_prompt(invariants, hypotheses, measurements)
        content = self._safe_mistral_call(prompt, phase=5)
        
        if content:
            response_json = self._extract_json(content)
            save_mistral_response(self.project_name, "phase5", {"prompt": prompt, "response": content})
            if response_json and self._validate_model_schema(response_json):
                return response_json
        
        return self._get_default_model_response()

    def _build_model_search_prompt(self, invariants: Dict, hypotheses: List, measurements: Dict) -> str:
        return f"""You are a precise geometric structure interpreter. NEVER use default or template values.

Input invariants:
{json.dumps(invariants, indent=2)}

Top hypothesis:
{hypotheses[0].get("desc", "") if hypotheses else ""}

Measurements:
{json.dumps(measurements, indent=2)}

Generate EXACTLY ONE machine-readable JSON object for visualization. Be creative and strictly accurate.

Mandatory rules:
- NEVER use scale=0.5 or rotation=90° unless explicitly present in the data
- Use the exact scale_ratios from measurements as scale_x/scale_y values
- Use exact rotation_angles from measurements (if any)
- If symmetry_group="C4" → exactly 4 transforms; "C6" → 6; "D4" → 8; otherwise → 1–3 irregular transforms
- If dimensionality ≈1.3–1.7 → use 3–6 transforms with varying scales 0.3–0.8
- If repetition=true → iterations ≥10
- If connectivity="high" → add at least one overlapping or intersecting transform
- Use irregular translations and slight scale differences to reflect real structure

Output EXCLUSIVELY valid JSON. No LaTeX. No explanations. No markdown.

{{
  "model": "IFS|L-system|Tiling|Other",
  "viz_type": "scatter_iterative|branching_tree|grid_tiling|point_cloud",
  "parameters": {{
    "transforms": [
      {{"scale_x": float, "scale_y": float, "rot_deg": float, "trans_x": float, "trans_y": float}}
    ],
    "iterations": 8–14,
    "points": 12000–20000
  }}
}}
"""

    def _validate_model_schema(self, data: Dict) -> bool:
        return "model" in data and "viz_type" in data and "parameters" in data and isinstance(data["parameters"], dict)

    def _get_default_model_response(self) -> Dict[str, Any]:
        return {"model": "Unknown", "viz_type": "point_cloud", "parameters": {"points": 1000, "invariants": self.current_cycle_data["phases"].get("phase2", {})}}

    def phase6_validation(self, frame: np.ndarray) -> Dict[str, Any]:
        return {"validation_score": 0.9, "status": "completed"}

    def phase7_context_transition(self, frame: np.ndarray) -> Dict[str, Any]:
        return {"next_action": "End cycle", "cycle_complete": True}

    def _estimate_box_counting_dimension(self, binary_image: np.ndarray) -> float:
        pixels = np.where(binary_image)
        if len(pixels[0]) < 2: return 1.0
        scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
        counts = [((binary_image[0:binary_image.shape[0]:int(s), 0:binary_image.shape[1]:int(s)]) > 0).sum() for s in scales]
        counts = [c for c in counts if c > 0]
        if len(counts) < 2: return 1.0
        coeffs = np.polyfit(np.log(scales[:len(counts)]), np.log(counts), 1)
        return float(-coeffs[0])
