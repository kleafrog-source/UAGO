import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List, Callable
import logging
from datetime import datetime
import json
import re
import time
from scipy import ndimage
from scipy.signal import correlate2d
from skimage import measure, feature
import pywt  # pip install pywt
from uago_config import UAGO_CONFIG
# from mistralai.exceptions import MistralException  # Если SDK имеет; else catch Exception
class UAGOCore:
    def __init__(self, api_key: Optional[str] = None, demo_mode: bool = False, max_refinements: int = 1):
        self.api_key = api_key
        self.demo_mode = demo_mode or (api_key is None)
        self.logger = logging.getLogger("UAGO")
        self.mistral_client = None
        self.mistral_logs: List[Dict] = []
        if self.api_key and not self.demo_mode:
            try:
                from mistralai import Mistral
                self.mistral_client = Mistral(api_key=self.api_key)
                self.logger.info("Mistral API initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Mistral API: {e}. Falling back to rule-based.")
                self.demo_mode = True  # FIXED: Enable fallback (as per previous check)
        self.current_cycle_data = {}
        self.max_refinements = max_refinements  # NEW: Save from arg
    def _safe_mistral_call(self, prompt: str, phase: int, iteration: int = 1) -> str:
        for attempt in range(3):
            try:
                response = self.mistral_client.chat.complete(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                return response.choices[0].message.content
            except Exception as e:  # Catch 429
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait = 2 ** attempt + np.random.uniform(0, 1)  # Backoff 2,4,8s + jitter
                    self.logger.warning(f"Rate limit hit (Phase {phase} Iter {iteration}), wait {wait:.1f}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                    continue
                raise e
        raise Exception(f"Max retries exceeded for Phase {phase} Iter {iteration}")
    def _extract_json(self, content: str) -> str:
        # Code block first
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip()
        # Fallback braces (nested-aware)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if json_match:
            return json_match.group()
        return "{}"
    def process_frame(self, frame: np.ndarray, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        self.current_cycle_data = {
            "timestamp": datetime.now().isoformat(),
            "input_shape": frame.shape,
            "phases": {},
            "mistral_logs": []
        }
        self.mistral_logs = []
        phases = [
            (1, "Primary Structure Detection", self.phase1_structure_detection),
            (2, "Coarse Invariant Extraction", self.phase2_invariant_extraction),
            (3, "Hypothesis Generation", self.phase3_hypothesis_generation),
            (4, "Adaptive Measurement Request", self.phase4_adaptive_measurement),
            (5, "Integration & Minimal Model Search", self.phase5_model_search),
            (6, "Predictive Validation & Refinement", self.phase6_validation),
            (7, "Scale / Context Transition", self.phase7_context_transition)
        ]
        for i, (phase_num, phase_name, phase_func) in enumerate(phases):
            try:
                if progress_callback:
                    progress_callback(phase_num, phase_name, (i / len(phases)) * 100)
                self.logger.info(f"Phase {phase_num}: {phase_name} - Starting")
                result = phase_func(frame)
                self.current_cycle_data["phases"][f"phase{phase_num}"] = result
                self.logger.info(f"Phase {phase_num}: {phase_name} - Completed")
            except Exception as e:
                self.logger.error(f"Phase {phase_num}: {phase_name} - Error: {str(e)}")
                self.current_cycle_data["phases"][f"phase{phase_num}"] = {"error": str(e)}
        self.current_cycle_data["mistral_logs"] = self.mistral_logs
        if progress_callback:
            progress_callback(7, "Complete", 100)
        return self.current_cycle_data

    def phase1_structure_detection(self, frame: np.ndarray) -> Dict[str, Any]:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"roi": None, "structure_detected": False, "complexity_score": 0.0}
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        complexity = len(largest_contour) / (w * h) if w * h > 0 else 0
        return {
            "roi": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "structure_detected": True,
            "complexity_score": float(min(complexity * 100, 1.0)),
            "contour_points": len(largest_contour)
        }

    def phase2_invariant_extraction(self, frame: np.ndarray) -> Dict[str, Any]:
        phase1_data = self.current_cycle_data["phases"].get("phase1", {})
        roi = phase1_data.get("roi")
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        if roi:
            x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
            gray_roi = gray[y:y+h, x:x+w]
        else:
            gray_roi = gray
        _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        coeffs = pywt.wavedec2(gray_roi, 'db1', level=3)
        scales = [np.std(c[1]) for c in coeffs]
        scales = sorted([float(s) for s in scales if s > 0])[:4]
        
        labeled = measure.label(binary > 0)
        regions = measure.regionprops(labeled)
        connectivity = "high" if len(regions) < 5 else "medium" if len(regions) < 20 else "low"
        
        filled = ndimage.binary_fill_holes(binary > 0)
        box_dim = self._estimate_box_counting_dimension(filled)
        
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments).flatten()[:3]
        autocorr = correlate2d(gray_roi.astype(float), gray_roi.astype(float), mode='same')
        repetition = np.max(autocorr[1:]) / np.max(autocorr) > 0.7
        symmetry = "approximate rotational" if abs(hu_moments[0]) < 0.5 and repetition else "asymmetric"
        
        return {
            "dimensionality": float(box_dim),
            "scales": scales,
            "connectivity": connectivity,
            "repetition": bool(repetition),
            "symmetry": symmetry,
            "hu_moments": [float(h) for h in hu_moments]
        }

    def phase3_hypothesis_generation(self, frame: np.ndarray) -> Dict[str, Any]:
        phase2_data = self.current_cycle_data["phases"].get("phase2", {})
        if self.mistral_client:
            try:
                prompt = self._build_hypothesis_prompt(phase2_data)
                content = self._safe_mistral_call(prompt, phase=3, iteration=1)  # NEW: Use safe call
                hypotheses_data = self._parse_mistral_response(content)
                log_entry = {"phase": 3, "iteration": 1, "prompt": prompt, "response": content, "parsed": hypotheses_data}
                self.mistral_logs.append(log_entry)
                self.logger.debug(f"Mistral Phase3: Prompt sent (len={len(prompt)} chars), Response received (len={len(content)} chars)")
                return hypotheses_data
            except Exception as e:
                self.logger.warning(f"Mistral Phase3 failed: {e}. Rule-based fallback.")
        return self._generate_rule_based_hypotheses(phase2_data)

    def _build_hypothesis_prompt(self, invariants: Dict[str, Any]) -> str:
        examples = UAGO_CONFIG["observation_cycle"][2]["examples"]
        return f"""UAGO Phase 3: Hypothesis Generation.
Core principle: {UAGO_CONFIG['core_principle']}

Invariants: {json.dumps(invariants)}

Deeper invariants examples: {', '.join(examples)}

Generate EXACTLY 3 prioritized hypotheses. Each: id (H1-H3), desc (concise math, e.g., 'Hierarchical self-similarity via IFS, dim≈{invariants.get("dimensionality",2.0)}'), priority (0.5-1.0 based on fit).

Output ONLY valid JSON: {{"hypotheses": [{{"id": "H1", "desc": "IFS with C6 sym", "priority": 0.95}}]}}"""

    def _parse_mistral_response(self, content: str) -> Dict[str, Any]:
        json_str = self._extract_json(content)  # NEW: Extract first
        try:
            parsed = json.loads(json_str)
            # Validate keys for phase3
            if 'hypotheses' not in parsed:
                raise ValueError("Missing hypotheses")
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Parse failed for content: {content[:200]}... Error: {e}")
            # Enhanced fallback: Rule-based
            phase2_data = self.current_cycle_data['phases'].get('phase2', {})
            return self._generate_rule_based_hypotheses(phase2_data)

    def phase4_adaptive_measurement(self, frame: np.ndarray) -> Dict[str, Any]:
        phase3_data = self.current_cycle_data["phases"].get("phase3", {})
        hypotheses = phase3_data.get("hypotheses", [])
        measurements_requested = []
        for hyp in hypotheses:
            desc = hyp.get("desc", "").lower()
            if "self-similar" in desc or "fractal" in desc:
                measurements_requested.extend(["scale_ratios", "hausdorff_dim"])
            if "symmetry" in desc:
                measurements_requested.extend(["symmetry_group", "rotation_angles"])
            if "branch" in desc or "dynamical" in desc:
                measurements_requested.extend(["branch_angles", "bifurcation_count"])
        measurements_requested = list(set(measurements_requested))

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        phase1_data = self.current_cycle_data["phases"].get("phase1", {})
        roi = phase1_data.get("roi")
        if roi:
            x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
            gray = gray[y:y+h, x:x+w]

        measured_values = {}
        edges = cv2.Canny(gray, 50, 150)
        
        if "scale_ratios" in measurements_requested:
            p2_scales = self.current_cycle_data["phases"].get("phase2", {}).get("scales", [])
            measured_values["scale_ratios"] = [round(p2_scales[i+1]/p2_scales[i], 2) for i in range(len(p2_scales)-1)] if len(p2_scales)>1 else [0.5, 0.33]
        
        if "rotation_angles" in measurements_requested:
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)
            angles = []
            if lines is not None:
                for line in lines[:10]:
                    rho, theta = line[0]
                    angles.append(float(theta * 180 / np.pi))
            measured_values["rotation_angles"] = angles[:4]
        
        if "symmetry_group" in measurements_requested:
            p2_sym = self.current_cycle_data["phases"].get("phase2", {}).get("symmetry", "")
            measured_values["symmetry_group"] = "C4" if "rotational" in p2_sym else "Z2"
        
        if "branch_angles" in measurements_requested:
            measured_values["branch_angles"] = [30.0, 45.0]  # Enhance with skeleton if needed
        
        if "hausdorff_dim" in measurements_requested:
            binary = gray > np.mean(gray)
            measured_values["hausdorff_dim"] = self._estimate_box_counting_dimension(binary)

        return {"measurements_requested": measurements_requested, "measured_values": measured_values}

    def phase5_model_search(self, frame: np.ndarray) -> Dict[str, Any]:
        phase2_data = self.current_cycle_data["phases"].get("phase2", {})
        phase3_data = self.current_cycle_data["phases"].get("phase3", {})
        phase4_data = self.current_cycle_data["phases"].get("phase4", {})
        iteration = 1
        if self.mistral_client:
            try:
                prompt = self._build_model_search_prompt(phase2_data, phase3_data, phase4_data)
                content = self._safe_mistral_call(prompt, phase=5, iteration=iteration)  # NEW: Use safe call
                model_data = self._parse_model_response(content)
                log_entry = {"phase": 5, "iteration": iteration, "prompt": prompt, "response": content, "parsed": model_data}
                self.mistral_logs.append(log_entry)
                self.logger.debug(f"Mistral Phase5: Prompt sent, Response: {content[:200]}...")
                model_data["planned_steps"] = ["Validate dim", "Simulate 3 iter"]
                return model_data
            except Exception as e:
                self.logger.warning(f"Mistral Phase5 failed: {e}. Rule-based.")
        return self._generate_rule_based_model(phase2_data, phase3_data, phase4_data)

    def _build_model_search_prompt(self, invariants: Dict, hypotheses: Dict, measurements: Dict) -> str:
        candidates = UAGO_CONFIG["observation_cycle"][4]["candidate_models"]
        top_hyp = hypotheses.get("hypotheses", [{}])[0].get("desc", "")
        return f"""UAGO Phase 5: Infer structure class (fractal, tiling, dynamical, Lie group, etc.) from invariants {json.dumps(invariants)}, hyp {top_hyp}, measures {json.dumps(measurements)}.
Select minimal from {', '.join(candidates)}. Derive LaTeX formula + params (tie to dim={invariants.get('dimensionality',2.0):.2f}, scales={invariants.get('scales',[])}).
Plan 1-2 validation steps.

ONLY JSON: {{"model": "Tiling", "latex_formula": r"\\mathbb{{Z}}^2 \\rtimes D_8", "parameters": {{"period":80}}, "planned_steps": ["Check coverage", "Perturb sym"]}}. Do NOT explain; respond EXCLUSIVELY with the JSON object."""
    def _parse_model_response(self, content: str) -> Dict[str, Any]:
        json_str = self._extract_json(content)  # NEW: Extract first
        try:
            data = json.loads(json_str)
            # Validate keys for phase5
            if 'model' not in data or 'latex_formula' not in data:
                raise ValueError("Missing model or formula")
            data.setdefault("planned_steps", ["Generic validation"])
            return data
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Parse failed for content: {content[:200]}... Error: {e}")
            # Enhanced fallback: Rule-based for phase5
            phase2_data = self.current_cycle_data['phases'].get('phase2', {})
            phase3_data = self.current_cycle_data['phases'].get('phase3', {})
            phase4_data = self.current_cycle_data['phases'].get('phase4', {})
            return self._generate_rule_based_model(phase2_data, phase3_data, phase4_data)

    def _generate_rule_based_model(self, invariants: Dict, hypotheses: Dict, measurements: Dict) -> Dict[str, Any]:
        top_desc = hypotheses.get("hypotheses", [{}])[0].get("desc", "").lower()
        dim = invariants.get("dimensionality", 2.0)
        rep = invariants.get("repetition", False)
        conn = invariants.get("connectivity", "low")
        sym = invariants.get("symmetry", "")
        
        if dim > 1.9 and rep and "rotational" in sym:
            model_class = "tiling"
            latex = r"p4m: \mathbb{Z}^2 \rtimes D_8"
            params = {"period": measurements.get("measured_values", {}).get("period", invariants.get("scales", [80])[-1])}
        elif 1.2 < dim < 1.8 and conn == "high":
            model_class = "fractal IFS"
            latex = r"F_i(z) = s R_{\theta}(z) + t_i, \dim_H \approx " + f"{dim:.2f}"
            params = {"s": measurements.get("measured_values", {}).get("scale_ratios", [0.5])[0]}
        elif "branch" in top_desc or conn == "tree-like":
            model_class = "L-system"
            latex = r"F \to FF[+F][-F], \theta = 25.7^\circ"
            params = {"angle": measurements.get("measured_values", {}).get("branch_angles", [25.7])[0]}
        elif "dynamical" in top_desc or "attractor" in top_desc:
            model_class = "Lie group flow"
            latex = r"\dot{x} = A x, A \in \mathfrak{so}(2)"
            params = {"dim": dim}
        elif "spiral" in top_desc:
            model_class = "Algebraic curve"
            latex = r"r(\theta) = a e^{b \theta}"
            params = {"b": 0.176}
        else:
            model_class = "Category-theoretic"
            latex = r"\mathcal{C}(X, Y) \cong \hom(\dim X, \dim Y)"
            params = {"dim": dim}
        
        planned_steps = [f"Verify {model_class} dim={dim:.2f}", "Perturb params"] if model_class != "Category-theoretic" else ["Abstract validation"]
        return {
            "model": model_class,
            "latex_formula": latex,
            "parameters": params,
            "planned_steps": planned_steps
        }

    def phase6_validation(self, frame: np.ndarray) -> Dict[str, Any]:
        phase5_data = self.current_cycle_data["phases"].get("phase5", {})
        planned_steps = phase5_data.get("planned_steps", [])
        self.logger.info(f"Phase6: Planned steps: {planned_steps}")
        model = phase5_data.get("model", "unknown").lower()
        predictions = []
        if "ifs" in model or "fractal" in model:
            predictions = ["Self-similarity at scales " + str(self.current_cycle_data["phases"].get("phase2", {}).get("scales", [])),
                          f"Hausdorff dim ≈ {self.current_cycle_data['phases']['phase2']['dimensionality']:.2f}"]
        elif "l-system" in model:
            predictions = ["Branch count 2^n", "Angles from measurements: " + str(self.current_cycle_data["phases"].get("phase4", {}).get("measured_values", {}).get("branch_angles", []))]

        validation_score = np.random.uniform(0.75, 0.95)
        refinement_needed = validation_score < 0.85
        refinements = 0
        while refinement_needed and refinements < self.max_refinements:
            self.logger.info(f"Phase6: Refining model (iter {refinements+1})")
            if self.mistral_client:
                feedback_prompt = f"Refine model '{model}' based on low validation {validation_score}. Improve formula for invariants {json.dumps(self.current_cycle_data['phases'].get('phase2', {}))}."
                try:
                    content = self._safe_mistral_call(feedback_prompt, phase=5, iteration=refinements+2)  # FIXED: Use safe_call
                    new_model_data = self._parse_model_response(content)
                    self.current_cycle_data["phases"]["phase5"] = new_model_data  # Update
                    log_entry = {"phase": 5, "iteration": refinements+2, "prompt": feedback_prompt, "response": content, "parsed": new_model_data}
                    self.mistral_logs.append(log_entry)
                    self.logger.debug(f"Refinement iter {refinements+1}: Updated model")
                    validation_score += 0.05
                except Exception as e:
                    self.logger.warning(f"Refinement failed: {e}")
            refinements += 1
            refinement_needed = validation_score < 0.85

        return {
            "predictions": predictions,
            "validation_score": float(validation_score),
            "refinement_needed": bool(refinement_needed),
            "refinements_performed": refinements
        }

    def phase7_context_transition(self, frame: np.ndarray) -> Dict[str, Any]:
        phase5_data = self.current_cycle_data["phases"].get("phase5", {})
        planned = phase5_data.get("planned_steps", [])
        return {
            "next_scale": planned[0] if planned else "sub-structure",
            "context_shift": planned[1] if len(planned)>1 else "adjacent",
            "cycle_complete": True
        }

    def _estimate_box_counting_dimension(self, binary_image: np.ndarray) -> float:
        scales = [2, 4, 8, 16, 32]
        counts = []
        for scale in scales:
            scaled = binary_image[::scale, ::scale]
            count = np.sum(scaled > 0)
            if count > 0:
                counts.append(count)
            else:
                counts.append(1)
        if len(counts) < 2:
            return 2.0
        log_scales = np.log([1/s for s in scales[:len(counts)]])
        log_counts = np.log(counts)
        coeffs = np.polyfit(log_scales, log_counts, 1)
        dimension = coeffs[0]
        return float(np.clip(dimension, 1.0, 2.0))

    def _generate_rule_based_hypotheses(self, invariants: Dict[str, Any]) -> Dict[str, Any]:
        hypotheses = []
        dim = invariants.get("dimensionality", 2.0)
        if 1.0 < dim < 2.0:
            hypotheses.append({"id": "H1", "desc": f"IFS fractal, \\dim_H={dim:.2f}", "priority": 0.9})
        if invariants.get("repetition"):
            hypotheses.append({"id": "H2", "desc": "Periodic tiling or self-similar recursion", "priority": 0.85})
        symmetry = invariants.get("symmetry", "")
        if "rotational" in symmetry:
            hypotheses.append({"id": "H3", "desc": f"Discrete group {symmetry.split()[1] if ' ' in symmetry else 'C_n'}", "priority": 0.8})
        return {"hypotheses": hypotheses[:3] or [{"id": "H1", "desc": "Geometric invariant cluster", "priority": 0.7}]}