import json
import os
import zipfile
from datetime import datetime
from typing import Dict, Any
import logging
import numpy as np
import cv2

def setup_logging(log_level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger("UAGO")
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load config: {e}")

    return {"api_key": None, "demo_mode": True}

def save_config(config: Dict[str, Any], config_path: str = "config.json"):
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save config: {e}")

def save_phase_data(phase_num: int, data: Dict[str, Any], output_dir: str = "output") -> str:
    os.makedirs(output_dir, exist_ok=True)

    # Use filesystem-safe timestamp format (no colons)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"phase{phase_num}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return filepath
    except Exception as e:
        logging.error(f"Failed to save phase data: {e}")
        return ""

def save_full_cycle(cycle_data: Dict[str, Any], output_dir: str = "output") -> str:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"uago_cycle_{timestamp}.zip"
    zip_filepath = os.path.join(output_dir, zip_filename)

    try:
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            cycle_json = f"cycle_{timestamp}.json"
            zipf.writestr(cycle_json, json.dumps(cycle_data, indent=2))

            for phase_num in range(1, 8):
                phase_key = f"phase{phase_num}"
                if phase_key in cycle_data.get("phases", {}):
                    phase_json = f"phase{phase_num}_{timestamp}.json"
                    zipf.writestr(
                        phase_json,
                        json.dumps(cycle_data["phases"][phase_key], indent=2)
                    )

            readme = f"""UAGO Observation Cycle Results
Generated: {timestamp}

This archive contains the complete mathematical analysis of an observed structure.

Files:
- cycle_{timestamp}.json: Complete cycle data
- phase1-7_{timestamp}.json: Individual phase results

Phases:
1. Primary Structure Detection
2. Coarse Invariant Extraction
3. Hypothesis Generation
4. Adaptive Measurement Request
5. Integration & Minimal Model Search
6. Predictive Validation & Refinement
7. Scale / Context Transition

For more information, see the UAGO framework documentation.
"""
            zipf.writestr("README.txt", readme)

        return zip_filepath
    except Exception as e:
        logging.error(f"Failed to create cycle archive: {e}")
        return ""

def process_image_input(image_path: str) -> np.ndarray:
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise

def process_video_input(video_path: str, frame_skip: int = 30) -> list:
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                frames.append(frame)

            frame_count += 1

        cap.release()
        return frames
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise

def capture_from_webcam() -> np.ndarray:
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not capture frame from webcam")

        return frame
    except Exception as e:
        logging.error(f"Error capturing from webcam: {e}")
        raise

def create_roi_overlay(image: np.ndarray, roi: Dict[str, int]) -> np.ndarray:
    overlay = image.copy()

    if roi and all(k in roi for k in ['x', 'y', 'width', 'height']):
        x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            "ROI",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    return overlay

def format_phase_result(phase_num: int, data: Dict[str, Any]) -> str:
    formatted = f"Phase {phase_num} Results:\n"
    formatted += "-" * 50 + "\n"

    for key, value in data.items():
        if isinstance(value, dict):
            formatted += f"{key}:\n"
            for k, v in value.items():
                formatted += f"  {k}: {v}\n"
        elif isinstance(value, list):
            formatted += f"{key}: {', '.join(map(str, value[:5]))}"
            if len(value) > 5:
                formatted += f" ... ({len(value)} total)"
            formatted += "\n"
        else:
            formatted += f"{key}: {value}\n"

    return formatted

class LogHandler(logging.Handler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        log_entry = self.format(record)
        self.callback(log_entry)
