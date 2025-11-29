import json
import os
import zipfile
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import numpy as np
import cv2
from pathlib import Path

# --- Project Path Management ---
PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)

def get_project_path(project_name: str) -> Optional[Path]:
    if not project_name:
        return None
    return PROJECTS_DIR / project_name

def setup_logging(project_name: str, log_level=logging.INFO) -> logging.Logger:
    """Sets up a logger that writes to a project-specific log file."""
    logger = logging.getLogger(f"UAGO-{project_name}")
    logger.setLevel(log_level)

    # Prevent duplicate handlers
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    log_dir = get_project_path(project_name) / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"uago_log_{timestamp}.txt"

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Also log to console for debugging
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def save_mistral_response(project_name: str, phase: str, response: Dict[str, Any]) -> str:
    """Saves the raw Mistral JSON response to the project's output directory."""
    project_path = get_project_path(project_name)
    output_dir = project_path / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mistral_{phase}_{timestamp}.json"
    filepath = output_dir / filename

    try:
        with open(filepath, 'w') as f:
            json.dump(response, f, indent=2)
        return str(filepath)
    except Exception as e:
        logging.error(f"Failed to save Mistral response: {e}")
        return ""

def save_phase_data(project_name: str, phase_num: int, data: Dict[str, Any]) -> str:
    """Saves data for a single phase to the project's output directory."""
    project_path = get_project_path(project_name)
    output_dir = project_path / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"phase{phase_num}_{timestamp}.json"
    filepath = output_dir / filename

    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return str(filepath)
    except Exception as e:
        logging.error(f"Failed to save phase data: {e}")
        return ""

def save_full_cycle(project_name: str, cycle_data: Dict[str, Any]) -> str:
    """Saves the entire cycle data as a JSON file in the project's output directory."""
    project_path = get_project_path(project_name)
    output_dir = project_path / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cycle_data_{timestamp}.json"
    filepath = output_dir / filename

    try:
        with open(filepath, 'w') as f:
            json.dump(cycle_data, f, indent=2)
        return str(filepath)
    except Exception as e:
        logging.error(f"Failed to save full cycle data: {e}")
        return ""

def export_project_zip(project_name: str) -> Optional[str]:
    """Creates a ZIP archive of the entire project directory."""
    project_path = get_project_path(project_name)
    if not project_path or not project_path.exists():
        logging.error(f"Project '{project_name}' not found for export.")
        return None

    export_dir = project_path / "output" / "exports"
    export_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"{project_name}_export_{timestamp}.zip"
    zip_filepath = export_dir / zip_filename

    try:
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in project_path.rglob('*'):
                # Don't include previous exports in the new zip
                if "exports" not in file_path.parts:
                    zipf.write(file_path, file_path.relative_to(project_path))
        return str(zip_filepath)
    except Exception as e:
        logging.error(f"Failed to create project archive for '{project_name}': {e}")
        return None

# --- Unchanged Utility Functions (No file I/O) ---

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
