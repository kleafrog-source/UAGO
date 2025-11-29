import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json
import logging
from datetime import datetime
import io
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

# --- UAGO Core Imports ---
from uago_core import UAGOCore
from uago_config import UAGO_CONFIG, DEMO_DATA
from uago_utils import (
    setup_logging, get_project_path, export_project_zip,
    save_phase_data, save_full_cycle,
    process_image_input, process_video_input,
    create_roi_overlay, format_phase_result
)

# --- Visualization Imports (Deterministic) ---
try:
    from visualizer.uago_viz import render_model, save_visualization_html
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False
    st.error("Visualization module not found. Please ensure `visualizer/uago_viz.py` exists.")

# --- Constants and Setup ---
PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)

# --- Project Management Functions ---
def list_projects() -> List[str]:
    """List all available projects by scanning the projects directory."""
    if not PROJECTS_DIR.exists():
        return []
    return sorted([d.name for d in PROJECTS_DIR.iterdir() if d.is_dir()])

def create_project(project_name: str) -> bool:
    """Creates a new project directory structure."""
    if not project_name or "/" in project_name or "\\" in project_name:
        st.error("Invalid project name.")
        return False
    project_path = get_project_path(project_name)
    if project_path.exists():
        st.warning(f"Project '{project_name}' already exists.")
        return False
    try:
        project_path.mkdir(parents=True)
        (project_path / 'output' / 'viz').mkdir(parents=True, exist_ok=True)
        (project_path / 'logs').mkdir(exist_ok=True)
        st.success(f"Project '{project_name}' created.")
        return True
    except Exception as e:
        st.error(f"Failed to create project: {e}")
        return False

def load_latest_cycle_data(project_name: str) -> Optional[Dict]:
    """Loads the most recent cycle_data JSON from a project's output directory."""
    project_path = get_project_path(project_name)
    output_dir = project_path / "output"
    if not output_dir.exists():
        return None

    cycle_files = sorted(output_dir.glob("cycle_data_*.json"), reverse=True)
    if not cycle_files:
        return None
        
    try:
        with open(cycle_files[0], 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading cycle data for '{project_name}': {e}")
        return None

def recover_orphaned_files(project_name: str):
    """Moves files from root output/ and exports/ to a project's recovered folder."""
    if not project_name:
        return

    project_path = get_project_path(project_name)
    recovery_dir = project_path / "output" / "recovered"

    for root_dir_name in ["output", "exports"]:
        root_dir = Path(root_dir_name)
        if root_dir.exists():
            files = [f for f in root_dir.iterdir() if f.is_file()]
            if files:
                recovery_dir.mkdir(exist_ok=True, parents=True)
                for f in files:
                    try:
                        shutil.move(str(f), recovery_dir)
                    except Exception as e:
                        logging.warning(f"Could not move orphaned file {f}: {e}")
                logging.info(f"Recovered {len(files)} orphaned files to {recovery_dir}")

# --- Page and Session State Setup ---
st.set_page_config(page_title="UAGO", page_icon="🔬", layout="wide")

# Initialize session state keys
default_states = {
    'current_project': 'default_project',
    'logs': [],
    'cycle_data': None,
    'logger': None,
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Ensure default project exists on first run
if not get_project_path(st.session_state.current_project).exists():
    create_project(st.session_state.current_project)

# Recover orphaned files on startup
recover_orphaned_files(st.session_state.current_project)

# --- Sidebar UI ---
with st.sidebar:
    st.header("Project Management")
    
    # Project selection and creation
    st.session_state.current_project = st.text_input(
        "Current Project",
        value=st.session_state.current_project,
        help="Enter a name to create a new project or switch to an existing one."
    )
    
    projects = list_projects()
    if st.session_state.current_project not in projects:
        if st.button(f"Create Project '{st.session_state.current_project}'"):
            if create_project(st.session_state.current_project):
                st.rerun()

    selected_project = st.selectbox(
        "Load Existing Project",
        options=[""] + projects,
        index=0,
        key="project_selector"
    )
    if selected_project:
        st.session_state.current_project = selected_project
        st.session_state.cycle_data = load_latest_cycle_data(selected_project)
        st.session_state.project_selector = "" # Reset selectbox
        st.rerun()

    st.markdown("---")

    # Configuration and About sections
    with st.expander("🔑 API Settings", expanded=True):
        st.session_state.api_key = st.text_input(
            "Mistral API Key",
            value=st.session_state.get('api_key', ''),
            type="password"
        )
        st.session_state.demo_mode = st.checkbox(
            "Demo Mode",
            value=not bool(st.session_state.api_key),
            help="Use preloaded data instead of making live API calls."
        )

    with st.expander("ℹ️ About UAGO", expanded=False):
        st.markdown(f"**{UAGO_CONFIG['name']}** v{UAGO_CONFIG['version']}")
        st.markdown(UAGO_CONFIG['description'])

# --- Main Application UI ---
st.title("🔬 Universal Adaptive Geometric Observer (UAGO)")

# Setup project-specific logger
if st.session_state.current_project:
    st.session_state.logger = setup_logging(st.session_state.current_project)

def add_log(message: str, level: str = "INFO"):
    if st.session_state.logger:
        getattr(st.session_state.logger, level.lower(), st.session_state.logger.info)(message)
    st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {level}: {message}")

# --- Main Tabs ---
tab1, tab2 = st.tabs(["📤 Process Input", "📊 Results & Export"])

with tab1:
    st.header("Input Selection")
    input_type = st.radio("Input Type", ["Upload Image", "Use Demo"], horizontal=True)

    frame = None
    if input_type == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image file", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            bytes_data = uploaded_file.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            st.image(frame, channels="BGR", caption="Uploaded Image")

    elif input_type == "Use Demo":
        selected_demo = st.selectbox("Select demo example", list(DEMO_DATA.keys()))
        frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        add_log("Using demo mode with synthetic data")
        st.info(f"Running demo: {selected_demo}")

    if st.button("▶️ Process", type="primary", disabled=frame is None or not st.session_state.current_project):
        st.session_state.logs = []
        st.session_state.cycle_data = None
        project_name = st.session_state.current_project

        with st.spinner(f"Running UAGO analysis on project '{project_name}'..."):
            try:
                add_log(f"Initializing UAGO for project: {project_name}")
                uago = UAGOCore(
                    project_name=project_name,
                    api_key=st.session_state.api_key if not st.session_state.demo_mode else None,
                    demo_mode=st.session_state.demo_mode
                )

                progress_bar = st.progress(0, "Starting...")
                def update_progress(phase, name, prog):
                    progress_bar.progress(int(prog), f"Phase {phase}: {name}...")

                cycle_data = uago.process_frame(frame, update_progress)
                st.session_state.cycle_data = cycle_data

                # Save results automatically
                save_full_cycle(project_name, cycle_data)

                progress_bar.progress(100, "Analysis complete!")
                st.success(f"Processing complete! All data saved to 'projects/{project_name}/output/'.")

            except Exception as e:
                st.error(f"Error during processing: {e}")
                add_log(f"FATAL ERROR: {e}", "ERROR")

    st.subheader("📝 Process Logs")
    if st.session_state.logs:
        log_text = "\n".join(st.session_state.logs[-100:])
        st.text_area("Logs", value=log_text, height=300, disabled=True, label_visibility="collapsed")

with tab2:
    st.header("Analysis Results")
    if not st.session_state.cycle_data:
        st.info("No cycle data to display. Process an input on the first tab or load a project.")
    else:
        cycle = st.session_state.cycle_data
        project_name = st.session_state.current_project

        # --- Visualization Tab ---
        st.subheader("🎨 Interactive Visualization")
        if VISUALIZATION_ENABLED:
            try:
                fig = render_model(cycle)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    # Auto-save visualization
                    save_path = save_visualization_html(project_name, fig)
                    st.success(f"Visualization saved to {save_path}")
                else:
                    st.warning("Could not generate a visualization for this model.")
            except Exception as e:
                st.error(f"Failed to generate visualization: {e}")
        else:
            st.warning("Visualization is disabled.")

        st.markdown("---")

        # --- Export Tab ---
        st.subheader("💾 Export Project")
        if st.button(f"Export Project '{project_name}' as ZIP"):
            with st.spinner("Zipping project files..."):
                zip_path = export_project_zip(project_name)
                if zip_path:
                    st.success(f"Project exported successfully to {zip_path}")
                    with open(zip_path, "rb") as fp:
                        st.download_button(
                            label="Download ZIP",
                            data=fp,
                            file_name=os.path.basename(zip_path),
                            mime="application/zip",
                        )
                else:
                    st.error("Failed to export project.")

        st.markdown("---")

        # --- Data Tabs ---
        phase_tabs = st.tabs([f"Phase {i}" for i in range(1, 8)] + ["Full JSON"])
        for i in range(1, 8):
            with phase_tabs[i-1]:
                phase_data = cycle.get("phases", {}).get(f"phase{i}", {})
                if phase_data:
                    st.json(phase_data, expanded=True)
                else:
                    st.info(f"No data for Phase {i}.")

        with phase_tabs[7]:
            st.json(cycle)
