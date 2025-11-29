import streamlit as st
import numpy as np
import cv2
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from uago_core import UAGOCore
from uago_config import UAGO_CONFIG, DEMO_DATA
from uago_utils import setup_logging, get_project_path, export_project_zip, save_full_cycle, process_image_input, process_video_input, format_phase_result
from visualizer.uago_viz import render_model, save_visualization_html

PROJECTS_DIR = Path("projects")
VALID_PROJECT_NAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]+$")

def list_projects() -> List[str]:
    return sorted([d.name for d in PROJECTS_DIR.iterdir() if d.is_dir()])

def create_project(project_name: str) -> bool:
    if not VALID_PROJECT_NAME_REGEX.match(project_name):
        st.error("Invalid name. Use only letters, numbers, _, -")
        return False
    project_path = get_project_path(project_name)
    if not project_path.exists():
        (project_path / 'output' / 'viz').mkdir(parents=True, exist_ok=True)
        (project_path / 'logs').mkdir(exist_ok=True)
        st.success(f"Project '{project_name}' created.")
    return True

def load_latest_cycle_data(project_name: str) -> Optional[Dict]:
    output_dir = get_project_path(project_name) / "output"
    if not output_dir.exists(): return None
    cycle_files = sorted(output_dir.glob("cycle_data_*.json"), reverse=True)
    if not cycle_files: return None
    try:
        with open(cycle_files[0], 'r') as f: return json.load(f)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

st.set_page_config(page_title="UAGO", page_icon="🔬", layout="wide")

if 'current_project' not in st.session_state:
    st.session_state.current_project = 'default_project'
if not get_project_path(st.session_state.current_project).exists():
    create_project(st.session_state.current_project)

with st.sidebar:
    st.markdown(f"### **Project: {st.session_state.current_project}**")
    
    new_project_name = st.text_input("Create or Load Project", help="Enter a valid name and press Enter.")
    if new_project_name and new_project_name != st.session_state.current_project:
        if VALID_PROJECT_NAME_REGEX.match(new_project_name):
            create_project(new_project_name)
            st.session_state.current_project = new_project_name
            st.session_state.cycle_data = load_latest_cycle_data(new_project_name)
            st.rerun()
        else:
            st.error("Invalid project name format.")

    st.markdown("---")
    
    with st.expander("🔑 API Settings", expanded=True):
        st.session_state.api_key = st.text_input("Mistral API Key", value=st.session_state.get('api_key', ''), type="password")
        st.session_state.demo_mode = st.checkbox("Demo Mode", value=not st.session_state.api_key)

logger = setup_logging(st.session_state.current_project)
st.title("🔬 Universal Adaptive Geometric Observer")

tab1, tab2 = st.tabs(["📤 Process Input", "📊 Results & Export"])

with tab1:
    input_type = st.radio("Input Type", ["Upload Image", "Upload Video", "Use Demo"], horizontal=True)
    frame = None
    if input_type == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            frame = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            st.image(frame, channels="BGR", caption="Uploaded Image")
    elif input_type == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
        if uploaded_file:
            tfile = Path(uploaded_file.name)
            with open(tfile, "wb") as f:
                f.write(uploaded_file.getbuffer())
            frames = process_video_input(str(tfile))
            if frames:
                frame = frames[0]
                st.image(frame, channels="BGR", caption="First frame of uploaded video")
            tfile.unlink()
    else:
        frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    if st.button("▶️ Process", type="primary", disabled=frame is None):
        project_name = st.session_state.current_project
        with st.spinner(f"Running analysis on project '{project_name}'..."):
            uago = UAGOCore(project_name=project_name, api_key=st.session_state.api_key, demo_mode=st.session_state.demo_mode)
            progress_bar = st.progress(0, "Starting...")
            st.session_state.cycle_data = uago.process_frame(frame, lambda p, n, prog: progress_bar.progress(int(prog), f"Phase {p}: {n}..."))
            save_full_cycle(project_name, st.session_state.cycle_data)
            st.success(f"Project saved: `projects/{project_name}/output/`")

with tab2:
    if not st.session_state.cycle_data:
        st.info("No data to display. Process an input or load a project.")
    else:
        project_name = st.session_state.current_project
        st.subheader("🎨 Interactive Visualization")
        fig = render_model(st.session_state.cycle_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            save_visualization_html(project_name, fig)
        
        st.subheader(f"💾 Export '{project_name}'")
        if st.button("Export Project as ZIP"):
            with st.spinner("Zipping project..."):
                zip_path_str = export_project_zip(project_name)
                if zip_path_str:
                    zip_path = Path(zip_path_str)
                    st.success(f"Project exported to {zip_path}")
                    with open(zip_path, "rb") as fp:
                        st.download_button("Download Project ZIP", fp, zip_path.name, "application/zip")

        st.subheader("🔬 Raw Data")
        phase_tabs = st.tabs([f"Phase {i}" for i in range(1, 8)] + ["Full JSON"])
        for i in range(1, 8):
            with phase_tabs[i-1]:
                st.markdown(f"**Goal:** {UAGO_CONFIG['observation_cycle'][i-1].get('goal', 'N/A')}")
                st.json(st.session_state.cycle_data.get("phases", {}).get(f"phase{i}", {}))
        with phase_tabs[7]:
            st.json(st.session_state.cycle_data)
