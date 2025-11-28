import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json
import logging
from datetime import datetime
import io
import os
import base64
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

# Setup project directories
PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True, parents=True)
output_dir = Path("output")
output_dir.mkdir(exist_ok=True, parents=True)

from uago_core import UAGOCore
from uago_config import UAGO_CONFIG, DEMO_DATA
from uago_utils import (
    setup_logging, load_config, save_config,
    save_phase_data, save_full_cycle,
    process_image_input, process_video_input,
    create_roi_overlay, format_phase_result
)

# Project management functions
def get_project_path(project_name: str) -> Path:
    """Get the path to a project directory."""
    return PROJECTS_DIR / project_name

def project_exists(project_name: str) -> bool:
    """Check if a project exists."""
    return get_project_path(project_name).exists()

def create_project(project_name: str) -> bool:
    """Create a new project directory structure."""
    if not project_name:
        return False
        
    project_path = get_project_path(project_name)
    if project_path.exists():
        return False
        
    try:
        project_path.mkdir(parents=True)
        (project_path / 'data').mkdir()
        (project_path / 'output').mkdir()
        (project_path / 'config').mkdir()
        
        # Create default project config
        project_config = {
            'name': project_name,
            'created_at': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'description': 'UAGO Project',
            'version': '1.0.0',
            'cycle_data': None
        }
        
        with open(project_path / 'project_config.json', 'w', encoding='utf-8') as f:
            json.dump(project_config, f, indent=2)
            
        return True
    except Exception as e:
        st.error(f"Failed to create project: {e}")
        return False

def delete_project(project_name: str) -> bool:
    """Delete a project and all its contents."""
    if not project_name:
        return False
        
    project_path = get_project_path(project_name)
    if not project_path.exists():
        return False
        
    try:
        shutil.rmtree(project_path)
        return True
    except Exception as e:
        st.error(f"Failed to delete project: {e}")
        return False

def save_project(project_name: str, cycle_data: Dict) -> bool:
    """Save project data to the project directory."""
    try:
        project_path = get_project_path(project_name)
        
        # Create necessary directories
        (project_path / 'output').mkdir(parents=True, exist_ok=True)
        (project_path / 'logs').mkdir(parents=True, exist_ok=True)
        (project_path / 'prompts').mkdir(parents=True, exist_ok=True)
        (project_path / 'debug').mkdir(parents=True, exist_ok=True)
        
        # Save cycle data
        cycle_data_path = project_path / 'cycle_data.json'
        with open(cycle_data_path, 'w', encoding='utf-8') as f:
            json.dump(cycle_data, f, indent=2)
        
        # Save logs if they exist
        if 'logs' in st.session_state and st.session_state.logs:
            log_path = project_path / 'logs' / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(st.session_state.logs))
            
        # Update project config
        config_path = project_path / 'project_config.json'
        config = {
            'name': project_name,
            'last_modified': datetime.now().isoformat(),
            'version': '1.0.0',
            'directories': {
                'output': str(project_path / 'output'),
                'logs': str(project_path / 'logs'),
                'prompts': str(project_path / 'prompts'),
                'debug': str(project_path / 'debug')
            },
            'files': {
                'cycle_data': str(cycle_data_path.relative_to(project_path))
            }
        }
        
        # Preserve existing config if it exists
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
                # Preserve non-overlapping keys
                for key, value in existing_config.items():
                    if key not in config:
                        config[key] = value
        
        # Save the updated config
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
            
        # Export a zip of the project for backup
        export_project(project_name)
            
        return True
    except Exception as e:
        add_log(f"Error saving project: {str(e)}", "ERROR")
        return False

def export_project(project_name: str) -> Optional[str]:
    """Export project to a zip file in the project's output directory."""
    try:
        project_path = get_project_path(project_name)
        if not project_path.exists():
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{project_name}_{timestamp}.zip"
        
        # Create exports directory in the project's output folder
        exports_dir = project_path / 'output' / 'exports'
        exports_dir.mkdir(parents=True, exist_ok=True)
        zip_path = exports_dir / zip_filename
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(project_path):
                # Skip the exports directory to avoid recursive zipping
                if 'exports' in root.split(os.sep):
                    continue
                    
                for file in files:
                    file_path = Path(root) / file
                    arcname = str(file_path.relative_to(project_path.parent))
                    zipf.write(file_path, arcname)
        
        # Also create a copy in the root exports directory for backward compatibility
        root_exports_dir = Path("exports")
        root_exports_dir.mkdir(exist_ok=True)
        shutil.copy2(zip_path, root_exports_dir / zip_filename)
                    
        return str(zip_path)
    except Exception as e:
        add_log(f"Error exporting project: {str(e)}", "ERROR")
        return None

def list_projects() -> List[str]:
    """List all available projects."""
    if not PROJECTS_DIR.exists():
        return []
    return [d.name for d in PROJECTS_DIR.iterdir() if d.is_dir()]

def load_project(project_name: str) -> Optional[Dict]:
    """Load project data from the project directory."""
    try:
        project_path = get_project_path(project_name)
        cycle_data_path = project_path / 'cycle_data.json'
        
        if not cycle_data_path.exists():
            return None
            
        with open(cycle_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        add_log(f"Error loading project: {str(e)}", "ERROR")
        return None

# Import visualization components
try:
    from visualizer.uago_viz import render_model
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False

# Page configuration
st.set_page_config(
    page_title="UAGO - Universal Adaptive Geometric Observer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for project management
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'project_initialized' not in st.session_state:
    st.session_state.project_initialized = False
if 'show_project_list' not in st.session_state:
    st.session_state.show_project_list = False

# Create default project if it doesn't exist
def create_default_project():
    default_project = "Default Project"
    if not project_exists(default_project):
        create_project(default_project)
        # Add some default data to the project
        default_data = {
            'name': default_project,
            'created_at': datetime.now().isoformat(),
            'description': 'Default project with example data',
            'version': '1.0.0',
            'cycle_data': None
        }
        project_path = get_project_path(default_project)
        with open(project_path / 'project_config.json', 'w', encoding='utf-8') as f:
            json.dump(default_data, f, indent=2)
        return default_project
    return default_project

if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'cycle_data' not in st.session_state:
    st.session_state.cycle_data = None
if 'visualization_html' not in st.session_state:
    st.session_state.visualization_html = None
if 'show_visualization' not in st.session_state:
    st.session_state.show_visualization = False
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = 0
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'api_key' not in st.session_state:
    config = load_config()
    st.session_state.api_key = config.get('api_key', '')
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = not bool(st.session_state.api_key)

def add_log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {level}: {message}")

def progress_callback(phase: int, phase_name: str, progress: float):
    st.session_state.current_phase = phase
    st.session_state.progress = progress
    add_log(f"Phase {phase}: {phase_name}", "INFO")

# Create default project if it doesn't exist
if not list_projects():
    create_default_project()

# Sidebar - Project Management
with st.sidebar:
    st.header("Project Management")
    
    # Current project display
    if st.session_state.get('current_project'):
        st.markdown(f"**Current Project:** {st.session_state.current_project}")
    else:
        st.markdown("**No active project**")
    
    # Project actions
    col1, col2 = st.columns(2)
    
    with col1:
        # Button to show project list
        if st.button("📂 Project List"):
            st.session_state.show_project_list = True
    
    with col2:
        # Clear project button
        if st.button("🆕 New Project"):
            st.session_state.show_new_project = True
    
    # Project list modal
    if st.session_state.get('show_project_list', False):
        with st.sidebar.expander("📋 Project List", expanded=True):
            projects = list_projects()
            if projects:
                for project in projects:
                    cols = st.columns([3, 1])
                    with cols[0]:
                        if st.button(f"📄 {project}", key=f"load_{project}"):
                            st.session_state.current_project = project
                            cycle_data = load_project(project)
                            if cycle_data:
                                st.session_state.cycle_data = cycle_data
                                st.session_state.project_initialized = True
                            st.session_state.show_project_list = False
                            st.rerun()
                    with cols[1]:
                        if st.button("🗑️", key=f"del_{project}"):
                            if delete_project(project):
                                if st.session_state.get('current_project') == project:
                                    st.session_state.current_project = None
                                    st.session_state.cycle_data = None
                                    st.session_state.project_initialized = False
                                st.rerun()
            else:
                st.info("No projects found")
            
            if st.button("Close"):
                st.session_state.show_project_list = False
                st.rerun()
    
    # New project input
    if st.session_state.get('show_new_project', False):
        new_project = st.text_input("New Project Name:", "", key="new_project_input", 
                                  on_change=lambda: st.session_state.update({"create_new_project": True}))
        
        if st.session_state.get('create_new_project', False) and new_project:
            if create_project(new_project):
                st.session_state.current_project = new_project
                st.session_state.project_initialized = True
                st.session_state.show_new_project = False
                st.session_state.create_new_project = False
                st.rerun()
    
    # Save Project button
    if st.session_state.get('current_project') and st.button("💾 Save Project"):
        if 'cycle_data' in st.session_state and st.session_state.cycle_data is not None:
            if save_project(st.session_state.current_project, st.session_state.cycle_data):
                st.sidebar.success(f"Project '{st.session_state.current_project}' saved successfully!")
            else:
                st.sidebar.error("Failed to save project")
        else:
            st.sidebar.warning("No cycle data to save")
    
    st.markdown("---")
    st.markdown("### System Status")
    st.markdown(f"**Project Initialized:** {st.session_state.get('project_initialized', False)}")
    
    # Load default project if none is loaded
    if not st.session_state.get('current_project') and not st.session_state.get('project_initialized', False):
        default_project = create_default_project()
        st.session_state.current_project = default_project
        cycle_data = load_project(default_project)
        if cycle_data:
            st.session_state.cycle_data = cycle_data
            st.session_state.project_initialized = True
            st.rerun()

# Main content
st.title("🔬 Universal Adaptive Geometric Observer (UAGO)")
st.markdown("*Autonomous discovery of deep mathematical structures in visual data*")

# Initialize session state for visualization
if 'visualization_html' not in st.session_state:
    st.session_state.visualization_html = None
if 'show_visualization' not in st.session_state:
    st.session_state.show_visualization = False

st.sidebar.header("Configuration")

with st.sidebar.expander("🔑 API Settings", expanded=True):
    api_key_input = st.text_input(
        "Mistral API Key",
        value=st.session_state.api_key,
        type="password",
        help="Enter your Mistral API key for AI-powered hypothesis generation"
    )

    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        config = load_config()
        config['api_key'] = api_key_input
        save_config(config)
        st.success("API key saved!")

    st.session_state.demo_mode = st.checkbox(
        "Demo Mode (use preloaded data)",
        value=not bool(st.session_state.api_key),
        help="Use demo data instead of live processing"
    )

    if 'refine_models' not in st.session_state:
        st.session_state.refine_models = False  # FIXED: Init session_state

    st.session_state.refine_models = st.checkbox(
        "Enable Model Refinement (extra API call)",
        value=st.session_state.refine_models,
        help="Enable model refinement using AI"
    )

    if st.session_state.demo_mode:
        st.info("Running in demo mode with preloaded examples")

with st.sidebar.expander("ℹ️ About UAGO", expanded=False):
    st.markdown(f"""
    **{UAGO_CONFIG['name']}**

    Version: {UAGO_CONFIG['version']}

    {UAGO_CONFIG['description']}

    **7 Observation Phases:**
    1. Primary Structure Detection
    2. Coarse Invariant Extraction
    3. Hypothesis Generation
    4. Adaptive Measurement Request
    5. Integration & Minimal Model Search
    6. Predictive Validation & Refinement
    7. Scale / Context Transition
    """)

tab1, tab2 = st.tabs(["📤 Process Input", "📊 Results & Export"])

with tab1:
    st.header("Input Selection")

    col1, col2 = st.columns([2, 1])

    with col1:
        input_type = st.radio(
            "Input Type",
            ["Upload Image", "Upload Video", "Use Demo"],
            horizontal=True
        )

        uploaded_file = None
        if input_type == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload an image file",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                help="Supported formats: PNG, JPG, JPEG, BMP"
            )
        elif input_type == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload a video file",
                type=['mp4', 'avi', 'mov'],
                help="Video will be analyzed at key frames"
            )
            frame_skip = st.slider(
                "Frame skip interval",
                min_value=10,
                max_value=120,
                value=30,
                help="Process every Nth frame"
            )

    with col2:
        st.markdown("### Preview")
        preview_placeholder = st.empty()

        if uploaded_file is not None:
            if input_type == "Upload Image":
                image = Image.open(uploaded_file)
                preview_placeholder.image(image, caption="Uploaded Image", width='stretch')
            elif input_type == "Upload Video":
                preview_placeholder.info("Video uploaded. Click 'Process' to analyze key frames.")
        elif input_type == "Use Demo":
            demo_keys = list(DEMO_DATA.keys())
            selected_demo = st.selectbox("Select demo example", demo_keys)
            st.info(f"Demo: {selected_demo}")

    st.markdown("---")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

    with col_btn1:
        process_button = st.button("▶️ Process", type="primary", width='stretch')

    with col_btn2:
        clear_button = st.button("🗑️ Clear Logs", width='stretch')

    if clear_button:
        st.session_state.logs = []
        st.session_state.cycle_data = None
        st.rerun()

    if process_button:
        st.session_state.logs = []
        st.session_state.cycle_data = None

        try:
            add_log("Initializing UAGO system...", "INFO")

            uago = UAGOCore(
                api_key=st.session_state.api_key if not st.session_state.demo_mode else None,
                demo_mode=st.session_state.demo_mode,
                max_refinements=1 if st.session_state.refine_models else 0  # FIXED: Pass from UI
            )

            frame = None

            if input_type == "Upload Image" and uploaded_file is not None:
                try:
                    # Reset file pointer to start in case it was read before
                    uploaded_file.seek(0)
                    # Read file content
                    file_bytes = uploaded_file.read()
                    # Convert to numpy array
                    nparr = np.frombuffer(file_bytes, np.uint8)
                    # Decode image
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError("Failed to decode image. The file might be corrupted or in an unsupported format.")
                    add_log(f"Image loaded successfully. Dimensions: {frame.shape}", "INFO")
                except Exception as e:
                    add_log(f"Error loading image: {str(e)}", "ERROR")
                    raise

            elif input_type == "Upload Video" and uploaded_file:
                temp_path = f"temp_video_{datetime.now().timestamp()}.mp4"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.read())

                frames = process_video_input(temp_path, frame_skip)
                if frames:
                    frame = frames[0]
                    add_log(f"Video processed: {len(frames)} key frames extracted", "INFO")
                os.remove(temp_path)

            elif input_type == "Use Demo":
                frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
                add_log("Using demo mode with synthetic data", "INFO")

            if frame is not None:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(phase, name, prog):
                    progress_callback(phase, name, prog)
                    progress_bar.progress(int(prog))
                    status_text.text(f"Phase {phase}: {name}... {int(prog)}%")

                cycle_data = uago.process_frame(frame, update_progress)
                st.session_state.cycle_data = cycle_data

                add_log("Processing complete!", "INFO")
                progress_bar.progress(100)
                status_text.success("✅ All phases completed successfully!")

            else:
                st.error("No valid input provided")
                add_log("Error: No valid input", "ERROR")

        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            add_log(f"Error: {str(e)}", "ERROR")

    st.markdown("---")

    st.subheader("📝 Process Logs")
    log_container = st.container()
    with log_container:
        if st.session_state.logs:
            log_text = "\n".join(st.session_state.logs[-50:])
            st.text_area(
                "Logs",
                value=log_text,
                height=200,
                disabled=True,
                label_visibility="collapsed"
            )
        else:
            st.info("No logs yet. Process an input to see logs.")

with tab2:
    st.header("Analysis Results")

    if st.session_state.cycle_data:
        cycle = st.session_state.cycle_data

        st.success(f"Cycle completed at {cycle.get('timestamp', 'N/A')}")

        phase_tabs = st.tabs([f"Phase {i}" for i in range(1, 8)] + ["Summary", "Export"])

        for i in range(1, 8):
            with phase_tabs[i-1]:
                phase_key = f"phase{i}"
                phase_data = cycle.get("phases", {}).get(phase_key, {})

                st.subheader(f"Phase {i}: {UAGO_CONFIG['observation_cycle'][i-1]['name']}")
                st.markdown(f"**Goal:** {UAGO_CONFIG['observation_cycle'][i-1].get('goal', 'N/A')}")

                if phase_data:
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.markdown("#### Results")
                        st.json(phase_data)

                    with col2:
                        st.markdown("#### Formatted Output")
                        formatted = format_phase_result(i, phase_data)
                        st.code(formatted, language="text")

                    if i == 5 and "model" in phase_data:
                        st.markdown("#### 📐 Mathematical Model")
                        st.latex(phase_data.get("formula", "N/A"))

        with phase_tabs[7]:
            st.subheader("📊 Complete Analysis Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Extracted Invariants")
                phase2 = cycle.get("phases", {}).get("phase2", {})
                if phase2:
                    st.write(f"**Dimensionality:** {phase2.get('dimensionality', 'N/A')}")
                    st.write(f"**Scales:** {phase2.get('scales', [])}")
                    st.write(f"**Connectivity:** {phase2.get('connectivity', 'N/A')}")
                    st.write(f"**Repetition:** {phase2.get('repetition', 'N/A')}")
                    st.write(f"**Symmetry:** {phase2.get('symmetry', 'N/A')}")

            with col2:
                st.markdown("#### Discovered Model")
                phase5 = cycle.get("phases", {}).get("phase5", {})
                if phase5:
                    st.write(f"**Model Type:** {phase5.get('model', 'N/A')}")
                    st.code(phase5.get('formula', 'N/A'))
                    st.write(f"**Parameters:** {phase5.get('parameters', {})}")

            st.markdown("#### Emergent Properties")
            phase6 = cycle.get("phases", {}).get("phase6", {})
            if phase6 and "predictions" in phase6:
                for pred in phase6["predictions"]:
                    st.write(f"- {pred}")
                st.write(f"**Validation Score:** {phase6.get('validation_score', 0):.2%}")

        with phase_tabs[8]:
            st.subheader("💾 Export & Visualization")

            if VISUALIZATION_ENABLED and st.session_state.cycle_data:
                st.markdown("#### 🎨 Interactive Visualization")

                # --- Interactive Controls ---
                cols = st.columns(2)
                with cols[0]:
                    iterations = st.slider("Iterations", min_value=1, max_value=10, value=5, help="Controls the point density for IFS fractals.")
                with cols[1]:
                    scale = st.slider("Scale", min_value=0.1, max_value=2.0, value=1.0, step=0.1, help="Global scaling factor for the visualization.")

                # --- Generate and Display Plotly Figure ---
                try:
                    fig = render_model(
                        json_data=st.session_state.cycle_data,
                        iterations=iterations,
                        scale=scale
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # --- HTML Export Button ---
                    html_buffer = io.StringIO()
                    fig.write_html(html_buffer)
                    st.download_button(
                        label="Download as HTML",
                        data=html_buffer.getvalue(),
                        file_name="uago_visualization.html",
                        mime="text/html",
                    )

                except Exception as e:
                    st.error(f"Failed to generate visualization: {e}")

                st.markdown("---")


            st.subheader("💾 Export Options")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### Save Individual Phase")
                phase_num = st.selectbox("Select phase", list(range(1, 8)))
                if st.button("Save Phase as JSON"):
                    phase_key = f"phase{phase_num}"
                    phase_data = cycle.get("phases", {}).get(phase_key, {})
                    filepath = save_phase_data(phase_num, phase_data)
                    if filepath:
                        st.success(f"Saved to {filepath}")
                        with open(filepath, 'r') as f:
                            st.download_button(
                                "Download JSON",
                                data=f.read(),
                                file_name=os.path.basename(filepath),
                                mime="application/json"
                            )

            with col2:
                st.markdown("#### Save Complete Cycle")
                if st.button("Save Full Cycle as ZIP"):
                    zip_path = save_full_cycle(cycle)
                    if zip_path:
                        st.success(f"Saved to {zip_path}")
                        with open(zip_path, 'rb') as f:
                            st.download_button(
                                "Download ZIP",
                                data=f.read(),
                                file_name=os.path.basename(zip_path),
                                mime="application/zip"
                            )

            with col3:
                st.markdown("#### Save Results")
                if st.button("Export Results"):
                    if st.session_state.cycle_data:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"uago_export_{timestamp}.json"
                        try:
                            with open(filename, 'w') as f:
                                json.dump(st.session_state.cycle_data, f, indent=2)
                            st.success(f"Results exported to {filename}")
                            
                            # Add download button for the exported file
                            with open(filename, 'r') as f:
                                st.download_button(
                                    "Download Export",
                                    data=f.read(),
                                    file_name=filename,
                                    mime="application/json"
                                )
                        except Exception as e:
                            st.error(f"Failed to export results: {e}")
                    else:
                        st.warning("No results to export")

            st.markdown("Debug Information")
            with st.expander("View Raw Cycle Data"):
                st.json(cycle)

            if 'mistral_logs' in cycle and cycle['mistral_logs']:
                with st.expander("Mistral API Logs (Prompts/Responses)"):
                    for log in cycle['mistral_logs']:
                        st.subheader(f"Phase {log['phase']} Iter {log.get('iteration', 1)}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text_area("Prompt", log['prompt'], height=100, key=f"prompt_{len(log['prompt'])}")
                        with col2:
                            st.text_area("Response", log['response'], height=100, key=f"resp_{len(log['response'])}")
                        st.json(log['parsed'])  # Parsed output
                        # Simple copy button (shows code in Streamlit; for clipboard, use st.download_button below)
                        if st.button("Copy Prompt", key=f"copy_{len(log['prompt'])}"):
                            st.code(log['prompt'])
                        # Optional: Download prompt as txt
                        st.download_button(
                            "Download Prompt TXT",
                            data=log['prompt'],
                            file_name=f"prompt_phase{log['phase']}_iter{log.get('iteration',1)}.txt",
                            mime="text/plain",
                            key=f"dl_prompt_{len(log['prompt'])}"
                        )
            else:
                st.info("No Mistral logs available (demo mode or no API calls).")

        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: gray;'>UAGO v1.0 | "
            "Universal Adaptive Geometric Observer | Autonomous Mathematical Structure Discovery</div>",
            unsafe_allow_html=True
        )
