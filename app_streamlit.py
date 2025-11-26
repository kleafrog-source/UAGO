import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json
import logging
from datetime import datetime
import io
import os

from uago_core import UAGOCore
from uago_config import UAGO_CONFIG, DEMO_DATA
from uago_utils import (
    setup_logging, load_config, save_config,
    save_phase_data, save_full_cycle,
    process_image_input, process_video_input,
    create_roi_overlay, format_phase_result
)

st.set_page_config(
    page_title="UAGO - Universal Adaptive Geometric Observer",
    page_icon="🔬",
    layout="wide"
)

if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'cycle_data' not in st.session_state:
    st.session_state.cycle_data = None
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

st.title("🔬 Universal Adaptive Geometric Observer (UAGO)")
st.markdown("*Autonomous discovery of deep mathematical structures in visual data*")

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
    st.header("Results & Analysis")

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
                st.markdown("Export as JSON")
                json_str = json.dumps(cycle, indent=2)
                st.download_button(
                    "Download Cycle JSON",
                    data=json_str,
                    file_name=f"uago_cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

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
