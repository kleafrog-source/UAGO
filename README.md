# UAGO - Universal Adaptive Geometric Observer

**Version 1.0**

A fully abstract, technology-agnostic framework for autonomous discovery of deep mathematical structure in any sensory stream.

## Overview

UAGO is an advanced system that analyzes visual data (images/videos) to extract mathematical invariants and discover underlying generative models. Unlike traditional computer vision that recognizes predefined objects, UAGO focuses on pure geometric, topological, and dynamical patterns.

## Core Principle

Rejection of pre-defined object ontologies in favor of pure geometric, topological, and dynamical invariants. The system does not recognize 'things'; it iteratively uncovers the hidden laws that generate the observable patterns.

## 7-Phase Observation Cycle

1. **Primary Structure Detection** - Detect coherent spatio-temporal organization
2. **Coarse Invariant Extraction** - Extract dimensionality, scales, connectivity, repetition, symmetry
3. **Hypothesis Generation** - Generate prioritized hypotheses about deeper structures
4. **Adaptive Measurement Request** - Specify and measure required next-level invariants
5. **Integration & Minimal Model Search** - Find smallest generative mathematical model
6. **Predictive Validation & Refinement** - Generate predictions and validate model
7. **Scale / Context Transition** - Shift to different scale or adjacent structure

## Features

### Two Operating Modes

**Mode 1: Web Application (Streamlit)**
- Browser-based interface
- Drag-and-drop file upload
- Real-time processing visualization
- Suitable for deployment and remote access

**Mode 2: Local GUI (Tkinter)**
- Desktop application
- Webcam support
- Local file processing
- No server required

### Key Capabilities

- **Mistral AI Integration** - Optional AI-powered hypothesis generation and model search
- **Demo Mode** - Preloaded examples for testing without API key
- **Progress Tracking** - Real-time phase-by-phase progress visualization
- **Comprehensive Logging** - Detailed logs with timestamps and log levels
- **Debug Information** - Full JSON dumps of all processing stages
- **Export Options** - Save individual phases or complete cycles as JSON/ZIP

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Obtain a Mistral API key from [Mistral AI](https://mistral.ai)

## Usage

### Quick Start

**Web Interface (Recommended):**
```bash
python main.py --mode web
```

Then open your browser to `http://localhost:8501`

**Local GUI:**
```bash
python main.py --mode local
```

### Custom Port (Web Mode)

```bash
python main.py --mode web --port 8080
```

### Direct Launch

**Streamlit:**
```bash
streamlit run app_streamlit.py
```

**Tkinter:**
```bash
python app_tkinter.py
```

## Configuration

### API Key Setup

1. In the application interface, navigate to Configuration/API Settings
2. Enter your Mistral API key
3. Click "Save" - the key is stored in `config.json`

Without an API key, the system runs in **Demo Mode** with preloaded examples.

### Demo Mode

Demo mode uses precomputed results from three sample patterns:
- **Snowflake** - IFS with 6-fold symmetry
- **Spiral** - Logarithmic spiral with golden ratio
- **Branching** - L-system with stochastic branching

## Input Types

### Web Mode
- Upload Image (PNG, JPG, JPEG, BMP)
- Upload Video (MP4, AVI, MOV) - analyzes key frames
- Use Demo data

### Local Mode
- Upload Image
- Upload Video
- Webcam Capture (live camera input)
- Use Demo data

## Output

### Per-Phase Results

Each phase produces structured JSON data:
- Phase 1: ROI coordinates, complexity score
- Phase 2: Dimensionality, scales, connectivity, symmetry
- Phase 3: Hypotheses with priorities
- Phase 4: Requested measurements and values
- Phase 5: Mathematical model with formula and parameters
- Phase 6: Predictions and validation score
- Phase 7: Next scale/context suggestions

### Export Formats

- **Individual Phase** - JSON file for specific phase
- **Complete Cycle** - ZIP archive with all phases + README
- **Raw JSON** - Full cycle data in single JSON file

## Architecture

```
project/
├── uago_config.py       # Configuration and demo data
├── uago_core.py         # Core UAGO processing engine (7 phases)
├── uago_utils.py        # Utilities (logging, I/O, visualization)
├── app_streamlit.py     # Web interface (Mode 1)
├── app_tkinter.py       # Local GUI (Mode 2)
├── main.py              # Launch script
├── test_uago.py         # Unit tests
├── requirements.txt     # Dependencies
└── README_UAGO.md       # This file
```

## Testing

Run unit tests:
```bash
pytest test_uago.py -v
```

Or:
```bash
python test_uago.py
```

## Technical Details

### Processing Pipeline

1. **Input Validation** - Convert to NumPy array, validate dimensions
2. **Phase Execution** - Sequential processing through 7 phases
3. **Progress Callbacks** - Real-time status updates to GUI
4. **Error Handling** - Try-except per phase with detailed logging
5. **Result Storage** - Structured JSON output per phase

### Mathematical Methods

- **Structure Detection** - Canny edge detection, contour analysis
- **Invariant Extraction** - Fourier analysis, box-counting dimension, moment analysis
- **Hypothesis Generation** - Mistral AI or rule-based heuristics
- **Measurement** - Hough transforms, morphological operations
- **Model Search** - Pattern matching to known generative systems (IFS, L-systems, etc.)

### API Integration

Mistral AI is used for:
- Phase 3: Hypothesis generation from extracted invariants
- Phase 5: Model selection based on measurements

Fallback to rule-based methods when API unavailable.

## Example Workflow

1. Launch application: `python main.py --mode web`
2. Enter API key (or enable Demo Mode)
3. Upload an image (e.g., fractal pattern, natural structure)
4. Click "Process"
5. Monitor progress through 7 phases
6. Review results:
   - Extracted invariants (dimensionality, scales, symmetry)
   - Discovered mathematical model (e.g., "IFS with 4 transformations")
   - Predictions and validation score
7. Export results as ZIP or JSON

## Troubleshooting

**Issue: "No module named 'mistralai'"**
- Solution: `pip install mistralai` or enable Demo Mode

**Issue: Webcam not working (Local Mode)**
- Solution: Check camera permissions, ensure no other app is using camera

**Issue: "API timeout" errors**
- Solution: Check internet connection, verify API key, or switch to Demo Mode

**Issue: Streamlit port already in use**
- Solution: Use custom port: `python main.py --mode web --port 8080`

## Limitations

- Video processing analyzes sampled frames (not full temporal dynamics)
- Fractal dimension estimation is approximate (box-counting method)
- Model search limited to predefined categories (IFS, L-systems, spirals, etc.)
- Symmetry detection is heuristic-based
- Best results with high-contrast, structured patterns

## Future Enhancements

- GPU acceleration for large images
- 3D structure analysis
- Temporal dynamics for videos
- Interactive ROI selection
- More generative model types
- Real-time video stream processing

## Credits

**Framework Design:** Conceptual architecture
**Implementation:** UAGO v1.0
**Date:** 2025-11-25

## License

This implementation is provided as-is for research and educational purposes.

## Support

For issues, questions, or contributions, refer to project documentation or contact the development team.

---

**UAGO - Discovering the geometric essence of reality**
