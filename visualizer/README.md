# UAGO Visualizer

This module integrates UAGO with p5.js for interactive visualization of geometric structures.

## Setup

```bash
cd visualizer
npm install
```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run ../app_streamlit.py
   ```

2. Upload an image or use the demo data
3. Click "Start Analysis"
4. View the interactive visualization in the "3D Visualization" tab

## Architecture

- `sandbox/` - p5.js visualization environment
- `ai_interpreter.py` - AI-powered code generation from UAGO outputs
- `templates/` - HTML/JS templates for the visualization
- `static/` - Static assets (JS, CSS)

## License

MIT
