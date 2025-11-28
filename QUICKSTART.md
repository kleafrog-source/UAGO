# UAGO Quick Start Guide

Get started with the Universal Adaptive Geometric Observer in 5 minutes!

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running UAGO

### Option 1: Web Interface (Recommended)

Launch the web application:
```bash
python main.py --mode web
```

Then open your browser to: `http://localhost:8501`

### Option 2: Local GUI

Launch the desktop application:
```bash
python main.py --mode local
```

## First Analysis

### Without API Key (Demo Mode)

1. Check "Demo Mode" in the configuration
2. Click "Use Demo" input type
3. Click "Process"
4. View results in the Results tab

### With Mistral API Key

1. Get your API key from [https://console.mistral.ai/](https://console.mistral.ai/)
2. Enter the key in "API Settings"
3. Click "Save API Key"
4. Upload an image with geometric patterns
5. Click "Process"
6. View the 7-phase analysis results

## What to Upload

UAGO works best with:
- Fractal patterns (snowflakes, ferns, coastlines)
- Spiral structures (shells, galaxies, phyllotaxis)
- Branching patterns (trees, lightning, river networks)
- Symmetric designs (mandalas, crystals, tessellations)
- Natural textures with mathematical structure

## Understanding Results

### Phase 2: Invariants
- **Dimensionality**: 1.0-2.0 (fractals have non-integer values)
- **Scales**: Characteristic length scales in the pattern
- **Symmetry**: Detected rotational or reflection symmetries

### Phase 5: Mathematical Model
- **IFS**: Iterated Function System (self-similar fractals)
- **L-system**: Lindenmayer system (branching structures)
- **Spiral**: Logarithmic or Archimedean curves
- **Formula**: Mathematical expression generating the pattern

### Phase 6: Validation
- **Predictions**: Testable properties of the model
- **Score**: Validation confidence (0-1, higher is better)

## Export Results

Click "Save Full Cycle" to get a ZIP with:
- All 7 phase results as JSON
- Complete cycle data
- README with analysis summary

## Testing

Run unit tests to verify installation:
```bash
pytest test_uago.py -v
```

## Troubleshooting

**Problem**: Import errors
**Solution**: `pip install -r requirements.txt`

**Problem**: API timeout
**Solution**: Enable "Demo Mode" or check internet connection

**Problem**: Webcam not working
**Solution**: Ensure no other app is using the camera

## Next Steps

1. Analyze your own images
2. Compare demo examples to understand output format
3. Export results for further analysis
4. Read README_UAGO.md for detailed documentation

## Example Workflow

```bash
# 1. Install
pip install -r requirements.txt

# 2. Create test image
python create_test_image.py

# 3. Launch web interface
python main.py --mode web

# 4. Upload test_image.jpg
# 5. Click Process
# 6. Explore results through 7 phases
# 7. Export as ZIP
```

## Support

For detailed documentation, see `README_UAGO.md`

---

Happy exploring! UAGO reveals the hidden mathematical beauty in visual patterns.
