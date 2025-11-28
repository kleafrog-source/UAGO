# UAGO System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     UAGO Application                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐              ┌──────────────┐            │
│  │   Mode 1     │              │   Mode 2     │            │
│  │ Web Interface│              │  Local GUI   │            │
│  │  (Streamlit) │              │  (Tkinter)   │            │
│  └──────┬───────┘              └──────┬───────┘            │
│         │                              │                     │
│         └──────────────┬───────────────┘                     │
│                        │                                     │
│                ┌───────▼────────┐                           │
│                │  UAGO Core     │                           │
│                │  7-Phase Engine│                           │
│                └───────┬────────┘                           │
│                        │                                     │
│         ┌──────────────┼──────────────┐                     │
│         │              │              │                     │
│  ┌──────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐            │
│  │   Config    │ │  Utils   │ │ Mistral API │            │
│  │& Demo Data  │ │I/O, Logs │ │  (Optional) │            │
│  └─────────────┘ └──────────┘ └─────────────┘            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      main.py (Launcher)                      │
│  - Command-line interface                                    │
│  - Mode selection (--mode web/local)                         │
│  - Port configuration                                        │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼──────┐    ┌───────▼──────┐
│app_streamlit │    │ app_tkinter  │
│   .py        │    │    .py       │
│              │    │              │
│ - Web UI     │    │ - Desktop UI │
│ - Tabs       │    │ - Notebook   │
│ - Upload     │    │ - Webcam     │
│ - Progress   │    │ - Threading  │
│ - Export     │    │ - Dialogs    │
└───────┬──────┘    └───────┬──────┘
        │                   │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │   uago_core.py    │
        │                   │
        │ class UAGOCore:   │
        │   __init__()      │
        │   process_frame() │
        │   phase1()        │
        │   phase2()        │
        │   phase3()        │
        │   phase4()        │
        │   phase5()        │
        │   phase6()        │
        │   phase7()        │
        └─────────┬─────────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
┌───────▼────┐ ┌──▼─────┐ ┌▼──────────┐
│uago_config │ │uago_   │ │ mistralai │
│   .py      │ │utils   │ │  (ext)    │
│            │ │ .py    │ │           │
│- Config    │ │        │ │- API      │
│- Phases    │ │- Logs  │ │  client   │
│- Demo data │ │- I/O   │ │- Chat     │
└────────────┘ │- Export│ │  complete │
               │- Webcam│ └───────────┘
               └────────┘
```

## Data Flow Diagram

```
┌──────────┐
│  Input   │  (Image/Video/Webcam)
└────┬─────┘
     │
     ▼
┌─────────────────┐
│  Input Handler  │  Load & validate
└────┬────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│          UAGO 7-Phase Pipeline          │
├─────────────────────────────────────────┤
│                                         │
│  Phase 1: Structure Detection           │
│    Input: Raw frame                     │
│    Output: ROI, complexity              │
│         │                               │
│         ▼                               │
│  Phase 2: Invariant Extraction          │
│    Input: Frame + ROI                   │
│    Output: Dimension, scales, symmetry  │
│         │                               │
│         ▼                               │
│  Phase 3: Hypothesis Generation         │
│    Input: Invariants                    │
│    Output: Hypotheses list              │
│    → Mistral API or rule-based          │
│         │                               │
│         ▼                               │
│  Phase 4: Adaptive Measurement          │
│    Input: Hypotheses + frame            │
│    Output: Measured values              │
│         │                               │
│         ▼                               │
│  Phase 5: Model Search                  │
│    Input: All prior data                │
│    Output: Mathematical model           │
│    → Mistral API or rule-based          │
│         │                               │
│         ▼                               │
│  Phase 6: Validation                    │
│    Input: Model                         │
│    Output: Predictions, score           │
│         │                               │
│         ▼                               │
│  Phase 7: Context Transition            │
│    Input: Full cycle                    │
│    Output: Next scale/context           │
│                                         │
└────────────┬────────────────────────────┘
             │
             ▼
      ┌──────────────┐
      │ Cycle Data   │  JSON structure
      └──────┬───────┘
             │
      ┌──────┴───────┐
      │              │
┌─────▼────┐   ┌─────▼────┐
│  GUI     │   │  Export  │
│ Display  │   │  Files   │
└──────────┘   └──────────┘
```

## Class Structure

```
UAGOCore
├── Properties
│   ├── api_key: str
│   ├── demo_mode: bool
│   ├── mistral_client: Mistral
│   ├── logger: Logger
│   └── current_cycle_data: dict
│
├── Public Methods
│   ├── __init__(api_key, demo_mode)
│   └── process_frame(frame, callback) → dict
│
├── Phase Methods
│   ├── phase1_structure_detection(frame) → dict
│   ├── phase2_invariant_extraction(frame) → dict
│   ├── phase3_hypothesis_generation(frame) → dict
│   ├── phase4_adaptive_measurement(frame) → dict
│   ├── phase5_model_search(frame) → dict
│   ├── phase6_validation(frame) → dict
│   └── phase7_context_transition(frame) → dict
│
└── Private Methods
    ├── _estimate_box_counting_dimension(image) → float
    ├── _build_hypothesis_prompt(invariants) → str
    ├── _build_model_search_prompt(...) → str
    ├── _parse_mistral_response(content) → dict
    ├── _parse_model_response(content) → dict
    ├── _generate_rule_based_hypotheses(inv) → dict
    └── _generate_rule_based_model(inv, hyp) → dict
```

## State Management

### Streamlit (Web Mode)
```
st.session_state:
├── logs: list[str]
├── cycle_data: dict
├── current_phase: int
├── progress: float
├── api_key: str
└── demo_mode: bool
```

### Tkinter (Local Mode)
```
UAGOApp instance:
├── config: dict
├── api_key: str
├── demo_mode: bool
├── current_frame: np.ndarray
├── cycle_data: dict
├── processing: bool
└── UI widgets (Entry, Text, Progress, etc.)
```

## Module Dependencies

```
uago_core.py
├── numpy
├── cv2 (opencv-python)
├── scipy.ndimage
├── skimage (feature, measure)
├── mistralai (optional)
├── logging
└── uago_config

uago_utils.py
├── json
├── os
├── zipfile
├── datetime
├── logging
├── numpy
└── cv2

app_streamlit.py
├── streamlit
├── numpy
├── cv2
├── PIL
├── json
├── logging
├── datetime
├── uago_core
├── uago_config
└── uago_utils

app_tkinter.py
├── tkinter
├── cv2
├── PIL
├── numpy
├── json
├── threading
├── datetime
├── uago_core
├── uago_config
└── uago_utils
```

## Processing Flow

1. **Initialization**
   ```
   User launches → main.py → app_streamlit.py or app_tkinter.py
   Load config.json → Initialize UAGOCore
   ```

2. **Input Stage**
   ```
   User selects input → Validate → Convert to np.ndarray
   Display preview
   ```

3. **Processing Stage**
   ```
   User clicks Process →
   Create UAGOCore instance →
   Call process_frame() →
   Loop through 7 phases →
   Update GUI via callback →
   Store results in cycle_data
   ```

4. **Output Stage**
   ```
   Display results in tabs/notebook →
   User views phase data →
   User exports (JSON/ZIP)
   ```

## Error Handling Strategy

```
┌─────────────────┐
│   Try Block     │  Each phase wrapped
└────────┬────────┘
         │
    ┌────▼─────┐
    │ Success? │
    └────┬─────┘
         │
    ┌────▼──────────┐
    │ Yes │   No    │
    │     │         │
    │     └─────────┤
    │               │
    │       ┌───────▼────────┐
    │       │ Log Error      │
    │       │ Store in phase │
    │       │ Continue cycle │
    │       └────────────────┘
    │
    └───────┬────────┐
            │        │
     ┌──────▼──┐  ┌──▼────────┐
     │ Return  │  │ GUI shows │
     │ result  │  │ error msg │
     └─────────┘  └───────────┘
```

## Configuration Flow

```
Application Start
       │
       ▼
Check config.json exists?
       │
   ┌───┴───┐
   │  Yes  │  No  │
   │       │      │
   ▼       ▼      │
Load   Create     │
file   default    │
   │       │      │
   └───┬───┘      │
       │          │
       ▼          ▼
Apply settings
       │
       ▼
Initialize UAGOCore
       │
       ▼
Check api_key?
       │
   ┌───┴────┐
   │  Set   │  Empty  │
   │        │         │
   ▼        ▼         │
Live    Demo         │
mode    mode         │
```

## API Integration

```
Phase 3 or Phase 5 reached
         │
         ▼
Check mistral_client exists?
         │
    ┌────┴────┐
    │ Yes  No │
    │         │
    ▼         ▼
Call API  Fallback
    │         │
    ├─────────┤
    │ Timeout │ 3 retries
    ├─────────┤
    │         │
    ▼         ▼
Parse    Rule-based
response algorithm
    │         │
    └────┬────┘
         │
         ▼
    Return result
```

## Testing Architecture

```
test_uago.py
├── Fixtures
│   ├── sample_image()
│   ├── uago_demo()
│   └── uago_live()
│
├── Initialization Tests
│   ├── test_demo_mode()
│   └── test_live_mode()
│
├── Phase Tests (1-7)
│   ├── test_phase1_demo()
│   ├── test_phase1_live()
│   ├── test_phase2_demo()
│   ├── test_phase2_live()
│   └── ... (all phases)
│
├── Integration Tests
│   ├── test_full_cycle_demo()
│   └── test_full_cycle_live()
│
├── Config Tests
│   ├── test_config_structure()
│   └── test_demo_data_structure()
│
└── Algorithm Tests
    ├── test_box_counting()
    ├── test_rule_based_hypotheses()
    └── test_rule_based_models()
```

## Summary

The UAGO architecture follows a layered design:

1. **Presentation Layer**: Streamlit (web) or Tkinter (desktop)
2. **Application Layer**: Core processing engine with 7 phases
3. **Data Layer**: Configuration, demo data, utilities
4. **External Layer**: Mistral API integration

Key architectural principles:
- **Separation of concerns**: Each module has a single responsibility
- **Modularity**: Components can be tested independently
- **Extensibility**: Easy to add new models or phases
- **Robustness**: Error handling at every level
- **Flexibility**: Multiple interfaces, optional API, demo mode
