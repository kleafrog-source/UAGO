import pytest
import numpy as np
import cv2
from uago_core import UAGOCore
from uago_config import UAGO_CONFIG, DEMO_DATA

@pytest.fixture
def sample_image():
    image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    cv2.circle(image, (150, 150), 80, (255, 255, 255), -1)
    cv2.circle(image, (150, 150), 40, (0, 0, 0), -1)

    return image

@pytest.fixture
def uago_demo():
    return UAGOCore(api_key=None, demo_mode=True)

@pytest.fixture
def uago_live():
    return UAGOCore(api_key=None, demo_mode=False)

def test_uago_initialization_demo_mode():
    uago = UAGOCore(api_key=None, demo_mode=True)
    assert uago.demo_mode == True
    assert uago.mistral_client is None

def test_uago_initialization_live_mode():
    uago = UAGOCore(api_key=None, demo_mode=False)
    assert uago.demo_mode == False

def test_phase1_structure_detection_demo(uago_demo, sample_image):
    result = uago_demo.phase1_structure_detection(sample_image)

    assert "roi" in result
    assert "structure_detected" in result
    assert "complexity_score" in result
    assert isinstance(result["structure_detected"], bool)

def test_phase1_structure_detection_live(uago_live, sample_image):
    result = uago_live.phase1_structure_detection(sample_image)

    assert "roi" in result
    assert "structure_detected" in result
    assert "complexity_score" in result

    if result["roi"]:
        assert "x" in result["roi"]
        assert "y" in result["roi"]
        assert "width" in result["roi"]
        assert "height" in result["roi"]

def test_phase2_invariant_extraction_demo(uago_demo, sample_image):
    uago_demo.current_cycle_data = {"phases": {"phase1": {"roi": None}}}
    result = uago_demo.phase2_invariant_extraction(sample_image)

    assert "dimensionality" in result
    assert "scales" in result
    assert "connectivity" in result
    assert "repetition" in result
    assert "symmetry" in result

def test_phase2_invariant_extraction_live(uago_live, sample_image):
    uago_live.current_cycle_data = {"phases": {"phase1": {"roi": {"x": 50, "y": 50, "width": 200, "height": 200}}}}
    result = uago_live.phase2_invariant_extraction(sample_image)

    assert "dimensionality" in result
    assert isinstance(result["dimensionality"], float)
    assert isinstance(result["scales"], list)
    assert result["connectivity"] in ["high", "medium", "low"]

def test_phase3_hypothesis_generation_demo(uago_demo, sample_image):
    uago_demo.current_cycle_data = {"phases": {"phase2": {"dimensionality": 1.5}}}
    result = uago_demo.phase3_hypothesis_generation(sample_image)

    assert "hypotheses" in result
    assert isinstance(result["hypotheses"], list)
    assert len(result["hypotheses"]) > 0

    for hyp in result["hypotheses"]:
        assert "id" in hyp
        assert "desc" in hyp
        assert "priority" in hyp

def test_phase3_hypothesis_generation_live(uago_live, sample_image):
    uago_live.current_cycle_data = {
        "phases": {
            "phase2": {
                "dimensionality": 1.7,
                "repetition": True,
                "symmetry": "approximate"
            }
        }
    }
    result = uago_live.phase3_hypothesis_generation(sample_image)

    assert "hypotheses" in result
    assert isinstance(result["hypotheses"], list)

def test_phase4_adaptive_measurement_demo(uago_demo, sample_image):
    uago_demo.current_cycle_data = {
        "phases": {
            "phase3": {
                "hypotheses": [
                    {"id": "H1", "desc": "self-similar structure", "priority": 0.9}
                ]
            }
        }
    }
    result = uago_demo.phase4_adaptive_measurement(sample_image)

    assert "measurements_requested" in result
    assert "measured_values" in result

def test_phase5_model_search_demo(uago_demo, sample_image):
    uago_demo.current_cycle_data = {
        "phases": {
            "phase2": {"dimensionality": 1.5},
            "phase3": {"hypotheses": []},
            "phase4": {"measurements_requested": []}
        }
    }
    result = uago_demo.phase5_model_search(sample_image)

    assert "model" in result
    assert "formula" in result
    assert "parameters" in result

def test_phase6_validation_demo(uago_demo, sample_image):
    uago_demo.current_cycle_data = {
        "phases": {
            "phase5": {"model": "IFS"}
        }
    }
    result = uago_demo.phase6_validation(sample_image)

    assert "predictions" in result
    assert "validation_score" in result
    assert isinstance(result["validation_score"], float)

def test_phase7_context_transition_demo(uago_demo, sample_image):
    uago_demo.current_cycle_data = {
        "phases": {
            "phase1": {"roi": {"x": 0, "y": 0, "width": 100, "height": 100}}
        }
    }
    result = uago_demo.phase7_context_transition(sample_image)

    assert "next_scale" in result
    assert "context_shift" in result

def test_full_cycle_demo(uago_demo, sample_image):
    result = uago_demo.process_frame(sample_image)

    assert "timestamp" in result
    assert "input_shape" in result
    assert "phases" in result

    for i in range(1, 8):
        phase_key = f"phase{i}"
        assert phase_key in result["phases"]

def test_full_cycle_live(uago_live, sample_image):
    result = uago_live.process_frame(sample_image)

    assert "timestamp" in result
    assert "input_shape" in result
    assert "phases" in result

    for i in range(1, 8):
        phase_key = f"phase{i}"
        assert phase_key in result["phases"]

def test_config_structure():
    assert "name" in UAGO_CONFIG
    assert "version" in UAGO_CONFIG
    assert "observation_cycle" in UAGO_CONFIG
    assert len(UAGO_CONFIG["observation_cycle"]) == 7

def test_demo_data_structure():
    assert len(DEMO_DATA) >= 3

    for key, data in DEMO_DATA.items():
        for i in range(1, 8):
            phase_key = f"phase{i}"
            assert phase_key in data

def test_box_counting_dimension(uago_live):
    binary_image = np.zeros((100, 100), dtype=bool)
    binary_image[25:75, 25:75] = True

    dim = uago_live._estimate_box_counting_dimension(binary_image)

    assert isinstance(dim, float)
    assert 1.0 <= dim <= 2.0

def test_rule_based_hypotheses(uago_live):
    invariants = {
        "dimensionality": 1.5,
        "repetition": True,
        "symmetry": "approximate"
    }

    result = uago_live._generate_rule_based_hypotheses(invariants)

    assert "hypotheses" in result
    assert len(result["hypotheses"]) > 0

def test_rule_based_model_fractal(uago_live):
    invariants = {"dimensionality": 1.5}
    hypotheses = {
        "hypotheses": [
            {"id": "H1", "desc": "Fractal structure", "priority": 0.9}
        ]
    }

    result = uago_live._generate_rule_based_model(invariants, hypotheses)

    assert "model" in result
    assert "IFS" in result["model"]

def test_rule_based_model_spiral(uago_live):
    invariants = {"dimensionality": 1.0}
    hypotheses = {
        "hypotheses": [
            {"id": "H1", "desc": "Spiral pattern", "priority": 0.9}
        ]
    }

    result = uago_live._generate_rule_based_model(invariants, hypotheses)

    assert "model" in result
    assert "spiral" in result["model"].lower()

def test_progress_callback(uago_live, sample_image):
    callback_calls = []

    def callback(phase, name, progress):
        callback_calls.append((phase, name, progress))

    uago_live.process_frame(sample_image, callback)

    assert len(callback_calls) >= 7
    assert callback_calls[-1][2] == 100

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
