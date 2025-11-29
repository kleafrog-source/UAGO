import pytest
import numpy as np
import cv2
from uago_core import UAGOCore

# --- Constants ---
TEST_PROJECT_NAME = "test_project"

# --- Fixtures ---

@pytest.fixture
def sample_image():
    """Generate a sample image with a simple geometric shape for detection."""
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    # Draw a prominent white rectangle that's easy to detect
    cv2.rectangle(image, (50, 50), (250, 250), (255, 255, 255), -1)
    return image

@pytest.fixture
def uago_instance():
    """Fixture for a UAGOCore instance, initialized for tests."""
    core = UAGOCore(project_name=TEST_PROJECT_NAME, api_key=None, demo_mode=True)
    # Initialize the cycle data structure, which is now done in process_frame
    core.current_cycle_data = {"phases": {}}
    return core

# --- Tests ---

def test_uago_initialization(uago_instance):
    """Test the basic initialization of the UAGOCore."""
    assert uago_instance.project_name == TEST_PROJECT_NAME
    assert uago_instance.demo_mode is True

def test_phase1_structure_detection(uago_instance, sample_image):
    """Test that Phase 1 correctly identifies the rectangle in the sample image."""
    result = uago_instance.phase1_structure_detection(sample_image)
    assert result["structure_detected"] is True
    assert "roi" in result
    # ROI should be close to the 200x200 rectangle we drew
    assert result["roi"]["width"] > 190

# def test_phase2_invariant_extraction(uago_instance, sample_image):
#     """Test that Phase 2 extracts basic geometric invariants."""
#     uago_instance.current_cycle_data["phases"]["phase1"] = uago_instance.phase1_structure_detection(sample_image)
#     result = uago_instance.phase2_invariant_extraction(sample_image)
#     assert "fractal_dimension" in result
#     assert "hu_moments" in result
#     # A solid rectangle should have a dimension close to 2
#     assert result["fractal_dimension"] > 1.8

def test_phase5_model_search_demo_mode(uago_instance, sample_image):
    """Test that Phase 5 in demo mode returns the default fallback model."""
    uago_instance.current_cycle_data["phases"]["phase2"] = {"fractal_dimension": 1.9}
    result = uago_instance.phase5_model_search(sample_image)

    assert result["model"] == "Unknown"
    assert result["viz_type"] == "point_cloud"
    assert "invariants" in result["parameters"]

def test_full_cycle_demo_mode(uago_instance, sample_image):
    """Test a full processing cycle in demo mode."""
    result = uago_instance.process_frame(sample_image)

    assert "timestamp" in result
    assert "phases" in result
    for i in range(1, 8):
        assert f"phase{i}" in result["phases"]

    assert result["phases"]["phase5"]["model"] == "Unknown"

# def test_box_counting_dimension_line(uago_instance):
#     """Test the fractal dimension of a line, which should be close to 1."""
#     binary_image = np.zeros((100, 100), dtype=np.uint8)
#     binary_image[50, :] = 255

#     dim = uago_instance._estimate_box_counting_dimension(binary_image)
#     # The dimension of a line should be ~1
#     assert 0.9 <= dim <= 1.2
