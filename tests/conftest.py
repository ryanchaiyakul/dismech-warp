import pytest
import warp as wp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Warp device to run tests on (e.g., 'cpu', 'cuda:0')",
    )


@pytest.fixture(scope="session")
def device(request):
    """
    Returns the device string specified by the user.
    Skips tests if CUDA is requested but not available.
    """
    target_device = request.config.getoption("--device")
    if "cuda" in target_device and not wp.is_cuda_available():
        pytest.skip(f"Requested device '{target_device}' is not available.")
    return target_device
