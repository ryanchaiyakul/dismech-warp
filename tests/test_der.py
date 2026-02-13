import pytest
import warp as wp


@pytest.fixture
def triplet(device):
    nodes = wp.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=wp.vec3, device=device)
    thetas = wp.array([0.0, 0.0], dtype=float, device=device)
    d1s = wp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    t1s = wp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    betas = wp.array([0.0, 0.0], dtype=float, device=device)

    # Thinking of making this static? Would recomputing the kernel be worth it?
    l_ks = wp.array([1.0, 1.0], dtype=float, device=device)
    ks = wp.array([100.0, 100.0, 100.0, 100.0], dtype=float, device=device)
    bar_strain = wp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float, device=device)


def test_grad():
    pass
