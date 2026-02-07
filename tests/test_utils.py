import pytest
import numpy as np
import warp as wp
import warp.autograd

from dismech_warp.utils import parallel_transport_kernel


@pytest.mark.parametrize(
    "input_u, input_t1, input_t2, expected",
    [
        # 1. Identity Transport (t1 == t2)
        ([1.0, 5.0, 2.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 5.0, 2.0]),
        # 2. 90-Degree Rotation (X -> Y)
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        # 3. Orthogonal Axis Preservation
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        # 4. 180-Degree Singularity (Regularized to Identity)
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        # 5. Zero Vector
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
        # 6. 45-Degree Rotation (XY Plane)
        (
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.70710678, 0.70710678, 0.0],
            [0.70710678, 0.70710678, 0.0],
        ),
        # 7. 60-Degree Rotation (XZ Plane)
        (
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.8660254],
            [0.5, 0.0, 0.8660254],
        ),
    ],
    ids=[
        "identity_transport",
        "90_deg_rotation",
        "orthogonal_preservation",
        "180_deg_singularity",
        "zero_vector",
        "45_deg_rotation",
        "60_deg_rotation",
    ],
)
def test_parallel_transport_cases(device, input_u, input_t1, input_t2, expected):
    u = wp.array([input_u], dtype=wp.vec3, device=device)
    t1 = wp.array([input_t1], dtype=wp.vec3, device=device)
    t2 = wp.array([input_t2], dtype=wp.vec3, device=device)
    out = wp.zeros_like(u)

    wp.launch(
        kernel=parallel_transport_kernel,
        dim=len(u),
        inputs=[u, t1, t2, out],
        device=device,
    )

    result = out.numpy()
    exp = np.array([expected], dtype=np.float32)
    np.testing.assert_allclose(result, exp)

    assert wp.autograd.gradcheck(
        parallel_transport_kernel,
        dim=len(u),
        inputs=[u, t1, t2],
        outputs=[out],
        device=device,
    )
