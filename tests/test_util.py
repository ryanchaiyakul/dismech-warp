import pytest
import numpy as np
import warp as wp
import warp.autograd

from dismech_warp.util import (
    parallel_transport,
    signed_angle,
    rotate_axis_angle,
)

# Kernels for local testing


@wp.kernel
def parallel_transport_kernel(
    u1s: wp.array(dtype=wp.vec3),
    t1s: wp.array(dtype=wp.vec3),
    t2s: wp.array(dtype=wp.vec3),
    u2s: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    u2s[tid] = parallel_transport(u1s[tid], t1s[tid], t2s[tid])


@wp.kernel
def signed_angle_kernel(
    us: wp.array(dtype=wp.vec3),
    vs: wp.array(dtype=wp.vec3),
    ns: wp.array(dtype=wp.vec3),
    angles: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    angles[tid] = signed_angle(us[tid], vs[tid], ns[tid])


@wp.kernel
def rotate_axis_angle_kernel(
    u1s: wp.array(dtype=wp.vec3),
    vs: wp.array(dtype=wp.vec3),
    thetas: wp.array(dtype=wp.float32),
    u2s: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    u2s[tid] = rotate_axis_angle(u1s[tid], vs[tid], thetas[tid])


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
)
def test_parallel_transport(device, input_u, input_t1, input_t2, expected):
    u = wp.array([input_u], dtype=wp.vec3, device=device, requires_grad=True)
    t1 = wp.array([input_t1], dtype=wp.vec3, device=device, requires_grad=True)
    t2 = wp.array([input_t2], dtype=wp.vec3, device=device, requires_grad=True)
    out = wp.zeros_like(u, requires_grad=True)

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


@pytest.mark.parametrize(
    "input_u, input_v, input_n, expected",
    [
        # 1. 0 Degrees (Parallel)
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0),
        # 2. +90 Degrees
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], np.pi / 2),
        # 3. +180 Degrees (Anti-Parallel)
        # ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], np.pi),
        # 4. -90 Degrees (270 Degrees)
        ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], -np.pi / 2),
        # 5. +45 Degrees
        ([1.0, 0.0, 0.0], [0.70710678, 0.70710678, 0.0], [0.0, 0.0, 1.0], np.pi / 4),
        # --- Negative Normal (Flipped Axis) ---
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], -np.pi / 2),
        # --- Arbitrary Planes (XZ, YZ) ---
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], -np.pi / 2),
    ],
)
def test_signed_angle(device, input_u, input_v, input_n, expected):
    u = wp.array([input_u], dtype=wp.vec3, device=device, requires_grad=True)
    v = wp.array([input_v], dtype=wp.vec3, device=device, requires_grad=True)
    n = wp.array([input_n], dtype=wp.vec3, device=device, requires_grad=True)
    out = wp.zeros(len(u), dtype=wp.float32, device=device, requires_grad=True)

    wp.launch(
        kernel=signed_angle_kernel,
        dim=len(u),
        inputs=[u, v, n, out],
        device=device,
    )

    result = out.numpy()
    exp = np.array([expected], dtype=np.float32)
    np.testing.assert_allclose(result, exp)

    assert wp.autograd.gradcheck(
        signed_angle_kernel,
        dim=len(u),
        inputs=[u, v, n],
        outputs=[out],
        device=device,
    )


@pytest.mark.parametrize(
    "input_u, input_v, input_theta, expected",
    [
        # 1. Identity (Theta = 0)
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, [1.0, 0.0, 0.0]),
        # 2. X-axis around Z-axis by 90 degrees -> Y-axis
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], np.pi / 2, [0.0, 1.0, 0.0]),
        # 3. Y-axis around X-axis by 90 degrees -> Z-axis
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], np.pi / 2, [0.0, 0.0, 1.0]),
        # 4. Z-axis around Y-axis by 90 degrees -> X-axis
        ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], np.pi / 2, [1.0, 0.0, 0.0]),
        # 5. X-axis around Z-axis by 180 -> -X
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], np.pi, [-1.0, 0.0, 0.0]),
        # 6. Y-axis around X-axis by 180 -> -Y
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], np.pi, [0.0, -1.0, 0.0]),
        # 7. X-axis around Z-axis by -90 degrees -> -Y-axis
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], -np.pi / 2, [0.0, -1.0, 0.0]),
        # 8. Rotating parallel to the axis (Should not change)
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], np.pi / 3, [0.0, 0.0, 1.0]),
        # 9. 45 Degree Rotation (XY Plane)
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], np.pi / 4, [0.70710678, 0.70710678, 0.0]),
    ],
)
def test_rotate_axis_angle(device, input_u, input_v, input_theta, expected):
    u = wp.array([input_u], dtype=wp.vec3, device=device, requires_grad=True)
    v = wp.array([input_v], dtype=wp.vec3, device=device, requires_grad=True)
    theta = wp.array([input_theta], dtype=wp.float32, device=device, requires_grad=True)
    out = wp.zeros_like(u, requires_grad=True)

    wp.launch(
        kernel=rotate_axis_angle_kernel,
        dim=len(u),
        inputs=[u, v, theta, out],
        device=device,
    )

    result = out.numpy()
    exp = np.array([expected], dtype=np.float32)
    np.testing.assert_allclose(result, exp, atol=1e-5)  # small non-zero

    assert wp.autograd.gradcheck(
        rotate_axis_angle_kernel,
        dim=len(u),
        inputs=[u, v, theta],
        outputs=[out],
        device=device,
    )
