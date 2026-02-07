import warp as wp


@wp.func
def parallel_transport(
    u1: wp.vec3, t1: wp.vec3, t2: wp.vec3, eps: float = 1e-8
) -> wp.vec3:
    """Differentiable parallel transport of u from t1 to t2.

    Args:
        u1 (wp.vec3): vector to be transported.
        t1 (wp.vec3): original tangent vector.
        t2 (wp.vec3): rotated tangent vector.
        eps (float, optional): denominator regularizaiton term near 180-singularity. Defaults to 1e-8.

    Returns:
        wp.vec3: rotated vector aligned with t2.
    """
    b = wp.cross(t1, t2)
    d = wp.dot(t1, t2)
    denom = 1.0 + d + eps
    b_cross_u = wp.cross(b, u1)
    return u1 + b_cross_u + wp.cross(b, b_cross_u) / denom


@wp.kernel
def parallel_transport_kernel(
    u1s: wp.array(dtype=wp.vec3),
    t1s: wp.array(dtype=wp.vec3),
    t2s: wp.array(dtype=wp.vec3),
    u2s: wp.array(dtype=wp.vec3),
):
    """Differentiable parallel transport of u1s to u2s from t1 to t2.

    Args:
        u1s (wp.array): vectors to be transported.
        t1s (wp.array): original tangent vectors.
        t2s (wp.array): rotated tangent vectors.
        u2s (wp.array): rotated vectors aligned with t2.
        eps (float, optional): denominator regularizaiton term near 180-singularity. Defaults to 1e-8.
    """
    tid = wp.tid()
    u2s[tid] = parallel_transport(u1s[tid], t1s[tid], t2s[tid])
