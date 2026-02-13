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
        eps (float, optional): denominator regularization term near 180-singularity. Defaults to 1e-8.

    Returns:
        wp.vec3: rotated vector aligned with t2.
    """
    b = wp.cross(t1, t2)
    d = wp.dot(t1, t2)
    denom = 1.0 + d + eps
    b_cross_u = wp.cross(b, u1)
    return u1 + b_cross_u + wp.cross(b, b_cross_u) / denom


# TODO: comments
@wp.func
def signed_angle(
    u: wp.vec3,
    v: wp.vec3,
    n: wp.vec3,
) -> wp.float32:
    w = wp.cross(u, v)
    uv = wp.dot(u, v)
    sin = wp.dot(w, n)
    return wp.atan2(sin, uv)


@wp.func
def rotate_axis_angle(
    u: wp.vec3,
    v: wp.vec3,
    theta: wp.float32,
) -> wp.vec3:
    c = wp.cos(theta)
    s = wp.sin(theta)
    return c * u + s * wp.cross(v, u) + wp.dot(v, u) * (1.0 - c) * v
