import warp as wp


@wp.func
def parallel_transport(
    u: wp.vec3,
    t0: wp.vec3,
    t1: wp.vec3,
    eps: float = 1e-8,
) -> wp.vec3:
    """Differentiable parallel transport of u rotated exactly as t1 rotates to t2.

    Args:
        u1 (wp.vec3): vector to be transported.
        t1 (wp.vec3): original tangent vector.
        t2 (wp.vec3): rotated tangent vector.
        eps (float, optional): denominator regularization term near 180-singularity. Defaults to 1e-8.

    Returns:
        wp.vec3: rotated vector aligned with t2.
    """
    b = wp.cross(t0, t1)
    d = wp.dot(t0, t1)
    denom = 1.0 + d + eps
    b_cross_u = wp.cross(b, u)
    return u + b_cross_u + wp.cross(b, b_cross_u) / denom


# TODO: comments
@wp.func
def signed_angle(
    u: wp.vec3,
    v: wp.vec3,
    n: wp.vec3,
) -> float:
    """Differentiable signed angle from u to v measured around the normal axis n.

    Args:
        u (wp.vec3): Original vector.
        v (wp.vec3): Rotated vector.
        n (wp.vec3): Normal axis.

    Returns:
        float: signed angle from u to v.
    """
    w = wp.cross(u, v)
    uv = wp.dot(u, v)
    sin = wp.dot(w, n)
    return wp.atan2(sin, uv)


@wp.func
def rotate_axis_angle(
    u: wp.vec3,
    v: wp.vec3,
    theta: float,
) -> wp.vec3:
    """Differentiable rotation of u around axis v by angle θ.

    Args:
        u (wp.vec3): Vector to be rotated.
        v (wp.vec3): Rotation axis.
        theta (float): Angle of rotation (radians).

    Returns:
        wp.vec3: Rotated vector of u around v.
    """
    c = wp.cos(theta)
    s = wp.sin(theta)
    return c * u + s * wp.cross(v, u) + wp.dot(v, u) * (1.0 - c) * v


@wp.func
def get_material_frame(
    d1_old: wp.vec3, t_old: wp.vec3, t_new: wp.vec3, theta: float
) -> tuple[wp.vec3, wp.vec3]:
    """Get material frame `{m1, m2, t}`: reference frame `{d1, d2, t}` rotated by edge twist θ.

    Args:
        d1_old (wp.vec3): Original d1 vector in reference frame `{d1, d2, t}`.
        t_old (wp.vec3): Original tangent vector.
        t_new (wp.vec3): New tangent vector.
        theta (float): Edge twist (radians).

    Returns:
        tuple[wp.vec3, wp.vec3]: `{m1, m2}`.
    """
    d1_new = parallel_transport(d1_old, t_old, t_new)
    d2_new = wp.cross(t_new, d1_new)
    c = wp.cos(theta)
    s = wp.sin(theta)
    m1 = c * d1_new + s * d2_new
    m2 = -s * d1_new + c * d2_new
    return m1, m2
