import warp as wp


@wp.func
def epsilon(n0: wp.vec3, n1: wp.vec3, l_k: float) -> float:
    e_len = wp.length(n1 - n0)
    return e_len / l_k - 1.0


@wp.func
def kappa(
    n0: wp.vec3,
    n1: wp.vec3,
    n2: wp.vec3,
    m1e: wp.vec3,
    m2e: wp.vec3,
    m1f: wp.vec3,
    m2f: wp.vec3,
) -> wp.vec2:
    e0 = n1 - n0
    e1 = n2 - n1
    t0 = wp.normalize(e0)
    t1 = wp.normalize(e1)

    # curvature binominal (2 * tan(phi/2))
    c = wp.cross(t0, t1)
    d = wp.dot(t0, t1)
    denom = 1.0 + d
    kb = 2.0 * c / denom

    # project onto m1/m2
    kappa1 = 0.5 * wp.dot(kb, m2e + m2f)
    kappa2 = -0.5 * wp.dot(kb, m1e + m1f)
    return kappa1, kappa2


@wp.func
def tau(theta_e: float, theta_f: float, ref_twist: float) -> float:
    return theta_f - theta_e + ref_twist
