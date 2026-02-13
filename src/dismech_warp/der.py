from typing import cast
import warp as wp

from .util import parallel_transport


@wp.func
def get_material_frame(
    d1: wp.vec3, t1: wp.vec3, t1_new: wp.vec3, theta: float
) -> tuple[wp.vec3, wp.vec3]:
    d1_new = parallel_transport(d1, t1, t1_new)
    d2_new = wp.cross(t1_new, d1_new)
    c = wp.cos(theta)
    s = wp.sin(theta)
    m1 = c * d1_new + s * d2_new
    m2 = -s * d1_new + c * d2_new
    return m1, m2


@wp.kernel
def E(
    nodes: wp.array(dtype=wp.vec3),
    thetas: wp.array(dtype=float),
    d1s: wp.array(dtype=wp.vec3),
    t1s: wp.array(dtype=wp.vec3),
    betas: wp.array(dtype=float),
    # TODO: Make these constants
    l_ks: wp.array(dtype=float),
    ks: wp.array(dtype=float),
    E: wp.array(dtype=float),
    # bar_strain: wp.array(dtype=float),
):
    idx = cast(int, wp.tid())
    n0, n1, n2 = nodes[idx], nodes[idx + 1], nodes[idx + 2]
    thetae, thetaf = thetas[idx], thetas[idx + 1]
    d1e, d1f = d1s[idx], d1s[idx + 1]
    t1e, t1f = t1s[idx], t1s[idx + 1]
    l_ke, l_kf = l_ks[idx], l_ks[idx + 1]
    beta = betas[idx]

    ee = n1 - n0
    ef = n2 - n1
    eps0 = wp.length(ee) / l_ke - 1.0
    eps1 = wp.length(ef) / l_kf - 1.0
    te = wp.normalize(ee)
    tf = wp.normalize(ef)
    m1e, m2e = get_material_frame(d1e, t1e, te, thetae)
    m1f, m2f = get_material_frame(d1f, t1f, tf, thetaf)
    kb = 2.0 * wp.cross(te, tf) / (1.0 + wp.dot(te, tf))
    kappa1 = 0.5 * wp.dot(kb, m2e + m2f)
    kappa2 = 0.5 * wp.dot(kb, m1e + m1f)
    tau = thetaf - thetae + beta

    base = idx * 4
    k1, k2, k3, k4 = ks[base], ks[base + 1], ks[base + 2], ks[base + 3]
    energy = 0.5 * (
        k1 * eps0 * eps0
        + k1 * eps1 * eps1
        + k2 * kappa1 * kappa1
        + k3 * kappa2 * kappa2
        + k4 * tau * tau
    )
    wp.atomic_add(E, 0, energy)


@wp.kernel
def grad_E():
    pass


@wp.kernel
def hess_E():
    pass
