import warp as wp


@wp.kernel
def epsilon_der(
    nodes: wp.array(dtype=wp.vec3),
    l_ks: wp.array(dtype=float),
    ks: wp.array(dtype=float),
    energy: wp.array(dtype=float),
):
    idx = wp.tid()
    n0, n1 = nodes[idx], nodes[idx + 1]
    k = ks[idx]

    e = n1 - n0
    e_len = wp.length(e)
    eps = e_len / l_ks[idx] - 1.0
    E = 0.5 * k * eps * eps
    wp.atomic_add(energy, 0, E)


@wp.kernel
def grad_epsilon_der(
    nodes: wp.array(dtype=wp.vec3),
    l_ks: wp.array(dtype=float),
    ks: wp.array(dtype=float),
    F: wp.array(dtype=float),
):
    idx = wp.tid()
    n0, n1 = nodes[idx], nodes[idx + 1]
    inv_lk = 1.0 / l_ks[idx]
    k = ks[idx]

    e = n1 - n0
    e_len = wp.length(e)
    eps = e_len * inv_lk - 1.0
    tangent = e / e_len
    deps = tangent * inv_lk
    dF = k * eps * deps

    # F = [-dF, dF]
    base = idx * 4  # skip a node
    wp.atomic_add(F, base + 0, -dF[0])
    wp.atomic_add(F, base + 1, -dF[1])
    wp.atomic_add(F, base + 2, -dF[2])
    wp.atomic_add(F, base + 4, dF[0])  # skip theta
    wp.atomic_add(F, base + 5, dF[1])
    wp.atomic_add(F, base + 6, dF[2])


@wp.kernel
def hess_epsilon_der(
    nodes: wp.array(dtype=wp.vec3),
    l_ks: wp.array(dtype=float),
    ks: wp.array(dtype=float),
    H: wp.array2d(dtype=float),
):
    idx = wp.tid()
    n0, n1 = nodes[idx], nodes[idx + 1]
    inv_lk = 1.0 / l_ks[idx]
    k = ks[idx]

    e = n1 - n0
    e_len = wp.length(e)
    eps = e_len * inv_lk - 1.0
    inv_e = 1.0 / e_len
    tangent = e * inv_e
    deps = tangent * inv_lk
    tt_T = wp.outer(tangent, tangent)
    I3 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    ddeps = (I3 - tt_T) * inv_lk * inv_e
    dJ = k * wp.outer(deps, deps) + k * eps * ddeps

    # H = [[dJ, -dJ], [-dJ, dJ]]
    base = idx * 4  # skip a node
    for r in range(3):
        for c in range(3):
            val = dJ[r, c]
            wp.atomic_add(H, base + r + 0, base + c + 0, val)  # [0,0]
            wp.atomic_add(H, base + r + 0, base + c + 4, -val)  # [0, 1]
            wp.atomic_add(H, base + r + 4, base + c + 0, -val)  # [1, 0]
            wp.atomic_add(H, base + r + 4, base + c + 4, val)  # [1, 1]
