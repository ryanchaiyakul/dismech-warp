import warp as wp


@wp.kernel
def tau_der(
    thetas: wp.array(dtype=float),
    betas: wp.array(dtype=float),
    ks: wp.array(dtype=float),
    E: wp.array(dtype=float),
):
    idx = wp.tid()
    thetae, thetaf = thetas[idx], thetas[idx + 1]
    beta = betas[idx]
    k = ks[idx]
    tau = thetaf - thetae + beta
    E[idx] = 0.5 * k * tau * tau


@wp.kernel
def grad_tau_der(
    nodes: wp.array(dtype=wp.vec3),
    thetas: wp.array(dtype=float),
    betas: wp.array(dtype=float),
    ks: wp.array(dtype=float),
    F: wp.array(dtype=float),
):
    idx = wp.tid()
    n0, n1, n2 = nodes[idx], nodes[idx + 1], nodes[idx + 2]
    thetae, thetaf = thetas[idx], thetas[idx + 1]
    beta = betas[idx]
    k = ks[idx]

    ee = n1 - n0
    ef = n2 - n1

    ne = wp.length(ee)
    nf = wp.length(ef)

    te = ee / ne
    tf = ef / nf

    kb = 2.0 * wp.cross(te, tf) / (1.0 + wp.dot(te, tf))
    tau = thetaf - thetae + beta
    factor = k * tau
    de = factor * 0.5 / ne * kb
    df = factor * 0.5 / nf * kb
    dthetae = factor * -1
    dthetaf = factor * 1

    base = idx * 4
    wp.atomic_add(F, base + 0, -de[0])
    wp.atomic_add(F, base + 1, -de[1])
    wp.atomic_add(F, base + 2, -de[2])
    wp.atomic_add(F, base + 3, dthetae)
    wp.atomic_add(F, base + 4, de[0] - df[0])
    wp.atomic_add(F, base + 5, de[1] - df[1])
    wp.atomic_add(F, base + 6, de[2] - df[2])
    wp.atomic_add(F, base + 7, dthetaf)
    wp.atomic_add(F, base + 8, df[0])
    wp.atomic_add(F, base + 9, df[1])
    wp.atomic_add(F, base + 10, df[2])
