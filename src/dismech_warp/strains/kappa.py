import warp as wp


@wp.kernel
def kappa_der(
    nodes: wp.array(dtype=wp.vec3),
    m1s: wp.array(dtype=wp.vec3),
    m2s: wp.array(dtype=wp.vec3),
    ks: wp.array(dtype=float),
    E: wp.array(dtype=float),
):
    idx = wp.tid()
    n0, n1, n2 = nodes[idx], nodes[idx + 1], nodes[idx + 2]
    m1e, m2e = m1s[idx], m2s[idx]
    m1f, m2f = m1s[idx + 1], m2s[idx + 1]
    k1, k2 = ks[idx], ks[idx + 1]

    ee = n1 - n0
    ef = n2 - n1
    te = wp.normalize(ee)
    tf = wp.normalize(ef)
    inv_denom = 1.0 / (1.0 + wp.dot(te, tf))  # TODO: add eps?
    kb = 2.0 * wp.cross(te, tf) * inv_denom
    kappa1 = 0.5 * wp.dot(kb, m2e + m2f)
    kappa2 = -0.5 * wp.dot(kb, m1e + m1f)
    E[idx] = 0.5 * (k1 * kappa1 * kappa1 + k2 * kappa2 * kappa2)


@wp.kernel
def grad_kappa_der(
    nodes: wp.array(dtype=wp.vec3),
    m1s: wp.array(dtype=wp.vec3),
    m2s: wp.array(dtype=wp.vec3),
    ks: wp.array(dtype=float),
    F: wp.array(dtype=float),
):
    idx = wp.tid()
    n0, n1, n2 = nodes[idx], nodes[idx + 1], nodes[idx + 2]
    m1e, m2e = m1s[idx], m2s[idx]
    m1f, m2f = m1s[idx + 1], m2s[idx + 1]
    k1, k2 = ks[idx], ks[idx + 1]

    ee = n1 - n0
    ef = n2 - n1
    inv_ee = 1.0 / wp.length(ee)
    inv_ef = 1.0 / wp.length(ef)
    te = ee * inv_ee
    tf = ef * inv_ef

    inv_denom = 1.0 / (1.0 + wp.dot(te, tf))  # TODO: add eps?
    kb = 2.0 * wp.cross(te, tf) * inv_denom

    kappa1 = 0.5 * wp.dot(kb, m2e + m2f)
    kappa2 = -0.5 * wp.dot(kb, m1e + m1f)

    k1_factor = k1 * kappa1
    k2_factor = k2 * kappa2
    tilde_t = (te + tf) * inv_denom
    tilde_d1 = (m1e + m1f) * inv_denom
    tidle_d2 = (m2e + m2f) * inv_denom
    de = k1_factor * inv_ee * (-kappa1 * tilde_t + wp.cross(tf, tidle_d2))
    de += k2_factor * inv_ee * (-kappa2 * tilde_t - wp.cross(tf, tilde_d1))
    df = k1_factor * inv_ef * (-kappa1 * tilde_t + wp.cross(tf, tidle_d2))
    df += k2_factor * inv_ef * (-kappa2 * tilde_t - wp.cross(tf, tilde_d1))
    dthetae = k1_factor * -0.5 * wp.dot(kb, m1e)
    dthetae += k2_factor * -0.5 * wp.dot(kb, m2e)
    dthetaf = k1_factor * -0.5 * wp.dot(kb, m1f)
    dthetaf += k2_factor * -0.5 * wp.dot(kb, m2f)

    base = idx * 4  # skip a node

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
