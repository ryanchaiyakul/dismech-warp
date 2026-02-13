import symengine as sp


# symengine lacks norm
def norm(u):
    return sp.sqrt(u.dot(u))


def parallel_transport(u, t1, t2):
    b = t1.cross(t2)
    d = t1.dot(t2)
    denom = 1.0 + d
    b_cross_u = b.cross(u)
    return u + b_cross_u + b.cross(b_cross_u) / denom


def signed_angle(u, v, n):
    w = u.cross(v)
    return sp.atan2(w.dot(n), u.dot(v))


def rotate_axis_angle(u, v, theta):
    c = sp.cos(theta)
    s = sp.sin(theta)
    return c * u + s * v.cross(u) + v.dot(u) * (1.0 - c) * v


def material_frame(d1, d2, theta):
    c = sp.cos(theta)
    s = sp.sin(theta)
    return c * d1 + s * d2, -s * d1 + c * d2


# Dofs: [x0, y0, z0, theta0, x1, y1, z1, theta1, x2, y2, z2]
x0 = sp.Matrix([sp.Symbol(f"x_{i}_0") for i in range(3)])
x1 = sp.Matrix([sp.Symbol(f"x_{i}_1") for i in range(3)])
x2 = sp.Matrix([sp.Symbol(f"x_{i}_2") for i in range(3)])
theta = sp.Matrix([sp.Symbol("theta_0"), sp.Symbol("theta_1")])

# Parameters
l_k = sp.Matrix([sp.Symbol("len_0"), sp.Symbol("len_1")])
t0_old = sp.Matrix([sp.Symbol(f"t0_old_{i}") for i in range(3)])
t1_old = sp.Matrix([sp.Symbol(f"t1_old_{i}") for i in range(3)])
d10_old = sp.Matrix([sp.Symbol(f"d10_old_{i}") for i in range(3)])
d11_old = sp.Matrix([sp.Symbol(f"d11_old_{i}") for i in range(3)])
ref_twist_old = sp.Symbol("beta_old")

e0 = x1 - x0
e1 = x2 - x1
n_e0 = norm(e0)
n_e1 = norm(e1)
t0 = e0 / n_e0
t1 = e1 / n_e1

# Transport
d10_new = parallel_transport(d10_old, t0_old, t0)
d11_new = parallel_transport(d11_old, t1_old, t1)
d20_new = t0.cross(d10_new)
d21_new = t1.cross(d11_new)

# Material Frames
m1e, m2e = material_frame(d10_new, d20_new, theta[0])
m1f, m2f = material_frame(d11_new, d21_new, theta[1])

# Stretch (Epsilon)
eps0 = n_e0 / l_k[0] - 1.0
eps1 = n_e1 / l_k[1] - 1.0

# Curvature (Kappa)
kb = (2.0 * t0.cross(t1)) / (1.0 + t0.dot(t1))
kappa1 = 0.5 * kb.dot(m2e + m2f)
kappa2 = -0.5 * kb.dot(m1e + m1f)

# Twist (Tau)
d1_transport = parallel_transport(d10_new, t0, t1)
d1_pred = rotate_axis_angle(d1_transport, t1, ref_twist_old)
delta_phi = signed_angle(d1_pred, d11_new, t1)
tau = theta[1] - theta[0] + ref_twist_old + delta_phi

# DER Energy (Example/base)
K1, K2, K3, K4 = sp.symbols("K_epsilon K_kappa1 K_kappa2 K_tau")
E = 0.5 * (K1 * eps0**2 + K1 * eps1**2 + K2 * kappa1**2 + K3 * kappa2**2 + K4 * tau**2)

# Derive analytical gradients/hessians
dofs = list(x0) + [theta[0]] + list(x1) + [theta[1]] + list(x2)
strains = [eps0, eps1, kappa1, kappa2, tau]

a = sp.diff(E, dofs[0])