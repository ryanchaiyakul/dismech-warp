from typing import cast
import warp as wp
import numpy as np

from .util import get_material_frame

from itertools import combinations
from newton.solvers import SolverBase
from newton import Model, ModelBuilder, State, Control, Contacts

from .util import parallel_transport, get_ref_twist

# Namespace
DER = "der"

# Copied from types.py
Vec3 = list[float] | tuple[float, float, float] | wp.vec3
"""A 3D vector represented as a list or tuple of 3 floats."""
Quat = list[float] | tuple[float, float, float, float] | wp.quat
"""A quaternion represented as a list or tuple of 4 floats (in XYZW order)."""


@wp.struct
class DERSpringConfig:
    idx: wp.int32  # parent rod
    n0: wp.int32
    n1: wp.int32
    n2: wp.int32
    e0: wp.int32
    e1: wp.int32
    voronoi_weight: float
    s0: float
    s1: float


@wp.struct
class DERRodConfig:
    EA: float
    EI1: float
    EI2: float
    GJ: float


@wp.kernel
def get_strain(
    nodes: wp.array(dtype=wp.vec3),
    thetas: wp.array(dtype=float),
    d1s: wp.array(dtype=wp.vec3),
    ts: wp.array(dtype=wp.vec3),
    betas: wp.array(dtype=float),
    spring_configs: wp.array(dtype=DERSpringConfig),
    l_ks: wp.array(dtype=float),
    strains: wp.array(dtype=float, ndim=2),
):
    idx = wp.tid()
    config = spring_configs[idx]

    # 0.0 = phantom, 1.0 = full spring
    is_full_junction = float((config.e0 != -1) and (config.e1 != -1))

    # e0 or e1 can be -1
    e0_safe = wp.max(0, config.e0)
    e1_safe = wp.max(0, config.e1)
    n0_safe = wp.max(0, config.n0)
    n1_safe = wp.max(0, config.n1)
    n2_safe = wp.max(0, config.n2)

    p0, p1, p2 = nodes[n0_safe], nodes[n1_safe], nodes[n2_safe]
    thetae, thetaf = thetas[e0_safe], thetas[e1_safe]
    l_ke, l_kf = l_ks[e0_safe], l_ks[e1_safe]
    d1e, d1f = d1s[e0_safe], d1s[e1_safe]
    te_old, tf_old = ts[e0_safe], ts[e1_safe]

    ee = p1 - p0
    ef = p2 - p1
    eps0 = (wp.length(ee) / l_ke - 1.0) * float(config.e0 != -1)
    eps1 = (wp.length(ef) / l_kf - 1.0) * float(config.e1 != -1)
    te = wp.normalize(ee)
    tf = wp.normalize(ef)
    m1e, m2e = get_material_frame(d1e, te_old, te, thetae)
    m1f, m2f = get_material_frame(d1f, tf_old, tf, thetaf)
    kb = 2.0 * wp.cross(te, tf) / (1.0 + wp.dot(te, tf) + 1e-9)
    kappa1 = 0.5 * wp.dot(kb, m2e + m2f) * is_full_junction
    kappa2 = 0.5 * wp.dot(kb, m1e + m1f) * is_full_junction
    tau = (thetaf - thetae + betas[idx]) * is_full_junction

    strains[idx, 0] = eps0
    strains[idx, 1] = eps1
    strains[idx, 2] = kappa1
    strains[idx, 3] = kappa2
    strains[idx, 4] = tau


class DERSolver(SolverBase):
    def __init__(self, model: Model):
        super().__init__(model)
        if not hasattr(model, DER):
            raise AttributeError(
                "DER custom attributes are missing from the model. "
                "Call DERSolver.register_custom_attributes() before building the model. "
                "If you called the function above, your model does not contain a DER rod. "
            )
        self.der = getattr(model, DER)

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        if not hasattr(state_in, DER):
            raise ValueError("DER custom attributes are missing from the state.")
        der_state = getattr(state_in, DER)
        nodes = state_in.particle_q
        thetas = der_state.theta
        d1s = der_state.d1
        t1s = der_state.t
        betas = der_state.beta
        spring_configs = self.der.spring_config
        l_ks = self.der.l_k
        rod_config = self.der.rod_config

        # 1. Determine number of springs
        num_springs = len(spring_configs)

        # 2. Allocate output array (num_springs x 5 strain components)
        # Using ndim=2 for [eps0, eps1, kappa1, kappa2, tau]
        strains_out = wp.zeros(shape=(num_springs, 5), dtype=float)

        # 3. Launch
        wp.launch(
            kernel=get_strain,
            dim=num_springs,
            inputs=[
                nodes,
                thetas,
                d1s,
                t1s,
                betas,
                spring_configs,
                l_ks,
            ],
            outputs=[strains_out],
        )

        # 4. Check results
        wp.synchronize()
        print("Initial Strains (first 5 springs):")
        print(strains_out.numpy()[:5])

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        if f"{DER}:theta" not in builder.custom_attributes.keys():
            # State attributes
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="theta",
                    frequency="edges",
                    assignment=Model.AttributeAssignment.STATE,
                    dtype=float,
                    namespace=DER,
                )
            )
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="d1",
                    frequency="edges",
                    assignment=Model.AttributeAssignment.STATE,
                    dtype=wp.vec3,
                    namespace=DER,
                )
            )
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="t",
                    frequency="edges",
                    assignment=Model.AttributeAssignment.STATE,
                    dtype=wp.vec3,
                    namespace=DER,
                )
            )
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="beta",
                    frequency="springs",
                    assignment=Model.AttributeAssignment.STATE,
                    dtype=float,
                    namespace=DER,
                )
            )

            # Model attributes
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="spring_config",
                    frequency="springs",
                    assignment=Model.AttributeAssignment.MODEL,
                    dtype=DERSpringConfig,
                    default=DERSpringConfig(),
                    namespace=DER,
                )
            )

            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="l_k",
                    frequency="edges",
                    assignment=Model.AttributeAssignment.MODEL,
                    dtype=float,
                    namespace=DER,
                )
            )
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="rod_config",
                    frequency="rods",
                    assignment=Model.AttributeAssignment.MODEL,
                    dtype=DERRodConfig,
                    default=DERRodConfig(),
                    namespace=DER,
                )
            )

    @classmethod
    def add_rod(
        cls,
        builder: ModelBuilder,
        positions: list[Vec3],
        quaternions: list[Quat] | None = None,
        radius: float = 0.1,
        cfg: ModelBuilder.ShapeConfig | None = None,
        stretch_stiffness: float | None = None,
        bend_stiffness: float | None = None,
        twist_stiffness: float | None = None,
        closed: bool = False,
    ) -> tuple[list[int], list[int]]:

        # Modeled after ModelBuilder().add_rod
        if cfg is None:
            cfg = builder.default_shape_cfg

        # TODO: Change constant to match behavior of default cable
        # Use same constant as ModelBuilder().add_rod
        stretch_stiffness = 1.0e9 if stretch_stiffness is None else stretch_stiffness
        bend_stiffness = 0.0 if bend_stiffness is None else bend_stiffness
        twist_stiffness = 0.0 if twist_stiffness is None else twist_stiffness

        if stretch_stiffness < 0.0 or bend_stiffness < 0.0 or twist_stiffness < 0.0:
            raise ValueError(
                "add_rod: stretch_stiffness, bend_stiffness, and twist_stiffness must be >= 0"
            )

        num_segments = len(positions) - 1

        if num_segments < 1:
            raise ValueError("add_rod: positions must contain at least 2 points")

        if quaternions is not None and len(quaternions) != num_segments:
            raise ValueError(
                f"add_rod: quaternions must have {num_segments} elements for {num_segments} segments, "
                f"got {len(quaternions)} quaternions"
            )

        edges = [(i, i + 1) for i in range(num_segments)]
        if closed:
            edges.append((num_segments, 0))

        return cls.add_rod_graph(
            builder=builder,
            node_positions=positions,
            edges=edges,
            radius=radius,
            cfg=cfg,
            stretch_stiffness=stretch_stiffness,
            bend_stiffness=bend_stiffness,
            twist_stiffness=twist_stiffness,
            quaternions=quaternions,
        )

    @classmethod
    def add_rod_graph(
        cls,
        builder: ModelBuilder,
        node_positions: list[Vec3],
        edges: list[tuple[int, int]],
        radius: float = 0.1,
        cfg: ModelBuilder.ShapeConfig | None = None,
        stretch_stiffness: float | None = None,
        bend_stiffness: float | None = None,
        twist_stiffness: float | None = None,
        quaternions: list[Quat] | None = None,
        junction_collision_filter: bool = True,
    ) -> tuple[list[int], list[int]]:
        cls.register_custom_attributes(builder)

        # Modeled after ModelBuilder().add_rod_graph
        if cfg is None:
            cfg = builder.default_shape_cfg

        # TODO: Change constant to match behavior of default cable
        # Use same constant as ModelBuilder().add_rod_graph
        stretch_stiffness = 1.0e9 if stretch_stiffness is None else stretch_stiffness
        bend_stiffness = 0.0 if bend_stiffness is None else bend_stiffness
        twist_stiffness = 0.0 if twist_stiffness is None else twist_stiffness

        if stretch_stiffness < 0.0 or bend_stiffness < 0.0 or twist_stiffness < 0.0:
            raise ValueError(
                "add_rod_graph: stretch_stiffness, bend_stiffness, and twist_stiffness must be >= 0"
            )

        num_nodes = len(node_positions)
        num_edges = len(edges)

        if num_nodes < 3:
            raise ValueError(
                "add_rod_graph: node_positions must contain at least 3 nodes"
            )
        if num_edges < 2:
            raise ValueError("add_rod_graph: edges must contain at least 2 edge")
        if quaternions is not None and len(quaternions) != num_edges:
            raise ValueError(
                f"add_rod_graph: quaternions must have {num_edges} elements for {num_edges} edges, "
                f"got {len(quaternions)} quaternions"
            )

        # TODO: convert quaternion to initial thetas

        # use numpy for CPU processing
        node_positions_np = np.asarray(node_positions, dtype=float)
        edges_idx_np = np.asarray(edges, dtype=int)
        edges_vec = (
            node_positions_np[edges_idx_np[:, 1]]
            - node_positions_np[edges_idx_np[:, 0]]
        )
        l_ks = np.linalg.norm(edges_vec, axis=1)
        tangents = edges_vec / l_ks[:, None]

        # Get mass for particles
        v_l_ks = np.zeros(num_nodes)
        np.add.at(v_l_ks, edges_idx_np[:, 0], 0.5 * l_ks)
        np.add.at(v_l_ks, edges_idx_np[:, 1], 0.5 * l_ks)
        mass = v_l_ks * wp.pi * radius**2 * cfg.density

        builder.add_particles(
            pos=node_positions,
            vel=[wp.vec3(0.0, 0.0, 0.0)] * num_nodes,
            mass=mass,
            radius=[radius] * num_nodes,
        )

        # TODO: get moment of inertia

        # Space transport (from edge to edge) to construct initial d1
        d1_list = [wp.vec3(0.0)] * num_edges

        # Initial d1
        d1_init = np.cross(tangents[0], np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(d1_init) < 1e-6:
            d1_init = np.cross(tangents[0], np.array([0.0, 0.0, -1.0]))

        # to wp.vec3 for parallel_transport compatibility
        d1 = wp.normalize(wp.vec3(d1_init))
        d1_list[0] = d1
        builder.add_custom_values(
            **{
                "der:d1": d1,
                "der:t": wp.vec3(tangents[0]),
                "der:theta": 0.0,
                "der:l_k": l_ks[0],
            }
        )

        for i in range(1, len(tangents)):
            t_prev, t_curr = wp.vec3(tangents[i - 1]), wp.vec3(tangents[i])
            d1 = parallel_transport(d1, t_prev, t_curr)
            d1 = wp.normalize(d1 - wp.dot(d1, t_curr) * t_curr)  # Gram-Schmidt
            d1_list[i] = d1
            builder.add_custom_values(
                **{
                    "der:d1": d1,
                    "der:t": t_curr,
                    "der:theta": 0.0,
                    "der:l_k": l_ks[i],
                }
            )

        # Adjacency lists
        into_edges = [[] for _ in range(num_nodes)]
        outof_edges = [[] for _ in range(num_nodes)]

        for e_idx, (u, v) in enumerate(edges):
            if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
                raise ValueError(
                    f"add_rod_graph: edge {e_idx} has invalid node indices ({u}, {v}) for {num_nodes} nodes"
                )
            if u == v:
                raise ValueError(
                    f"add_rod_graph: edge {e_idx} connects a node to itself ({u} -> {v})"
                )
            outof_edges[u].append(e_idx)
            into_edges[v].append(e_idx)

        # 0-indexed so len(.values) is the next free index
        idx = len(builder.get_custom_attributes_by_frequency("rods")[0].values)

        def add_spring(
            n0: int, n1: int, n2: int, e0: int, e1: int, w: float, s0: float, s1: float
        ):
            beta = 0.0
            # Ignore phantom springs
            if e0 != -1 and e1 != -1:
                beta = get_ref_twist(
                    wp.vec3(d1_list[e0]),
                    wp.vec3(d1_list[e1]),
                    wp.vec3(s0 * tangents[e0]),
                    wp.vec3(s1 * tangents[e1]),
                    0.0,
                )
            config = DERSpringConfig()
            config.idx = idx
            config.n0 = n0
            config.n1 = n1
            config.n2 = n2
            config.e0 = e0
            config.e1 = e1
            config.voronoi_weight = w
            config.s0 = s0
            config.s1 = s1
            builder.add_custom_values(**{"der:beta": beta, "der:spring_config": config})

        for i in range(num_nodes):
            into = into_edges[i]
            outof = outof_edges[i]
            degree = len(into) + len(outof)

            if degree == 0:
                raise ValueError(f"add_rod_graph: node {i} has no connecting edges")

            # Phantom triplet with half the weight
            if degree == 1:
                if len(into) == 1:
                    add_spring(edges[into[0]][0], i, -1, into[0], -1, 0.5, 1.0, 0.0)
                else:
                    add_spring(-1, i, edges[outof[0]][1], -1, outof[0], 0.5, 0.0, 1.0)
                continue

            w = 0.5 / (degree - 1)

            # 1. all combinations of two edges that point into the node
            if len(into) >= 2:
                for e0, e1 in combinations(into, 2):
                    add_spring(edges[e0][0], i, edges[e1][0], e0, e1, w, 1.0, -1.0)

            # 2. all combinations of two edges that point out of the node
            if len(outof) >= 2:
                for e0, e1 in combinations(outof, 2):
                    add_spring(edges[e0][1], i, edges[e1][1], e0, e1, w, -1.0, 1.0)

            # 3. all combinations of an edge into/out of the node
            if len(into) > 0 and len(outof) > 0:
                for e0 in into:
                    for e1 in outof:
                        add_spring(edges[e0][0], i, edges[e1][1], e0, e1, w, 1.0, 1.0)

        Y = 1e7
        poisson = 0.5
        EA = Y * wp.pi * radius**2
        EI1 = EI2 = Y * wp.pi * radius**4 / 4
        GJ = Y / (2 * (1 + poisson)) * wp.pi * radius**4 / 2

        config = DERRodConfig()
        config.EA = EA
        config.EI1 = EI1
        config.EI2 = EI2
        config.GJ = GJ
        builder.add_custom_values(**{"der:rod_config": config})

        # TODO: Add kinematic capsules (future)
        return [], []


@wp.kernel
def E(
    # SoA
    nodes: wp.array(dtype=wp.vec3),
    thetas: wp.array(dtype=float),
    d1s: wp.array(dtype=wp.vec3),
    t1s: wp.array(dtype=wp.vec3),
    betas: wp.array(dtype=float),
    # indexing to support non-linear networks
    l_ks: wp.array(dtype=float),  # indexed like thetas
    # Should be a single 5 length arr? How do I deal with boundary/double count of stretch
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
