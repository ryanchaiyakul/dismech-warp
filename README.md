# DiSMech-Warp

DiSMech backend for Newton.

## Getting Started

As Newton is in beta development, we install Newton from a a local Git submodule.

```bash
git clone git@github.com:ryanchaiyakul/dismech-warp.git
cd dismech-warp
git submodule update --init
uv sync
```

## Roadmap

- [ ] [Discrete elastic rods](https://www.cs.columbia.edu/cg/pdfs/143-rods.pdf).
- [ ] Rod-rod [implicit contact model](https://arxiv.org/abs/2205.10309).
- [ ] Newton SolverBase integration (Kinematic proxy capsules).
- [ ] Featherstone two-way coupling (spatial kernels).
- [ ] SPH two-way coupling (boundary particles).
- [ ] Differentiable rendering (???).
- [ ] Curriculum guidance (ghost forces).