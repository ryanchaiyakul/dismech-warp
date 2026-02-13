# Definitions

Let subscripts denote nodal or stencil and superscripts denote edge quantites. For an edge $e^i$, let

- $\mathbf{t}^i$ be the tangent vector.
- $\{\mathbf{d}_1^i, \mathbf{d}_2^i, \mathbf{t}^i\}$ be the orthonormal reference frame.
- $\{\mathbf{m}_1^i, \mathbf{m}_2^i, \mathbf{t}^i\}$ be the orthonormal material frame and the rotation of reference frame by a signed angle $\theta^i$ along $\mathbf{t}^i$.

# Formulation

For a 11 node stencil, let

- $\mathbf{q}=[x_0, y_0, z_0, \theta^0, x_1, y_1, z_1, \theta^1, x_2, y_2, z_2]^T$ be the state vector.
- $\mathbf{aux}=[\mathbf{t}^0_{\textrm{old}}, \mathbf{t}^1_{\textrm{old}}, \mathbf{d}_{1,\textrm{old}}^0, \mathbf{d}_{1,\textrm{old}}^1, \beta_\textrm{old}]^T$ be the auxiliary vector.
- $\boldsymbol{\epsilon}=[\epsilon^0, \epsilon^1, \kappa_1, \kappa_2, \tau]^T$ be the strain vector.
- $E=f(\boldsymbol{\epsilon}\boldsymbol{\epsilon}^T)$ be the energy.

## Frame Reconstruction

First, we calcualte the $\mathbf{d}_{1,\textrm{new}}^i$ from $\mathbf{aux}$

$$
\mathbf{d}_{1,\textrm{new}}^i=\textrm{ParallelTransport}(\mathbf{d}_{1,\textrm{old}}^i,\mathbf{t}_{\textrm{old}}^i, \mathbf{t}_{\textrm{new}}^i)
$$

With $\mathbf{t}_{\textrm{new}}^i$ and $\mathbf{d}_{1,\textrm{new}}^i$ we construct the **new** orthonormal reference frame $\{\mathbf{d}_1^i, \mathbf{d}_2^i, \mathbf{t}^i\}$ as

$$
\mathbf{d}_{2,\textrm{new}}^i=\mathbf{t}^i_{\textrm{new}}\times \mathbf{d}_{1,\textrm{new}}^i
$$

With $\{\mathbf{d}_1^i, \mathbf{d}_2^i, \mathbf{t}^i\}$ and $\theta^i$, we compute the material frame $\{\mathbf{m}_1^i, \mathbf{m}_2^i, \mathbf{t}^i\}$

$$
\begin{align*}
m_1^i&=\cos(\theta^i)\mathbf{d}_1^i+\sin(\theta^i)\mathbf{d}_2^i \\
m_2^i&=-\sin(\theta^i)\mathbf{d}_1^i+\cos(\theta^i)\mathbf{d}_2^i
\end{align*}
$$

## Strains

### Stretching Strain 

The scalar stretching strain is the ratio of elongation / compression of an edge:

$$
\epsilon^i=\frac{\|\mathbf{e}^i\|}{\|\mathbf{\bar e}^i\|}-1
$$

### Bending Strains

The discrete curvature binormal $(\kappa\mathbf{b})$ at node $i$ represents the integrated curvature between edges $\mathbf{e}^0$ and $\mathbf{e}^1$:

$$
(\kappa\mathbf{b})=\frac{2(\mathbf{t^0}\times \mathbf{t^1})}{1+\mathbf{t^0}\cdot \mathbf{t^1}}
$$

The scalar bending strains are the projections of this binormal onto the averaged material directors:

$$
\begin{align*}
\kappa_1&=\frac{1}{2}(\kappa\mathbf{b})\cdot(\mathbf{m}_2^0+\mathbf{m}_2^1) \\
\kappa_2&=-\frac{1}{2}(\kappa\mathbf{b})\cdot(\mathbf{m}_1^0+\mathbf{m}_1^1) 
\end{align*}
$$

### Twisting Strain

The twisting strain $\tau$ accounts for the relative rotation of material frames plus the geometric holonomy:

$$
\tau=\theta^1-\theta^0+\beta
$$

where $\beta$ is the angle of parallel transport between reference frames. For $\mathbf{t}^0 \cdot \mathbf{t}^1 > -1$:

$$\beta = \operatorname{atan2}\left( (\mathbf{d}_{1,\textrm{new}}^0 \times \mathbf{d}_{1,\textrm{new}}^1) \cdot \mathbf{\bar{t}}, \,\, \mathbf{d}_{1,\textrm{new}}^0 \cdot \mathbf{d}_{1,\textrm{new}}^1 \right), \quad \mathbf{\bar{t}} = \frac{\mathbf{t}^0 \times \mathbf{t}^1}{\|\mathbf{t}^0 \times \mathbf{t}^1\|}$$

## Energy

To calculate $\frac{dE}{d\mathbf{q}}$ and $\frac{d^2E}{d\mathbf{q}^2}$, we can apply the chain rule:

$$
\begin{align*}
E&=f(\boldsymbol{\epsilon}\boldsymbol{\epsilon}^T) \\
\frac{dE}{d\mathbf{q}}&=\frac{df}
\end{align*}
$$
