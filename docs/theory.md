# Theory

## 1. Governing equation

An Euler-Bernoulli beam with bending stiffness EI under a distributed load q(x)
satisfies

    EI · w''''(x) = q(x)

where w is the transverse deflection. Bending moment and shear force follow from
the derivatives of w:

    M = EI · w''        V = EI · w'''

- **E** (Young's modulus) is the stiffness of the material.
- **I** (second moment of area) is the stiffness of the cross-section's shape.
  For a rectangle of width b and height h, I = b·h³/12.

## 2. The beam element (Step 1a)

Each element has two nodes, and each node has two degrees of freedom: deflection
w and rotation θ = w'. The element's degrees of freedom are ordered
[w₁, θ₁, w₂, θ₂].

With no load along its length, the element satisfies EI·w'''' = 0, so its shape
is a cubic:

    w(x) = a + b·x + c·x² + d·x³

The four end values fix the four coefficients. Each column of the stiffness
matrix is found by setting one end value to 1, holding the other three at 0,
and computing the end forces and moments needed.

**Column 1** (w₁ = 1): the shape is w = 1 − 3(x/L)² + 2(x/L)³, so

- shear force = EI·w''' = 12·EI/L³
- end moment = EI·w''(0) = 6·EI/L² (magnitude)

**Column 2** (θ₁ = 1): the shape is w = x − 2x²/L + x³/L², so

- moment at the tilted end = 4·EI/L
- moment at the far end = 2·EI/L
- shear force = 6·EI/L²

Columns 3 and 4 follow by symmetry. Factoring out EI/L³:

             [  12    6L   -12    6L  ]
    k = EI/L³[  6L   4L²   -6L   2L²  ]
             [ -12   -6L    12   -6L  ]
             [  6L   2L²   -6L   4L²  ]

The matrix is symmetric (Maxwell-Betti reciprocity) and produces zero force for
rigid-body lift and tilt.

## 3. Assembly (Step 1b)

A beam of n elements has n + 1 nodes and 2(n + 1) degrees of freedom. Element e
connects nodes e and e + 1, so its degrees of freedom map to global positions
[2e, 2e+1, 2e+2, 2e+3]. Each element matrix is added into the global matrix K
at those positions. Where elements share a node, their contributions add.

K is banded: each node couples only to its neighbours. Before supports are
applied, K is singular, because the unsupported beam can move as a rigid body.

## 4. Supports and solution (Step 1c)

The global system is

    K · u = F

For a cantilever, the clamped node has w = 0 and θ = 0. Those rows and columns
are removed, and the reduced system is solved for the free degrees of freedom.

**Validation.** For a tip load P on a cantilever of length L:

    tip deflection  δ = P·L³ / (3EI)
    tip rotation    θ = P·L² / (2EI)
    deflection      w(x) = P·x²·(3L − x) / (6EI)

The FEM result matches these exactly (to rounding) for any number of elements,
because the exact solution is itself a cubic and is contained in the element's
shape functions. For distributed loads and for vibration (Step 2), the result
converges as the number of elements increases.
