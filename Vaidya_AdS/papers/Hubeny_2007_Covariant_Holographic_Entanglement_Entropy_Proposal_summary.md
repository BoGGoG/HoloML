# Hubeny, Rangamani, Takayanagi (2007) — *A Covariant Holographic Entanglement Entropy Proposal*

## One-sentence takeaway
This paper introduces the HRT prescription: in time-dependent spacetimes, holographic entanglement entropy is computed by covariant extremal surfaces, not by minimal surfaces on a preferred constant-time slice.

## What problem the paper addresses
The original Ryu-Takayanagi prescription works naturally for static spacetimes because one can choose a constant-time slice and find a minimal-area surface anchored on the boundary subregion. In Lorentzian time-dependent geometries, however, there is no preferred bulk time slice, and minimizing area on an arbitrary slice is not covariant.

The authors therefore formulate a covariant prescription suitable for time-dependent field-theory states and dynamical bulk geometries, including black-hole formation.

## Setup and method
The paper proposes that the entanglement entropy of a boundary region \(A\) is given by

\[
S_A = \frac{\mathrm{Area}(\gamma_A)}{4G_N},
\]

where \(\gamma_A\) is a codimension-two extremal surface in the Lorentzian bulk, anchored on \(\partial A\). In AdS3, this extremal surface is a spacelike geodesic.

The authors motivate the prescription through light-sheets and the covariant entropy bound. Equivalently, the relevant surface has vanishing null expansions. They compare several candidate covariant constructions and argue that the extremal-surface/light-sheet construction is the correct generalization of RT.

## Main results
The HRT prescription reduces to RT in static backgrounds. The paper checks the proposal in several examples, including AdS, BTZ, rotating BTZ, and Vaidya-AdS. The rotating BTZ example is especially important because it shows that a naive constant-time construction can fail even in a stationary but non-static spacetime; the correct geodesic is genuinely covariant.

For Vaidya-AdS collapse, the authors show how the prescription can be used to compute time-dependent entanglement entropy in a boundary state dual to black-hole formation. The resulting behavior is consistent with the expectation that entanglement entropy increases during thermalization.

## Important ideas for your project
- In Vaidya-AdS, you should use spacelike extremal geodesics, not equal-time minimal geodesics.
- Boundary endpoints may have equal boundary time, but the bulk geodesic generally moves through the time coordinate.
- A covariant geodesic solver should solve for both the radial and time profiles of the geodesic.
- The HRT surface can probe regions that are not captured by simple constant-time intuition.

## Relevance to reconstructing Vaidya-AdS from boundary entanglement data
This paper supplies the geometric rule that makes your target problem well-defined. Your ML data should be generated from HRT geodesic lengths in Vaidya-AdS. If the data were generated from non-covariant equal-time geodesics, the inverse problem would learn the wrong map.

In practical terms, for AdS3-Vaidya you need a robust solver for spacelike geodesics anchored at boundary endpoints \((t, x=\pm l/2)\). The training signal is the regularized geodesic length. The inverse model can then try to infer the collapse profile \(m(v)\) or a generalized time-dependent metric from the full family of lengths \(L_{\mathrm{reg}}(l,t)\).
