# Abajo-Arrastia, Aparício, López (2010) — *Holographic Evolution of Entanglement Entropy*

## One-sentence takeaway
This paper is the core Vaidya-AdS reference for your project: it shows how boundary interval entanglement entropy in a 2D CFT can be computed from spacelike geodesic lengths in a dynamical AdS3-Vaidya collapse geometry, and how that entropy evolves after a global perturbation.

## What problem the paper addresses
The authors study far-from-equilibrium dynamics in a holographic 2D CFT using a Vaidya geometry, i.e. gravitational collapse of null dust forming a BTZ black hole. The boundary process is analogous to a global quantum quench: the CFT starts in the vacuum, is driven into an excited state, and then locally equilibrates. Instead of using only near-boundary observables such as the stress tensor, the paper probes the bulk through entanglement entropy, which depends on extended spacelike geodesics anchored at the AdS boundary.

## Setup and method
The relevant bulk geometry is AdS3-Vaidya in ingoing Eddington-Finkelstein-type coordinates. The time-dependent mass profile interpolates from pure AdS to a black hole, representing a collapsing shell. For a boundary interval of length \(l\) at boundary time \(t\), the holographic entanglement entropy is obtained from the regularized length of the bulk spacelike geodesic connecting the two boundary endpoints.

The paper numerically studies these geodesics and tracks whether they cross the apparent horizon or event horizon during the collapse. In 2+1 bulk dimensions, extremal surfaces are geodesics, so this paper is directly relevant to your current numerical geodesic integration work.

## Main results
The paper recovers a central result known from 2D quantum quenches: entanglement spreads with maximal velocity, so local equilibration occurs, but global equilibration does not occur in the same sense. For intervals smaller than the causal scale, the entropy approaches the thermal value; for very large intervals, the entropy retains information about the initial long-range entanglement pattern.

A key result is that geodesics relevant for entanglement entropy are not blocked by the apparent horizon. Some geodesics probe behind the apparent and event horizons, and the authors argue that reconstructing the boundary entanglement evolution can require bulk information from regions that are not visible in a naive horizon-limited picture. Early in the evolution, the apparent horizon has little effect on the entanglement entropy, although later it strongly influences geodesic shapes and lengths.

The Vaidya model differs from the standard Calabrese-Cardy quench because the perturbation does not generate only short-range entangled quasiparticles. Instead, the initial state retains long-range correlations close to the CFT vacuum scaling. Nevertheless, the same broad conclusion about ballistic entanglement spreading in 2D is recovered.

## Important formulas and concepts for your project
- Boundary entanglement entropy in AdS3/CFT2 is computed from regularized spacelike geodesic length.
- The vacuum CFT interval entropy scales as \(S(l) \sim \frac{c}{3}\log(l/\epsilon)\).
- The thermal CFT interval entropy scales as \(S_\beta(l) \sim \frac{c}{3}\log\left[\frac{\beta}{\pi\epsilon}\sinh(\pi l/\beta)\right]\).
- The Vaidya geometry provides a controlled time-dependent interpolation between pure AdS and BTZ.
- Horizon crossing by geodesics is not a pathology; it is part of the HRT prescription in a time-dependent geometry.

## Relevance to reconstructing Vaidya-AdS from boundary entanglement data
This is the primary target paper for data generation. Your ML model should eventually learn an inverse map from \(S(l,t)\), or equivalently regularized geodesic lengths as a function of interval size and boundary time, back to features of the time-dependent Vaidya metric such as the mass profile \(m(v)\). The paper also warns that entanglement data can encode information from behind apparent horizons, so the ML reconstruction should not be restricted to exterior-horizon data only.

For your current stage, the most useful subproblem is: generate reliable pairs

\[
(l,t) \longmapsto L_{\mathrm{reg}}(l,t)
\]

from AdS3-Vaidya geodesics, then verify that late-time results converge to BTZ and early-time results converge to pure AdS. This gives a clean sanity check before attempting full inverse reconstruction.
