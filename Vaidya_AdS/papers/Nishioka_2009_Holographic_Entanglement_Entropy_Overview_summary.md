# Nishioka, Ryu, Takayanagi (2009) — *Holographic Entanglement Entropy: An Overview*

## One-sentence takeaway
This review gives the conceptual and formula-level background for holographic entanglement entropy, including the RT formula, standard CFT results, black-hole applications, and the covariant extension needed for time-dependent geometries.

## What problem the paper addresses
The review explains why entanglement entropy is a universal nonlocal observable in quantum many-body systems and quantum field theory, and why it is especially natural in holography. Entanglement entropy measures correlations between a region and its complement and obeys an area law in many continuum QFTs, echoing the Bekenstein-Hawking area law for black-hole entropy.

## Main content
The paper reviews the definition of entanglement entropy via the reduced density matrix,

\[
S_A = -\mathrm{tr}_A(\rho_A \log \rho_A),
\]

and key properties such as equality of \(S_A\) and \(S_B\) for pure states, subadditivity, and strong subadditivity. It then reviews standard QFT results, especially 2D CFT formulas for interval entropy at zero and finite temperature.

The holographic core is the Ryu-Takayanagi formula,

\[
S_A = \frac{\mathrm{Area}(\gamma_A)}{4G_N},
\]

where \(\gamma_A\) is the minimal bulk surface anchored on \(\partial A\). In AdS3/CFT2 this reduces to computing geodesic lengths. The review also describes higher-dimensional strip and disk regions, confinement/deconfinement applications, black-hole entropy as entanglement entropy, and the covariant HRT formulation.

## Main results and lessons
The paper is not a single new calculation but a structured overview of the field as of 2009. Its main value is to connect many pieces that your project uses separately: CFT entanglement formulas, RT geometry, minimal surfaces, black-hole entropy, and covariant holography.

For your project, the most important parts are:

- the 2D CFT benchmark formulas for vacuum and thermal interval entropy;
- the interpretation of geodesic length as entanglement entropy in AdS3/CFT2;
- the UV divergence structure and the need for regularized finite entropies/geodesic lengths;
- the role of holographic entanglement as a nonlocal probe of bulk geometry;
- the reminder that static RT must be replaced by covariant HRT in time-dependent settings.

## Relevance to reconstructing Vaidya-AdS from boundary entanglement data
This paper is the background reference that helps you avoid normalization and convention mistakes. Before training a model on Vaidya data, you should verify that your code reproduces the known AdS3/CFT2 vacuum result and the finite-temperature BTZ result. These are the natural calibration limits of Vaidya-AdS collapse.

It is also useful for deciding what quantity to feed into ML. Because entanglement entropy contains UV divergences, the inverse problem should use a regularized or finite part of the entropy/geodesic length, with consistent subtraction conventions across pure AdS, BTZ, and Vaidya.
