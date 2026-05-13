# Overall Summary of the Uploaded Holographic Entanglement Papers

## Big-picture connection to your project
Your long-term goal is to reconstruct Vaidya-AdS spacetime from boundary entanglement data using machine learning. The papers fit together into a clear pipeline: Nishioka et al. provide the general RT/HRT and CFT background; Hubeny-Rangamani-Takayanagi define the correct covariant prescription for time-dependent geometries; Abajo-Arrastia et al. apply that prescription to AdS3-Vaidya collapse and compute the entanglement evolution; Ahn et al. give the closest ML-based inverse-reconstruction methodology; Velni et al. point toward richer multi-region data through the entanglement wedge cross section.

## Per-paper summaries

### Nishioka, Ryu, Takayanagi (2009) — *Holographic Entanglement Entropy: An Overview*
This is the conceptual background review. It introduces entanglement entropy, its QFT properties, the RT formula, geodesic/minimal-surface calculations in AdS/CFT, UV divergences, CFT2 benchmark formulas, and applications to black holes and phase transitions. For your project, it is most useful for checking conventions and validating pure AdS and BTZ limits.

### Hubeny, Rangamani, Takayanagi (2007) — *A Covariant Holographic Entanglement Entropy Proposal*
This paper gives the HRT prescription, replacing static minimal surfaces by Lorentzian extremal surfaces. It is essential for Vaidya-AdS because the bulk is time dependent and no preferred constant-time bulk slice exists. For AdS3-Vaidya, this means your data must come from spacelike extremal geodesics, not naive equal-time curves.

### Abajo-Arrastia, Aparício, López (2010) — *Holographic Evolution of Entanglement Entropy*
This is the central Vaidya-AdS dynamics paper. It studies spacelike geodesics in AdS3-Vaidya and computes the time evolution of interval entanglement entropy after a quench-like perturbation. It finds ballistic entanglement spreading and local equilibration, and emphasizes that relevant geodesics can probe behind apparent/event horizons. This is the most direct target for your synthetic data generation.

### Ahn, Jeong, Kim, Yun (2025) — *Holographic Reconstruction of Black Hole Spacetime: Machine Learning and Entanglement Entropy*
This is the main ML template. It reconstructs continuous static bulk metric functions from boundary entanglement entropy data using neural ODEs and Monte-Carlo integration. The paper shows how to place the holographic forward map inside the training loop. Your Vaidya project is a time-dependent extension: replace the RT integral map by an HRT geodesic solver and reconstruct the Vaidya mass profile or a more general dynamical metric.

### Babaei Velni, Mohammadi Mozaffar, Vahidinia (2020) — *Evolution of Entanglement Wedge Cross Section Following a Global Quench*
This paper studies a more advanced observable, the entanglement wedge cross section, in Vaidya geometries. It finds early quadratic growth, intermediate linear growth, and late saturation for thermal quenches, with logarithmic growth replacing linear growth in extremal electromagnetic quenches. For your project, this is a future extension: multi-region entanglement observables may help resolve degeneracies in metric reconstruction.

## Suggested project roadmap implied by these papers

1. **Validate the forward geodesic solver.** Reproduce pure AdS and BTZ entanglement/geodesic-length formulas using the conventions from the overview paper.
2. **Use HRT consistently.** For Vaidya-AdS, solve covariant spacelike geodesics with both radial and time profiles, following the logic of HRT.
3. **Generate synthetic Vaidya data.** Build datasets \((l,t) \mapsto L_{\mathrm{reg}}(l,t)\) for controlled mass profiles \(m(v)\).
4. **Start with a constrained inverse problem.** Train a model to recover a small set of Vaidya mass-profile parameters before trying a fully flexible neural \(m(v)\).
5. **Borrow Ahn et al.’s training architecture.** Parameterize the unknown geometry with continuous neural functions and put the holographic forward calculation inside the loss.
6. **Add richer observables later.** If single-interval entropy is insufficient, consider mutual information or EWCS-type observables as additional boundary data.

## Most important immediate conclusion
Do not start with a fully general spacetime reconstruction problem. First reconstruct the Vaidya mass profile from synthetic HRT geodesic-length data in AdS3-Vaidya, after verifying pure AdS and BTZ limits. That gives a controlled proof of concept before attempting general dynamical metric reconstruction.
