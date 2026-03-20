---
title: "Open Research Questions"
description: "Unsolved problems and active research directions for ChronosVector"
---

## Validated Findings

These questions from the original research agenda have been partially or fully answered through experimentation:

1. **Embedding anisotropy degrades all cosine-based operations.** Confirmed empirically: MentalRoBERTa embeddings have pairwise cosine similarity ~0.96. Mean-centering amplifies discriminative signal 30x (gap from 0.011 to 0.33). This is consistent with Ethayarajh (2019), Su et al. (2021). Native centering is planned (RFC-012 Part B).

2. **HNSW hierarchy produces semantically meaningful clusters.** Confirmed: Level 2 regions show depression ratios from 0.15 to 0.85, demonstrating unsupervised specialization without labels. `region_assignments()` enables efficient O(N) single-pass exploration.

3. **Temporal vector databases can serve as episodic memory for AI agents.** Confirmed with caveats: E1-E4 experiments show consistent improvement (E3: 3.3%→20% ALFWorld completion), but the effect is moderate. Key gaps identified: no outcome awareness, no causal continuation in Python API, no context filtering.

4. **Anchor projection creates interpretable clinical features.** Confirmed: DSM-5 anchor projections achieve F1=0.744 (vs 0.600 baseline) on eRisk with proper temporal split. Anchor-based features are both more interpretable and more discriminative than PCA features.

---

## Open Questions — Temporal Index

1. **Optimal semantic-temporal distance combination.** The linear blend $\alpha \cdot d_{sem} + (1-\alpha) \cdot d_{time}$ has a scale mismatch: cosine $\in [0,2]$ vs temporal $\in [0,1]$. With $\alpha=0.5$, semantic has 2x effective weight. Normalization or learned combination needed.

2. **Region stability across insertions.** HNSW hubs are emergent — reinserting data changes which nodes become hubs. This makes region-based analytics non-deterministic. Does this matter in practice? Can we stabilize regions without breaking HNSW properties?

3. **Parallel HNSW construction.** Sequential `bulk_insert` takes ~30min for 1.3M x D=768. Two-phase parallel construction with rayon (sequential allocation + parallel neighbor connection) should give 4-6x speedup. See RFC-012 Part A.

4. **Snapshot versioning.** Current postcard serialization has no version field. Struct changes silently break deserialization. Needs `version: u32` with migration support.

## Open Questions — Agent Memory

5. **How to consolidate episodic memory at scale?** With millions of episodes, retrieval degrades. Biological memory consolidates repeated experiences into prototypes. What's the right consolidation algorithm? Region-based clustering (via `region_assignments`) is a candidate, but episode integrity (can't remove step 3 of a 5-step sequence) adds complexity.

6. **Outcome-weighted retrieval.** Should reward be a filter (hard cutoff) or a scoring factor (soft weight)? Hard cutoff loses potentially informative failures. Soft weighting risks retrieving many failures with high similarity scores.

7. **Multi-space context filtering.** An agent needs "similar state + same goal". Should this be (a) metadata-indexed pre-filtering, (b) multi-space indexing (separate HNSW per dimension), or (c) concatenated embeddings? Each has different performance/flexibility trade-offs.

## Open Questions — Auxiliary Structures

8. **Do agent memory needs require structures beyond HNSW?** Causal relationships, conditional dependencies, and taxonomic knowledge are fundamentally non-vectorizable. Knowledge graphs, Bayesian networks, and causal DAGs could complement HNSW. But when does the added complexity justify itself? What's the minimum viable integration? See RFC-012 Part D for initial analysis.

9. **Bayesian networks over region transitions.** HNSW regions define a discrete state space. Region transitions over time form a Markov chain. Can we learn $P(\text{success} | \text{region}, \text{context})$ as a lightweight Bayesian network? This would enable decision-theoretic retrieval: not just "what's similar" but "what's likely to succeed."

10. **Knowledge graphs as structured metadata.** Entity types and relations (tool requires-skill, action is-a manipulation) could be encoded as indexed metadata leveraging Gap 3's infrastructure. But is a flat key-value index sufficient, or do we need graph traversal (multi-hop queries)?

## Open Questions — Stochastic Analytics

11. **Optimal path signature depth.** Depth 2 captures drift and volatility ($K + K^2$ features). Depth 3+ captures higher-order interactions but grows exponentially. For D=768, direct computation is intractable — PCA reduction to 5-10 dims is required. How much information is lost?

12. **Neural CDEs vs Neural SDEs.** Neural CDEs (Kidger et al., 2020) handle irregular observations better. Neural SDEs capture stochastic dynamics. Which is more appropriate for embedding trajectories? The current Neural ODE (TorchScript) ignores both.

13. **Mean reversion in embedding spaces.** If concept embeddings mean-revert, Ornstein-Uhlenbeck is the right model. If not, random walk or trending models are better. The anisotropy findings suggest the answer may depend on whether embeddings are centered.

## Open Questions — Implementation

14. **`TemporalIndexAccess` trait is too large.** 12 methods with empty defaults violates Interface Segregation. Should split into `TemporalSearch`, `TrajectoryAccess`, `RegionAccess`. But this is a breaking internal change — when is the right time?

15. **Python API bypasses query engine.** `cvx-python` calls `cvx-index` and `cvx-analytics` directly, not through `cvx-query`. The query engine has 15 query types; Python exposes ~10. Should Python route through the query engine for consistency?

16. **Whitening vs centering.** Full whitening (centering + rotation by inverse covariance) requires storing a D x D matrix (2.4M floats for D=768). Su et al. (2021) report only 2-5% STS improvement over centering alone. Is it worth the complexity for CVX-specific tasks (anchor projection, drift, regions)?
