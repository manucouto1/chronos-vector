---
title: "Path Signatures"
description: "Universal trajectory descriptors from rough path theory: theory, computation, and applications"
---

## What Are Path Signatures?

Path signatures are a mathematical tool from **rough path theory** that provide a universal descriptor of the shape of a trajectory. Given a path through space, the signature captures its geometry through a sequence of iterated integrals — encoding displacement, curvature, volatility, and arbitrarily complex patterns into a single, fixed-dimensional vector.

For ChronosVector, path signatures unlock a fundamentally new capability: **trajectory similarity search**. Traditional vector kNN finds entities with similar *positions* at a given time. Signature-based kNN finds entities with similar *evolution patterns* — a query type no existing vector database supports.

---

## Mathematical Definition

Formally, the path signature of a continuous path $X: [0,T] \to \mathbb{R}^d$ is the collection of iterated integrals:

$$
S(X)^{i_1, \ldots, i_k} = \int_{0 < u_1 < \cdots < u_k < T} dX^{i_1}(u_1) \otimes \cdots \otimes dX^{i_k}(u_k)
$$

The full signature is an infinite series organized by **depth** (the number of iterated integrals):

$$
S(X) = \left(1, \; S^{(1)}(X), \; S^{(2)}(X), \; S^{(3)}(X), \; \ldots \right)
$$

where $S^{(k)}(X)$ is a tensor of rank $k$ containing all $d^k$ iterated integrals at that depth. In practice, we truncate at a finite depth, which is sufficient for most applications.

---

## Key Properties

Path signatures have remarkable mathematical properties that make them ideal for trajectory analysis:

| Property | Description | Why it matters for CVX |
|----------|-------------|----------------------|
| **Universality** | Any continuous function on paths can be approximated as a linear function of the signature | Signatures are *sufficient statistics* for trajectory classification |
| **Reparametrization invariance** | The signature depends only on the shape of the path, not the speed of traversal | Robust to different sampling rates across entities |
| **Hierarchical structure** | Each depth level captures increasingly complex geometric features | Can truncate to desired level of detail |
| **Uniqueness** | Up to tree-like equivalences, the signature uniquely determines the path | No information loss at infinite depth |
| **Multiplicativity** | $S(X\vert_{[0,T]}) = S(X\vert_{[0,s]}) \otimes S(X\vert_{[s,T]})$ (Chen's identity) | Signatures can be updated incrementally as new data arrives |

---

## Intuition by Level

The hierarchical structure of signatures provides a natural interpretation at each depth:

### Level 1: Displacement

$$
S^i(X) = X^i(T) - X^i(0)
$$

The net displacement in each dimension. This answers: *"Where did the entity end up relative to where it started?"* For a 768-dimensional embedding, level 1 is a 768-dimensional vector — the same information as subtracting the first vector from the last.

### Level 2: Area and Volatility

$$
S^{i,j}(X) = \int_0^T \bigl(X^i(t) - X^i(0)\bigr) \, dX^j(t)
$$

The **signed area** between pairs of dimensions. This encodes rotation, correlation structure, and quadratic variation (volatility). Level 2 captures information that displacement alone misses: two trajectories can have identical start and end points but very different level-2 signatures if one followed a straight line while the other oscillated wildly.

### Level 3+: Complex Patterns

Higher-order interactions capturing complex path geometry — analogous to higher moments of a probability distribution. Level 3 terms capture how the signed areas themselves evolve, encoding asymmetries, skewness of the path, and interaction effects between three or more dimensions.

---

## Practical Computation for CVX

### The Dimensionality Challenge

For $D = 768$ dimensional embeddings, computing the full signature quickly becomes intractable:

| Depth $k$ | Number of terms $d^k$ (with $d = 768$) | Feasible? |
|-----------|----------------------------------------|-----------|
| 1 | 768 | Yes |
| 2 | 589,824 | Marginal |
| 3 | 452,984,832 | No |

### Solution: PCA Reduction + Truncated Signature

CVX uses a two-stage approach:

1. **Project** the trajectory onto its principal components, reducing to $d_{\text{reduced}} = 5\text{-}10$ dimensions. This retains the dominant modes of variation while making signature computation tractable.
2. **Compute** the truncated signature at depth 2-3 on the reduced trajectory.

The resulting signature is a fixed-dimensional vector of manageable size:

| $d_{\text{reduced}}$ | Depth | Signature dimensions | Log-signature dimensions |
|----------------------|-------|---------------------|-------------------------|
| 5 | 2 | 30 | 15 |
| 5 | 3 | 155 | 35 |
| 10 | 2 | 110 | 55 |
| 10 | 3 | 1,110 | 175 |

The PCA step is computed per-entity on its trajectory matrix $X \in \mathbb{R}^{T \times D}$, and CVX reports the variance explained so users can assess the quality of the projection.

---

## Log-Signature: The Compact Alternative

The **log-signature** is a more compact representation that removes redundant terms via the Baker-Campbell-Hausdorff (BCH) formula. It lives in the free Lie algebra rather than the tensor algebra, and contains the same information in fewer dimensions.

For practical purposes: where a depth-3 signature of a 5-dimensional path has 155 components, the corresponding log-signature has only 35. This makes the log-signature the preferred representation for storage and similarity search, with the full signature available when needed for specific analyses.

CVX defaults to computing the log-signature (`use_log_signature: true` in the configuration).

---

## Time Augmentation

By default, signatures are reparametrization-invariant: a trajectory traversed quickly produces the same signature as one traversed slowly. This is often desirable (robustness to sampling rate), but sometimes the *timing* of movement matters.

Adding time as an extra dimension — $\tilde{X}(t) = (t, X(t))$ — breaks reparametrization invariance but captures speed information. CVX supports this via the `time_augmentation` configuration option. Use it when:

- You care about *when* changes happened, not just *what* changed
- Comparing trajectories with very different temporal extents
- The velocity profile (fast vs. slow change) is part of the pattern you want to match

---

## Trajectory Similarity Search

This is the most transformative capability that path signatures bring to CVX.

### Traditional vs. Signature-Based kNN

| Query type | What it finds | Example |
|-----------|---------------|---------|
| Position kNN | Entities near entity X at time $t$ | "What concepts are similar to 'AI' right now?" |
| Trajectory kNN | Entities that followed a similar path | "What concepts evolved like 'AI' did over the last 5 years?" |

Two entities can be far apart in embedding space but have identical signatures — they underwent the same *type* of evolution (same direction, speed pattern, volatility structure) in different regions of the space. Conversely, two nearby entities might have very different signatures.

### How It Works

1. Compute signatures for all entities (or a filtered subset) over a specified time window
2. Store signatures as vectors in a dedicated signature index
3. Query with a reference signature to find entities with similar trajectories
4. Return results ranked by signature distance (L2 or cosine on the signature vectors)

This enables queries that are impossible with position-based search: finding historical precedents, detecting recurring patterns, and classifying trajectory types across a corpus.

---

## References

- Lyons, T.J. (1998). *Differential Equations Driven by Rough Signals*. Revista Matematica Iberoamericana. The foundational paper on rough path theory and path signatures.
- Chevyrev, I. & Kormilitzin, A. (2016). *A Primer on the Signature Method in Machine Learning*. arXiv:1603.03788. An accessible introduction to signatures for ML practitioners.
- Kidger, P. & Lyons, T.J. (2021). *Signatory: Differentiable Computations of the Signature and Logsignature Transforms, on Both CPU and GPU*. ICLR 2021. The reference implementation for efficient signature computation.
- Arribas, I.P., Goodwin, G.M., Geddes, J.R., Lyons, T.J., & Saunders, K.E.A. (2020). *A Signature-Based Machine Learning Model for Distinguishing Bipolar Disorder and Borderline Personality Disorder*. Translational Psychiatry. Demonstrates signatures for clinical trajectory classification.
- Chen, K.T. (1958). *Integration of Paths, Geometric Invariants and a Generalized Baker-Hausdorff Formula*. Annals of Mathematics. The original work on iterated integrals and Chen's identity.
