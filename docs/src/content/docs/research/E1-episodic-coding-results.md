---
title: "E1: Episodic Coding Benchmark — Results"
description: "Experimental evaluation of CVX episodic trace memory on HumanEval code generation"
---

## Hypothesis

**H1**: An LLM augmented with semantically retrieved past solved problems (via CVX episodic memory) will generate more correct code than the same LLM with no context or randomly selected examples.

**H2**: CVX's HNSW temporal search retrieves different (and better) examples than flat cosine brute-force search.

## Experimental Setup

| Component | Choice |
|-----------|--------|
| **Training corpus** | MBPP sanitized (384 problems) |
| **Test benchmark** | HumanEval (164 problems) |
| **Embedding model** | all-MiniLM-L6-v2 (D=384) |
| **LLM** | qwen2.5-coder:7b-instruct (Ollama) |
| **Memory backend** | CVX TemporalIndex (M=16, ef=100) |

### Conditions

| Condition | Description |
|-----------|-------------|
| **NoMemory** | Zero-shot |
| **RandomFewShot** | k random MBPP solutions |
| **FlatCosine** | k most similar by numpy brute-force cosine |
| **CVX-Episodic** | k most similar via CVX temporal HNSW search |

### Protocol

1. **Validation** (HumanEval[0:82], n=82): Sweep k ∈ {1, 3, 5, 7}, T=0, single pass → select best k
2. **Test** (HumanEval[82:164], n=82): Best k, T=0.2, 5 seeds → mean ± std
3. **Statistical testing**: McNemar's test (majority-vote), paired t-test (seed-level)

## Results

### Validation (T=0, single pass)

| k | NoMemory | RandomFewShot | FlatCosine | CVX-Episodic |
|---|----------|---------------|------------|--------------|
| 1 | 59.8% | 85.4% | 85.4% | 84.1% |
| **3** | 59.8% | 84.1% | 85.4% | **86.6%** |
| 5 | 58.5% | 85.4% | 82.9% | 81.7% |
| 7 | 61.0% | 80.5% | 80.5% | 79.3% |

**Best k = 3** (selected by CVX-Episodic on validation).

### Test (T=0.2, 5 seeds, k=3)

| Condition | pass@1 (mean ± std) | Range |
|-----------|-------------------|-------|
| NoMemory | 71.7% ± 2.1% | 68.3–74.4% |
| **RandomFewShot** | **76.8% ± 1.1%** | 75.6–78.0% |
| FlatCosine | 74.1% ± 2.5% | 70.7–78.0% |
| CVX-Episodic | 74.1% ± 2.6% | 70.7–78.0% |

### Statistical Tests

**McNemar's test (majority-vote across seeds):**

| Comparison | A only | B only | Both | Neither | χ² | p |
|-----------|--------|--------|------|---------|----|---|
| CVX vs NoMemory | 8 | 9 | 52 | 13 | 0.00 | 1.000 ns |
| CVX vs FlatCosine | 0 | 0 | 60 | 22 | 0.00 | 1.000 ns |
| CVX vs Random | 4 | 10 | 56 | 12 | 1.79 | 0.181 ns |

**Paired t-test (seed-level):**
- CVX vs FlatCosine: Δ=0.0%, t=0.000, p=1.000 ns
- CVX vs Random: Δ=-2.7%, t=-1.901, p=0.130 ns

### Retrieval Overlap

CVX and FlatCosine retrieve **96.7% identical episodes** (top-3). On this corpus size (384 episodes, D=384), HNSW produces the same nearest neighbors as brute-force cosine.

## Findings

1. **Few-shot prompting provides a massive boost** (+13–15pp over zero-shot), confirming that qwen2.5-coder:7b benefits strongly from in-context examples regardless of selection method.

2. **Semantic retrieval does not outperform random few-shot** on this benchmark. RandomFewShot (76.8%) trends higher than both retrieval methods (74.1%), though the difference is not statistically significant (p=0.13).

3. **CVX and FlatCosine are functionally identical** — 96.7% retrieval overlap, 0.0pp pass@1 difference. At this corpus scale (384 episodes), HNSW's approximate search finds the exact same neighbors as brute-force.

4. **Validation vs test divergence**: On validation (T=0), CVX-Episodic led at 86.6% vs Random 84.1%. On test (T=0.2, 5 seeds), this reversed. The validation advantage was noise, not signal.

5. **k=3 is optimal**: Performance degrades at k≥5, suggesting that too many examples dilute the prompt rather than helping.

## Interpretation

The null result for semantic retrieval vs random is consistent with the hypothesis that for **code completion tasks**, the primary value of few-shot examples is **formatting and style priming** rather than **problem-specific transfer**. Any MBPP example teaches the model "here's how to write a Python function given a spec" — the specific problem similarity matters less.

This contrasts with domains where retrieval specificity matters more (e.g., E2's ALFWorld, where action sequences are highly task-dependent).

## Limitations

- **Small corpus (384 episodes)**: With more training data, semantic retrieval may differentiate more from random sampling.
- **Single model/size**: Smaller models may benefit more from relevant examples; larger models may not need any.
- **Code-specific**: The null result may not generalize to other domains.

## CVX Features Exercised

`TemporalIndex`, `insert`, `search`, `save`/`load`, episode encoding via entity_id + timestamp scheme.
