---
title: "E1: Episodic Coding Benchmark — Results"
description: "Experimental evaluation of CVX episodic trace memory on HumanEval code generation"
---

## Hypothesis

**H1**: Semantic retrieval outperforms random few-shot for code generation.

**H2**: CVX-Causal (match any step, return continuation) outperforms CVX-Episodic (match problem, return full episode) by leveraging temporal structure.

## Experimental Setup

| Component | Choice |
|-----------|--------|
| **Training corpus** | MBPP sanitized (384 problems, 3 steps each) |
| **Test benchmark** | HumanEval (164 problems) |
| **Embedding** | all-MiniLM-L6-v2 (D=384) |
| **LLM** | qwen2.5-coder:7b-instruct (Ollama) |

### Conditions

| Condition | Retrieval | Formatting |
|-----------|-----------|------------|
| NoMemory | None | Zero-shot |
| RandomFewShot | Random MBPP | Full problem + solution |
| FlatCosine | numpy cosine on problems | Full problem + solution |
| CVX-Episodic | CVX search, step_0 only | Full problem + solution |
| **CVX-Causal** | CVX search, ALL steps | **Continuation only** (what happened after match) |

### Protocol

- **Validation**: HumanEval[0:82], T=0, k ∈ {1,3,5,7}
- **Test**: HumanEval[82:164], T=0.2, 5 seeds, best k from validation
- **Statistics**: McNemar (majority-vote), paired t-test (seed-level)

## Results

### Validation (T=0)

| k | NoMemory | Random | Flat | CVX-Episodic | **CVX-Causal** |
|---|----------|--------|------|-------------|---------------|
| 1 | 59.8% | 85.4% | 85.4% | 84.1% | 81.7% |
| 3 | 58.5% | 78.0% | 85.4% | 84.1% | 78.0% |
| **5** | 58.5% | 82.9% | 81.7% | 81.7% | **82.9%** |
| 7 | 58.5% | 85.4% | 79.3% | 79.3% | 81.7% |

### Test (T=0.2, 5 seeds, k=5)

| Condition | pass@1 (mean ± std) |
|-----------|-------------------|
| NoMemory | 71.2% ± 1.5% |
| CVX-Causal | 75.4% ± 2.6% |
| RandomFewShot | 77.3% ± 1.7% |
| CVX-Episodic | 77.3% ± 2.0% |
| **FlatCosine** | **78.5% ± 1.8%** |

## Key Findings

1. **H1 rejected**: Semantic retrieval (FlatCosine 78.5%, CVX-Episodic 77.3%) does not significantly outperform random few-shot (77.3%). For code generation, any example teaches formatting; similarity adds marginal value.

2. **H2 rejected**: CVX-Causal (75.4%) underperforms both CVX-Episodic (77.3%) and FlatCosine (78.5%). Matching on plan/solution steps without interactive state feedback introduces noise rather than signal.

3. **CVX ≈ FlatCosine**: 96.7% retrieval overlap at this corpus size. HNSW approximate search finds the same neighbors as brute-force.

4. **CVX-Causal step distribution**: When searching across all steps, the model finds matches at plan (step 1) and solution (step 2) — but the resulting continuations (just the solution without the problem context) are less useful than the full episode.

## Interpretation

The CVX-Causal null result is **expected** for code generation with static retrieval. The causal hypothesis ("show what happened after a similar state") requires the query to be an **in-progress state**, not a task description. For code:

- The "state" at query time is always the same: "I have a problem description, generate code"
- There is no mid-episode state differentiation — the problem is either solved or not
- Matching on plan/solution vectors returns examples based on code similarity rather than problem similarity, which is less useful for prompting

**Where CVX-Causal should work**: Interactive environments (ALFWorld, games) where the agent has genuine mid-episode states that evolve. This requires step-by-step environment interaction, not static plan generation.

## CVX Features Exercised

`TemporalIndex`, `insert`, `search` (across all steps, not just step_0), `save`/`load`, episode encoding, timestamp-based step filtering.
