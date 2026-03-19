---
title: "E2: Episodic ALFWorld Benchmark — Results"
description: "Experimental evaluation of CVX episodic trace memory on embodied agent planning with AgentInstruct"
---

## Hypothesis

**H1**: An LLM agent augmented with retrieved expert trajectories will produce plans more similar to expert solutions than zero-shot or random trajectory injection.

**H2**: Semantic retrieval (CVX or flat cosine) selects more useful trajectories than random selection.

## Experimental Setup

| Component | Choice |
|-----------|--------|
| **Dataset** | AgentInstruct ALFWorld split (336 GPT-4 expert trajectories) |
| **LLM** | qwen2.5-coder:7b-instruct (Ollama) |
| **Embedding model** | all-MiniLM-L6-v2 (D=384) |
| **Memory backend** | CVX TemporalIndex (M=16, ef=100) |
| **Retrieval** | top-k=3, leave-one-out evaluation |

### Conditions

| Condition | Description |
|-----------|-------------|
| **NoMemory** | Zero-shot |
| **RandomTrajectory** | k random expert trajectories |
| **FlatCosine** | k most similar by numpy cosine |
| **CVX-Episodic** | k most similar via CVX temporal search |

### Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Semantic Similarity** | Cosine sim between plan and expert trajectory embeddings | [0, 1] |
| **Action Overlap** | Fraction of expert action verbs present in plan | [0, 1] |
| **Object Recall** | Fraction of expert objects mentioned in plan | [0, 1] |
| **Step Ratio** | plan_steps / expert_steps | [0, ∞) |

### Evaluation Protocol

- **n=99** episodes, stratified by task type (put: 47, clean: 20, cool: 14, examine: 11, heat: 7)
- **Leave-one-out**: Each episode excluded from its own retrieval
- **T=0**, deterministic generation

## Results

### Primary Metrics

| Condition | Semantic Sim | Action Overlap | Object Recall | Step Ratio |
|-----------|-------------|----------------|--------------|-----------|
| NoMemory | 0.600 | 0.60 | 0.10 | 0.56 |
| RandomTrajectory | 0.612 | 0.78 | 0.17 | 0.72 |
| **FlatCosine** | **0.700** | **0.88** | **0.30** | 0.74 |
| **CVX-Episodic** | **0.708** | **0.88** | **0.28** | 0.74 |

### Statistical Tests (Wilcoxon Signed-Rank, one-sided)

**Semantic Similarity:**

| Comparison | Δ | W | p | Sig |
|-----------|---|---|---|-----|
| CVX vs NoMemory | +0.108 | 4521 | <0.0001 | *** |
| CVX vs Random | +0.096 | 4035 | <0.0001 | *** |
| CVX vs FlatCosine | +0.008 | 1071 | 0.254 | ns |
| Flat vs NoMemory | +0.100 | 4389 | <0.0001 | *** |
| Flat vs Random | +0.088 | 3968 | <0.0001 | *** |

**Object Recall:**

| Comparison | Δ | W | p | Sig |
|-----------|---|---|---|-----|
| CVX vs NoMemory | +0.187 | 332 | <0.0001 | *** |
| CVX vs Random | +0.110 | 162 | 0.003 | ** |
| CVX vs FlatCosine | -0.016 | 4 | 0.844 | ns |
| Flat vs NoMemory | +0.203 | 385 | <0.0001 | *** |
| Flat vs Random | +0.126 | 158 | 0.001 | *** |

### Retrieval Overlap

CVX and FlatCosine retrieve **67% overlapping episodes**. Unlike E1 (96.7%), here CVX and flat search diverge meaningfully — CVX searches over action-observation embeddings (multi-step), while flat cosine searches task descriptions only.

## Findings

1. **Semantic retrieval significantly outperforms zero-shot and random** (p<0.0001 on both semantic similarity and object recall). This is the central positive result.

2. **CVX-Episodic ≈ FlatCosine** (Δ=+0.008 semantic sim, p=0.254 ns). The two retrieval methods produce statistically indistinguishable results, despite 33% different retrievals.

3. **Object recall is the most discriminating metric** — semantic retrieval triples it (0.10 → 0.28–0.30). When the agent sees expert trajectories with the right objects and locations, it names them in its own plan.

4. **Action vocabulary improves sharply** (0.60 → 0.88) — retrieved trajectories teach the model the ALFWorld action format (go to X, take Y, put Y in/on Z).

5. **Random trajectories help modestly** — they teach action format but not task-specific content, explaining the gap to semantic retrieval.

## Interpretation

Unlike E1 (code generation, null result), E2 shows **clear value for semantic retrieval in embodied planning**. The difference likely stems from:

- **Action sequences are task-dependent**: "clean mug" requires sink → fridge, while "heat egg" requires microwave. Random trajectories may demonstrate wrong action sequences.
- **Object grounding matters**: The right retrieved trajectory names the objects and locations relevant to the current task.
- **Code generation is more format-dependent**: Any Python example teaches function structure; ALFWorld tasks require specific procedural knowledge.

The CVX vs FlatCosine null result suggests that at this scale (336 episodes), exact nearest neighbors matter more than the search method used to find them.

## Limitations

- **No environment execution**: Plans are evaluated against expert trajectories, not by running them in ALFWorld. A correct plan that differs from the expert could score low.
- **All episodes are successful**: With 100% success rate, reward filtering is vacuous. A mixed corpus would better test this feature.
- **Single T=0 pass**: No confidence intervals (but Wilcoxon operates per-problem).
- **Metric validity**: Semantic similarity and action overlap are proxies for plan quality, not task completion.

## CVX Features Exercised

`TemporalIndex`, `insert`, `search`, `save`/`load`, episode encoding, reward-filtered retrieval.
