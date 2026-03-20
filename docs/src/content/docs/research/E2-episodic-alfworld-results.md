---
title: "E2: Episodic ALFWorld Benchmark — Results"
description: "Experimental evaluation of CVX episodic trace memory on embodied agent planning"
---

## Hypothesis

**H1**: Semantic retrieval significantly outperforms random/zero-shot for embodied planning.

**H2**: CVX-Causal (match mid-episode states, return continuation) outperforms full-episode retrieval by providing targeted procedural context.

## Experimental Setup

| Component | Choice |
|-----------|--------|
| **Dataset** | AgentInstruct ALFWorld (336 GPT-4 expert trajectories, 3–35 steps) |
| **LLM** | qwen2.5-coder:7b-instruct (Ollama) |
| **Embedding** | all-MiniLM-L6-v2 (D=384) |
| **Evaluation** | 99 episodes, stratified by task type, leave-one-out, k=3 |

### Conditions

| Condition | Retrieval | Context |
|-----------|-----------|---------|
| NoMemory | None | Zero-shot |
| RandomTrajectory | Random | Full expert trajectory |
| FlatCosine | numpy cosine on tasks | Full trajectory |
| CVX-Episodic | CVX search | Full trajectory |
| **CVX-Causal** | CVX search ALL steps | **Continuation only** |

### Metrics

- **Semantic Similarity**: cosine(plan embedding, expert trajectory embedding)
- **Action Overlap**: fraction of expert action verbs in plan
- **Object Recall**: fraction of expert objects in plan
- **Step Ratio**: plan_steps / expert_steps

## Results

| Condition | SemSim | ActOvl | ObjRec | StepR |
|-----------|--------|--------|--------|-------|
| NoMemory | 0.600 | 0.60 | 0.09 | 0.56 |
| RandomTrajectory | 0.614 | 0.78 | 0.17 | 0.70 |
| **FlatCosine** | **0.702** | **0.89** | **0.30** | 0.74 |
| **CVX-Episodic** | **0.709** | **0.88** | **0.28** | 0.74 |
| CVX-Causal | 0.588 | 0.70 | 0.21 | 0.43 |

### Statistical Tests (Wilcoxon, semantic similarity)

| Comparison | Δ | p | Sig |
|-----------|---|---|-----|
| CVX-Episodic vs NoMemory | +0.109 | <0.0001 | *** |
| FlatCosine vs NoMemory | +0.102 | <0.0001 | *** |
| CVX-Episodic vs Random | +0.095 | <0.0001 | *** |
| CVX-Episodic vs FlatCosine | +0.007 | 0.213 | ns |
| CVX-Causal vs NoMemory | -0.012 | 0.927 | ns |
| CVX-Causal vs FlatCosine | -0.114 | 1.000 | ns |

### CVX-Causal Match Step Distribution

```
Step  0:   21 ( 7.1%)  ██
Step  1:    8 ( 2.7%)  █
Step  2:    9 ( 3.0%)  █
Step  3:   22 ( 7.4%)  ██
Step  5:   23 ( 7.7%)  ███
Step  7:   39 (13.1%)  █████
Step 10+: 109 (36.7%)  ███████████████
```

## Key Findings

1. **H1 confirmed**: Semantic retrieval (FlatCosine/CVX-Episodic) significantly outperforms zero-shot (+0.10 semantic sim, p<0.0001) and random (+0.09, p<0.0001). Object recall triples (0.09 → 0.30).

2. **H2 rejected**: CVX-Causal (0.588) is **worse than NoMemory** (0.600). The continuation-only format, without full task context, confuses the LLM.

3. **CVX-Episodic ≈ FlatCosine**: +0.007 semantic sim (p=0.213 ns), 67% retrieval overlap.

4. **Why CVX-Causal fails**: 36.7% of matches are at step 10+, meaning the continuation is just the last 2-3 steps of an episode (e.g., "put X in/on Y"). Without the preceding context (which objects, which rooms), these fragments are useless. The query (task description) matches late-episode states because they contain the task text as a prefix.

## Root Cause Analysis: CVX-Causal

The causal retrieval hypothesis is sound but the **implementation context** is wrong:

| What we do | What we should do |
|-----------|------------------|
| Query = task description | Query = current agent state (observation) |
| Match against any step | Match against observation-only embeddings |
| Static plan generation | Interactive step-by-step with environment |
| Return text continuation | Return structured action sequence |

**CVX-Causal requires an interactive loop**:
1. Agent takes action in environment
2. Environment returns observation
3. Embed observation → `search()` in CVX
4. Find similar mid-episode state in memory
5. Return continuation → agent uses it for next action
6. Repeat until task completion

This is a **fundamentally different architecture** than static plan generation. It requires the ALFWorld simulator running in the loop, which is the natural next step.

## What These Results Mean for CVX

- **As a vector store** (insert + search): CVX ≈ numpy. No differentiation at this scale.
- **As an episodic memory** with temporal structure: The value is real but requires an **interactive agent architecture** to exploit it. Static benchmarks don't exercise the temporal features.
- **Next step**: Implement the interactive ALFWorld agent loop where the query to CVX changes at each step based on the actual environment state.

## CVX Features Exercised

`TemporalIndex`, `insert`, `search` (multi-step), `save`/`load`, episode encoding, timestamp-based step extraction, continuation slicing.
