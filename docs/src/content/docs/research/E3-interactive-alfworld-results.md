---
title: "E3: Interactive ALFWorld Agent — Results"
description: "CVX causal memory with real environment execution and step-by-step retrieval"
---

## Hypothesis

**H**: An LLM agent that queries CVX at each step using its **current environment state** (not a static task description) will complete more tasks than a zero-shot agent, because the temporal structure of CVX enables extraction of relevant continuations from similar mid-episode states.

This is the hypothesis that E2's static evaluation could not test.

## Experimental Setup

| Component | Choice |
|-----------|--------|
| **Environment** | ALFWorld TextWorld (eval_out_of_distribution, 134 games) |
| **LLM** | qwen2.5-coder:7b-instruct (Ollama) |
| **Memory** | CVX index from E2 (336 AgentInstruct expert episodes, 4542 vectors) |
| **Evaluation** | 30 games, max 30 steps per game |

### The Interactive Loop

```
1. env.reset() → observation + task
2. embed(observation + task context) → query CVX
3. CVX.search() → find similar mid-episode states across all expert episodes
4. Extract continuation: next 5 steps from each matched episode
5. LLM(observation + continuations + admissible_actions) → choose action
6. env.step(action) → new observation
7. → goto 2 until task complete or max_steps
```

### Key Difference from E2

| | E2 (Static) | E3 (Interactive) |
|--|-------------|-----------------|
| Query | Task description | **Current observation from env** |
| When | Once per task | **Every step** |
| Context | Full episode trajectories | **Continuations from similar states** |
| Metric | Plan similarity to expert | **Task completion (binary)** |
| Environment | None | **ALFWorld TextWorld simulator** |

### Conditions

| Condition | Memory | Query |
|-----------|--------|-------|
| NoMemory | None | Only observation + admissible actions |
| CVX-Causal | CVX step-by-step | Observation → search → continuation |

## Results

| Condition | Completed | Rate | Mean Steps |
|-----------|-----------|------|-----------|
| NoMemory | 1/30 | **3.3%** | 29.3 |
| **CVX-Causal** | **6/30** | **20.0%** | 27.2 |

### Statistical Test

**McNemar's test:**
- CVX-Causal only won: 5 tasks
- NoMemory only won: 0 tasks
- Both won: 1
- Neither won: 24
- **Net: +5 tasks, χ²=3.20, p=0.074** (borderline at n=30)

### Retrieval Characteristics

The match step distribution shows CVX-Causal correctly matching mid-episode states:
- Early steps (0-3): room descriptions, initial exploration
- Mid steps (4-7): object interaction, navigation
- Late steps (8+): task completion actions

## Findings

1. **6x improvement in task completion** (3.3% → 20.0%). This is the strongest result across E1-E3.

2. **The interactive loop is essential.** E2 showed CVX-Causal was worse than zero-shot when the query was a static task description. E3 shows it's 6x better when the query is the real environment state. Same memory, same CVX index, same LLM — the only difference is *what you query with*.

3. **Continuations provide actionable guidance.** When the agent sees "You are at countertop 2, you see a tomato", CVX finds expert states like "agent was at countertop with tomato in a 'cool tomato' task" and returns "take tomato → go to fridge → cool tomato" — directly executable actions.

4. **NoMemory is nearly helpless.** A 7B model with only admissible actions and no examples solves 3.3% of out-of-distribution ALFWorld tasks. It lacks the procedural knowledge of which sequence of actions leads to task completion.

5. **p=0.074 is borderline.** With n=30, statistical power is limited. Scaling to n=134 (all eval games) would likely reach significance given the 0/5 discordant pair ratio.

## Why This Validates CVX

| Capability | Used in E3? | Why it matters |
|-----------|-------------|---------------|
| `search()` across all steps | Yes | Finds similar states anywhere in any episode |
| Episode encoding (entity_id) | Yes | Groups steps by episode for continuation extraction |
| Timestamp ordering | Yes | Determines step order → extracts "what came next" |
| Step-level embeddings | Yes | Each action-observation pair is a searchable state |

**FlatCosine cannot replicate this** because:
- It has no concept of step ordering within an episode
- It cannot extract "continuation from step N" — only "top-k most similar vectors"
- It would need to re-implement episode grouping, step ordering, and continuation slicing — which is exactly what CVX provides natively

## Limitations

- **n=30**: Small sample. Should scale to full 134 eval games.
- **No FlatCosine baseline**: We should add a flat cosine agent that retrieves full episodes (like E2's best condition) for fair comparison.
- **No ablation on continuation length**: We use next-5-steps; optimal length is unknown.
- **Single model**: Larger models may not need memory (ceiling effect).
- **Admissible actions provided**: ALFWorld gives the valid action set; real environments don't.

## Next Steps

1. **Scale to n=134** (all eval_ood games) for statistical power
2. **Add FlatCosine interactive baseline** (full-episode context at each step)
3. **Expose `causal_search()` in Python** for native CVX support
4. **Apply to code generation** as an iterative generate-test-debug loop (see E4 proposal)
