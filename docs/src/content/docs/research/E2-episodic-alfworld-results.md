---
title: "E2: Episodic ALFWorld Benchmark — Results"
description: "Experimental evaluation of CVX episodic trace memory on embodied agent planning with AgentInstruct"
---

## Hypothesis

**H1**: An LLM agent augmented with retrieved expert trajectories (via CVX episodic memory) will produce more detailed and expert-aligned action plans than the same agent with no context or random trajectories.

**H2**: Semantic retrieval selects trajectories from similar task types (e.g., "clean mug" retrieves "clean plate", not "examine lamp"), producing more transferable procedural knowledge than random selection.

## Experimental Setup

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Dataset** | AgentInstruct ALFWorld split (336 GPT-4 expert trajectories) | Real expert demonstrations, parsed from conversation format |
| **LLM** | qwen2.5-coder:7b-instruct (Ollama) | Open-weight, reproducible local inference |
| **Embedding model** | all-MiniLM-L6-v2 (D=384) | Consistent with E1 |
| **Memory backend** | CVX TemporalIndex (M=16, ef=100) | Episode encoding: `entity_id = ep_idx << 16 \| step_idx` |
| **Retrieval** | top-k=3, reward ≥ 0.7 filter | Only retrieve successful trajectories |

### Dataset Characteristics

| Property | Value |
|----------|-------|
| Total episodes | 336 |
| Successful (expert) | 336/336 (100%) |
| Task types | put (158), clean (68), cool (49), examine (37), heat (24) |
| Steps/episode | min=3, median=10, max=35 |

### Episode Encoding

Each ALFWorld trajectory step is stored as a temporal vector:
- **Text**: `"Task: {task} | Action: {action} | Obs: {observation}"`
- **Timestamp**: `ep_idx * 10000 + step_idx`
- **Entity ID**: `ep_idx << 16 | step_idx`

### Conditions

| Condition | Context Provided | Controls For |
|-----------|-----------------|--------------|
| **NoMemory** | None (zero-shot) | Baseline planning ability |
| **RandomTrajectory** | 3 random expert trajectories | Effect of any trajectory examples |
| **CVX-Episodic** | 3 most similar expert trajectories via CVX | Value of semantic retrieval |

### Evaluation Protocol

- **Metric**: Mean plan steps and expert alignment ratio (plan_steps / expert_steps)
- **n=30**: First 30 episodes evaluated across all conditions
- **Temperature**: 0.0 (deterministic)
- **Note**: Without a live ALFWorld environment, we measure plan granularity rather than task completion. A ratio closer to 1.0 indicates plans with expert-level detail.

## Results

### Primary Metrics

| Condition | Mean Steps | Expert Mean | Ratio (plan/expert) |
|-----------|-----------|-------------|---------------------|
| NoMemory | 6.4 | 14.0 | **0.45** |
| RandomTrajectory | 6.6 | 14.0 | **0.47** |
| CVX-Episodic | 8.1 | 14.0 | **0.58** |

### Retrieval Quality

Sample retrieval for "find two laptop and put them in bed":

| Rank | Episode | Similarity | Task |
|------|---------|-----------|------|
| 1 | #0 | 0.240 | find two laptop and put them in bed |
| 2 | #259 | 0.307 | put two laptop in bed |
| 3 | #45 | 0.493 | find two book and put them in bed |

Retrieval correctly identifies semantically related tasks (same objects, same locations, same action types).

## Findings

1. **CVX-Episodic generates 26% more detailed plans than zero-shot** (8.1 vs 6.4 steps). When the agent sees how similar tasks were solved with 10-14 steps, it produces more granular action sequences rather than collapsing steps.

2. **Random trajectories provide minimal benefit** (+0.2 steps, ratio 0.47 vs 0.45). Unrelated trajectories (e.g., showing "examine lamp" when the task is "clean mug") don't transfer useful procedural knowledge.

3. **Semantic retrieval finds genuinely similar tasks.** The retrieval correctly groups by task structure: "find X and put in Y" retrieves other "find/put" tasks with related objects. This validates CVX's search operating on the action-observation embedding space.

4. **Gap to expert remains large** (ratio 0.58 vs 1.0). The 7B model underspecifies plans even with examples — likely a model capacity issue rather than a retrieval failure.

## Limitations & Threats to Validity

- **No environment execution**: The primary limitation — we measure plan length, not task completion rate. Plan length is a proxy: more detailed plans are not necessarily correct plans. Full evaluation requires the `alfworld` simulator.
- **All episodes are successful**: With reward=1.0 for all trajectories, the reward filter is vacuous. A mixed-quality corpus would better demonstrate the value of reward-filtered retrieval.
- **n=30**: Small evaluation set. Full 336-episode leave-one-out evaluation would be more rigorous.
- **Single model**: A larger model (e.g., 70B) might produce expert-length plans even without retrieval (ceiling effect).
- **Plan length ≠ plan quality**: A plan with 14 steps could be wrong in different ways than a plan with 6 steps. Step count measures verbosity as much as correctness.

## CVX Features Exercised

`TemporalIndex`, `insert`, `search`, `save`/`load`, episode encoding, reward-filtered retrieval (structural — all rewards are 1.0 in this dataset).
