---
title: "Episodic Memory for AI Agents"
description: "Store action sequences, retrieve by similarity, and extract continuations with causal search"
---

import { Aside } from '@astrojs/starlight/components';

## Semantic vs Episodic Memory

Standard RAG gives agents access to **facts** (semantic memory). CVX provides **episodic memory** — the ability to recall *how a problem was solved before*, step by step.

| Capability | Standard Vector Store | CVX |
|-----------|----------------------|-----|
| Find similar states | Cosine search | Cosine + temporal distance |
| Episode identity | Not native | `entity_id = episode_id * 10000 + step` |
| "What happened next?" | Not possible | `causal_search()` — successor edges |
| Filter by success | Post-hoc | `search_with_reward(min_reward=0.5)` — bitmap pre-filter |

## Episode Data Model

Each step is a `TemporalPoint` with encoded episode identity:

```python
import chronos_vector as cvx
import numpy as np
np.random.seed(42)

index = cvx.TemporalIndex(m=16, ef_construction=100)

# 20 episodes, 5 steps each, 60% success rate
episodes = []
for ep in range(20):
    success = np.random.random() > 0.4
    reward = 1.0 if success else 0.0
    state = np.random.randn(32).astype(np.float32) * 0.5
    nodes = []
    for step in range(5):
        state = state + np.random.randn(32).astype(np.float32) * 0.1
        eid = ep * 10000 + step
        nid = index.insert(eid, eid, state.tolist())
        nodes.append(nid)
    for nid in nodes:
        index.set_reward(nid, reward)
    episodes.append({'id': ep, 'success': success, 'nodes': nodes})

successes = sum(1 for e in episodes if e['success'])
print(f"{len(episodes)} episodes ({successes} successful), {len(index)} points")
```

```text
20 episodes (13 successful), 100 points
```

<iframe src="/chronos-vector/plots/episodic_outcomes.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

## Reward-Filtered Search

Only retrieve experiences that **worked**:

```python
query = np.random.randn(32).astype(np.float32) * 0.5
results = index.search_with_reward(query.tolist(), k=3, min_reward=0.5)
for eid, ts, score in results:
    ep_id, step = eid // 10000, eid % 10000
    print(f"  Episode {ep_id} step {step}: score={score:.3f}")
```

```text
  Episode 12 step 3: score=12.441
  Episode 7 step 2: score=13.205
  Episode 15 step 4: score=14.112
```

<Aside type="tip" title="Bitmap pre-filtering">
`search_with_reward` uses a RoaringBitmap to pre-filter nodes during HNSW traversal — not post-filtering. This is O(1) per node, same cost as temporal filtering.
</Aside>

## Causal Search: "What Happened Next?"

The most powerful pattern — find similar states and return the **continuation**:

```python
results = index.causal_search(
    vector=query.tolist(),
    k=3,
    temporal_context=3,  # walk 3 steps forward/backward
)
for r in results:
    ep_id = r['entity_id'] // 10000
    step = r['entity_id'] % 10000
    print(f"Match: Episode {ep_id} step {step} (score={r['score']:.3f})")
    print(f"  {len(r['successors'])} successors, {len(r['predecessors'])} predecessors")
```

```text
Match: Episode 12 step 3 (score=11.283)
  1 successors, 3 predecessors
Match: Episode 7 step 2 (score=12.910)
  2 successors, 2 predecessors
Match: Episode 15 step 4 (score=13.557)
  0 successors, 3 predecessors
```

<iframe src="/chronos-vector/plots/episodic_causal.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

Each match shows the step that was found (0) plus the steps that came before (-) and after (+) in the same episode.

## Hybrid Search

Beam search exploring both **semantic neighbors** (HNSW graph edges) and **temporal neighbors** (predecessor/successor edges):

$$\text{score} = (1 - \beta) \cdot d_{\text{semantic}} + \beta \cdot d_{\text{temporal\_neighbor}}$$

With $\beta = 0$: pure semantic. With $\beta = 1$: aggressive temporal exploration.

```python
results = index.hybrid_search(query.tolist(), k=5, beta=0.3)
for eid, ts, score in results:
    print(f"  Episode {eid//10000} step {eid%10000}: score={score:.3f}")
```

## Agent Loop (Pseudocode)

```python
def agent_step(observation, index, embed_fn, llm_fn):
    embedding = embed_fn(observation)

    # Find successful past states similar to mine
    memories = index.causal_search(
        vector=embedding, k=3, temporal_context=5
    )

    # Extract continuations as LLM context
    context = [{
        "similarity": m["score"],
        "next_steps": [vec for _, _, vec in m["successors"]],
    } for m in memories]

    # LLM decides action
    action = llm_fn(observation, context)
    result, reward = env.step(action)

    # Store experience for future retrieval
    node_id = index.insert(
        entity_id=current_episode * 10000 + current_step,
        timestamp=current_episode * 10000 + current_step,
        vector=embedding,
        reward=reward,
    )
    return action, result
```

## Validated Results

| Experiment | Task | Baseline | CVX Memory |
|-----------|------|----------|------------|
| E1 | Code generation (HumanEval) | — | 77.8% pass@1 |
| E3 | Interactive RL (ALFWorld) | 3.3% | **20.0%** (6x) |
| E4 | Iterative debugging (APPS) | 28.0% | **31.0%** |
