---
title: "Episodic Memory for AI Agents"
description: "Store action sequences, retrieve by similarity, and extract continuations with causal search"
---

This tutorial shows how to use CVX as long-term episodic memory for an AI agent. No external LLM needed — we use synthetic episodes.

## Concept

Standard vector stores retrieve facts. CVX retrieves **experience sequences**:
- "Find past states similar to mine"
- "Show me what happened next"
- "Only show me successful experiences"

## Episode Data Model

Each step in an episode is a TemporalPoint:

```python
import chronos_vector as cvx
import numpy as np

np.random.seed(42)
D = 32

index = cvx.TemporalIndex(m=16, ef_construction=100)
```

### Encoding Episodes

Episodes use the entity_id to group steps:

```
entity_id = episode_id * 10000 + step_index
timestamp = episode_id * 10000 + step_index
```

```python
def encode_episode(episode_id, step_index):
    """Encode episode:step into entity_id and timestamp."""
    entity_id = episode_id * 10000 + step_index
    timestamp = episode_id * 10000 + step_index
    return entity_id, timestamp

def decode_episode(entity_id):
    """Decode entity_id back to (episode_id, step_index)."""
    return entity_id // 10000, entity_id % 10000
```

## Store Episodes with Outcomes

```python
# Simulate 20 episodes, each 5 steps, with varying success
episodes = []
for ep in range(20):
    success = np.random.random() > 0.4  # 60% success rate
    reward = 1.0 if success else 0.0

    steps = []
    state = np.random.randn(D).astype(np.float32) * 0.5
    for step in range(5):
        # Each step modifies the state
        action = np.random.randn(D).astype(np.float32) * 0.1
        state = state + action
        embedding = state.tolist()

        eid, ts = encode_episode(ep, step)
        # Annotate last step with reward (or all steps)
        step_reward = reward if step == 4 else None
        node_id = index.insert(eid, ts, embedding, reward=step_reward)
        steps.append((eid, ts, node_id, embedding))

    episodes.append({"id": ep, "success": success, "reward": reward, "steps": steps})

successes = sum(1 for e in episodes if e["success"])
print(f"Stored {len(episodes)} episodes ({successes} successful), {len(index)} total points")
```

## Retrieve Similar States

Standard search finds the most similar vectors:

```python
# Agent is in a new state — find similar past states
current_state = episodes[0]["steps"][2][3]  # reuse a known state
results = index.search(current_state, k=3)
for eid, ts, score in results:
    ep_id, step = decode_episode(eid)
    print(f"  Episode {ep_id}, step {step}: score={score:.4f}")
```

## Reward-Filtered Search

Only retrieve from successful episodes:

```python
# First, annotate all steps of successful episodes with reward
for ep in episodes:
    if ep["success"]:
        for eid, ts, node_id, _ in ep["steps"]:
            index.set_reward(node_id, 1.0)
    else:
        for eid, ts, node_id, _ in ep["steps"]:
            index.set_reward(node_id, 0.0)

# Search only successful experiences
results = index.search_with_reward(current_state, k=3, min_reward=0.5)
for eid, ts, score in results:
    ep_id, step = decode_episode(eid)
    r = index.reward(results[0][0] // 10000 * 10000)  # check
    print(f"  Episode {ep_id}, step {step}: score={score:.4f} (successful)")
```

## Causal Search: "What Happened Next?"

The most powerful pattern — find similar states and return the continuation:

```python
results = index.causal_search(
    vector=current_state,
    k=3,
    temporal_context=3,  # walk 3 steps forward/backward
)

for r in results:
    ep_id, step = decode_episode(r["entity_id"])
    print(f"\nMatch: Episode {ep_id}, step {step} (score={r['score']:.4f})")

    if r["successors"]:
        print(f"  What happened next ({len(r['successors'])} steps):")
        for node_id, ts, vec in r["successors"]:
            s_ep, s_step = decode_episode(ts // 1)
            print(f"    Step {s_step}: vec[0:3]={[f'{v:.2f}' for v in vec[:3]]}")

    if r["predecessors"]:
        print(f"  What happened before ({len(r['predecessors'])} steps):")
        for node_id, ts, vec in r["predecessors"]:
            p_ep, p_step = decode_episode(ts // 1)
            print(f"    Step {p_step}: vec[0:3]={[f'{v:.2f}' for v in vec[:3]]}")
```

## Hybrid Search: Semantic + Temporal Exploration

Beam search that explores both semantic neighbors AND temporal edges:

```python
results = index.hybrid_search(
    vector=current_state,
    k=5,
    beta=0.3,  # 30% weight on temporal edge exploration
)
for eid, ts, score in results:
    ep_id, step = decode_episode(eid)
    print(f"  Episode {ep_id}, step {step}: score={score:.4f}")
```

## Full Agent Loop (Pseudocode)

```python
def agent_step(observation, index, embed_fn, llm_fn):
    """One step of an agent with CVX episodic memory."""

    # 1. Embed current observation
    embedding = embed_fn(observation)

    # 2. Query CVX for similar past states (successful only)
    memories = index.causal_search(
        vector=embedding,
        k=3,
        temporal_context=5,
        min_reward=0.5,  # only successful experiences
    )

    # 3. Extract continuations as context for the LLM
    context = []
    for m in memories:
        continuation = [vec for _, _, vec in m["successors"]]
        context.append({
            "similarity": m["score"],
            "what_happened_next": continuation,
        })

    # 4. LLM decides action based on observation + retrieved continuations
    action = llm_fn(observation, context)

    # 5. Execute action, get result
    result, reward = env.step(action)

    # 6. Store this step in CVX for future retrieval
    node_id = index.insert(
        entity_id=encode_episode(current_episode, current_step)[0],
        timestamp=encode_episode(current_episode, current_step)[1],
        vector=embedding,
        reward=reward,
    )

    return action, result
```

## Save & Load

```python
index.save("agent_memory")

# Later — restore memory without rebuilding
index = cvx.TemporalIndex.load("agent_memory")
print(f"Restored {len(index)} memories")
```
