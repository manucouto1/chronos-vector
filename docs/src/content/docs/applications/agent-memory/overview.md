---
title: "AI Agent Long-Term Memory"
description: "Using CVX as episodic memory for AI agents: storing and retrieving successful action sequences dependent on context"
---

## The Vision

Standard RAG gives LLM agents access to **facts** — but not to **how a problem was solved before**. This maps to cognitive science's separation between **semantic memory** (what you know) and **episodic memory** (what you've done and how it went).

Current vector stores are unordered — they cannot return the *sequence* of states an episode traversed. CVX provides episode identity, temporal ordering, causal continuation, reward filtering, and trajectory analytics in a single index.

---

## State of the Art (2024-2025)

### Memory-Augmented Agent Systems

| System | Architecture | ALFWorld | Model | Year |
|--------|-------------|----------|-------|------|
| **AutoManual** | Online rule learning (procedural) | **97.4%** | GPT-4-turbo | NeurIPS 2024 |
| **Reflexion** | Verbal self-reflection (episodic) | **~97%** | GPT-4 | NeurIPS 2023 |
| **ReflAct** | Goal-state reflection | 94.8% | GPT-4o | 2025 |
| **Memp** | Procedural memory repository | 87.1% | GPT-4o | 2025 |
| **AutoManual** | Same, smaller model | 86.2% | GPT-3.5-turbo | NeurIPS 2024 |
| **ExpeL** | Experience extraction + insights | ~59% | GPT-4 (1-shot) | 2023 |
| **CLIN** | Continual causal abstractions | +11pp unseen | GPT-4 | COLM 2024 |
| **CVX-Causal** | Temporal episodic (causal retrieval) | **20%** | qwen2.5:7b (1-shot) | This work |
| No memory | Baseline | 3.3% | qwen2.5:7b | This work |

**Key insight**: CVX achieves 6x improvement (3.3%→20%) with a 7B model. The SOTA systems use GPT-4/GPT-4-turbo. The critical missing experiment: **CVX + GPT-4o vs ExpeL + GPT-4** (both 1-shot, same conditions).

### Memory Architecture Comparison

| System | Storage | Retrieval | Temporal? | Reward filter? |
|--------|---------|-----------|-----------|---------------|
| Generative Agents | NL stream | Recency + relevance + importance | Decay only | No |
| Reflexion | Verbal reflections | Sliding window | No | No |
| ExpeL | Trajectories + rules | Task similarity | Partial | Post-hoc |
| Voyager | Code skill library | Embedding similarity | No | No |
| MemGPT/Letta | Hierarchical paging | OS-style tier routing | Partial | No |
| CLIN | Causal abstractions | Persistent updates | Trial-level | No |
| Zep/Graphiti | Temporal knowledge graph | Time + semantic + graph | **Yes** | No |
| **CVX** | **Temporal vector index** | **HNSW + causal search** | **Yes** | **Yes (bitmap)** |

CVX is the only system that combines episode identity, temporal ordering, causal continuation, reward pre-filtering, and trajectory analytics (signatures, changepoints) in a single index.

### Relevant Benchmarks

| Benchmark | Domain | CVX Fit | Current SOTA |
|-----------|--------|---------|-------------|
| **ALFWorld** | Household RL | High — step-by-step causal retrieval | 97.4% (AutoManual, GPT-4-turbo) |
| **LongMemEval** | Temporal reasoning | High — timestamps, ordering | 95% (OM+gpt-5-mini) |
| **Mem2ActBench** | Action from memory | High — temporal span, metadata | New benchmark, few baselines |
| **MemoryAgentBench** | Knowledge updates | Medium — changepoint detection | No system masters all 4 tasks |
| **HumanEval** | Code generation | Low — saturated by frontier models | 96.2% (o1-mini) |

---

## CVX Capabilities for Agent Memory

### Implemented (RFC-012 P1-P4)

| Feature | API | Purpose |
|---------|-----|---------|
| **Episode encoding** | `entity_id = episode_id * 10000 + step` | Group steps into episodes |
| **Causal search** | `index.causal_search(vec, k, temporal_context)` | "What happened next?" — successor/predecessor edges |
| **Hybrid search** | `index.hybrid_search(vec, k, beta)` | Beam search exploring semantic + temporal neighbors |
| **Reward filtering** | `index.search_with_reward(vec, k, min_reward)` | Only retrieve successful experiences (bitmap pre-filter) |
| **Metadata pre-filtering** | `index.insert(..., metadata={"goal": "clean"})` | Context-dependent retrieval via inverted index |
| **Native centering** | `index.set_centroid(centroid)` | 30x signal improvement for anisotropic embeddings |
| **Path signatures** | `cvx.path_signature(traj, depth)` | Trajectory shape comparison |
| **Change point detection** | `cvx.detect_changepoints(id, traj)` | Identify regime shifts in agent behavior |

### Episode Data Model

```
entity_id:  episode_id * 10000 + step_index
timestamp:  episode_id * 10000 + step_index  (monotonic within episode)
vector:     embed(observation + action)
reward:     0.0-1.0 (set retroactively via set_reward())
metadata:   {"goal": "clean", "room": "kitchen", "task_type": "pick"}
```

---

## Proof of Concept Results (E1-E4, Ollama qwen2.5:7b)

All experiments use a local 7B model via Ollama on HPC (NVIDIA 3090). These establish the baseline signal before scaling to frontier models.

### E1: Code Generation (MBPP → HumanEval)

- **Memory**: 384 MBPP episodes × 3 steps (problem, plan, solution)
- **Retrieval**: Flat cosine + CVX episodic + CVX causal
- **Result**: **77.8% pass@1** with episodic retrieval (vs 72.9% no memory)
- **Insight**: Causal continuation (return solution from matched problem) works as well as full-episode retrieval

### E2: ALFWorld Plan Quality

- **Memory**: 336 AgentInstruct expert trajectories (3-35 steps)
- **Metric**: Plan semantic similarity, action verb overlap, object recall
- **Result**: 0.709 semantic similarity (CVX-Episodic)
- **Insight**: Causal continuation underperforms for planning because LLM needs a *plan*, not the *next action*

### E3: Interactive ALFWorld (Key Experiment)

- **Memory**: Same 336 expert trajectories indexed with episode encoding
- **Protocol**: Agent queries CVX at each step with current observation, receives continuation from similar past states
- **Result**: **3.3% → 20.0% task completion (6× improvement)**
- **Statistical test**: McNemar p=0.074 (marginally significant, 30 games)
- **Insight**: Step-by-step retrieval from current state outperforms full-plan retrieval

### E4: Iterative Debugging (APPS)

- **Memory**: 172 debug traces (error → attempt → fix)
- **Protocol**: Agent retries failed code, querying CVX with error embedding
- **Result**: **28% → 31%** (+3 problems rescued)
- **Insight**: Modest effect with 7B model — debug memory is more valuable with stronger base models

---

## Research Roadmap

### Phase 1: Scale to Frontier Models (In Progress)

**E5: ALFWorld with GPT-4o** — The critical experiment.

| Condition | Model | Memory | Expected |
|-----------|-------|--------|----------|
| NoMemory | GPT-4o | None | ~70-75% (ReAct baseline) |
| CVX-Causal | GPT-4o | 336 expert trajectories | Target: >59% (beat ExpeL 1-shot) |
| CVX-Causal + reward | GPT-4o | Same, filtered by success | Target: >65% |
| CVX-Causal + retry | GPT-4o | + Reflexion-style retry (3 rounds) | Target: >85% |

If CVX-Causal + GPT-4o > ExpeL (59%), this is a publishable result demonstrating that **temporal vector memory outperforms experience extraction** for interactive agents.

**E6: ALFWorld with GPT-4o-mini** — Cost-effective scaling test.

Same conditions as E5 but with GPT-4o-mini (~$0.15/1M tokens). Tests whether CVX's memory compensates for a weaker model — if CVX + 4o-mini approaches NoMemory + 4o, memory is a model-size substitute.

### Phase 2: Temporal Reasoning Benchmarks

**E7: LongMemEval — Temporal Reasoning Subtask**

CVX's temporal features (timestamps, ordering, causal_search) directly address the temporal reasoning questions in LongMemEval:
- "When did X happen relative to Y?"
- "What changed between session 3 and session 7?"
- "What was the most recent update to topic Z?"

Current SOTA: 71.2% (Zep/Graphiti + GPT-4o) on temporal reasoning, 95% (OM + gpt-5-mini) overall.

**E8: Mem2ActBench**

New benchmark (2025) testing whether agents can infer constraints from history and ground them into tool calls. Few baselines exist — early entry opportunity. CVX's metadata filtering and temporal span tracking are directly applicable.

### Phase 3: Advanced Memory Patterns

| Experiment | CVX Feature | Research Question |
|-----------|-------------|-------------------|
| **Reward-weighted retrieval** | `search_with_reward()` | Does filtering by success improve action quality? |
| **Context-conditioned search** | Metadata pre-filtering | Does goal-aware retrieval outperform embedding-only? |
| **Trajectory signature matching** | `path_signature()` | Can we match solution *shapes* rather than states? |
| **Memory regime detection** | `detect_changepoints()` | Can we detect when agent strategy shifts? |
| **Cross-episode Granger causality** | `granger_causality()` | Do actions in episode A *cause* improvements in B? |

### Phase 4: Competitive Publication

**Target paper**: "Trajectory-Aware Vector Memory for Interactive Agents"

- **Venue**: NeurIPS / ICML / COLM
- **Core claim**: Temporal vector memory with causal continuation outperforms unstructured retrieval and procedural memory for interactive agents
- **Experiments**: E5 (ALFWorld + GPT-4o) + E7 (LongMemEval) + E8 (Mem2ActBench)
- **Baseline comparison**: ExpeL, Memp, Reflexion, CLIN, Zep

---

## Key References

### Memory Architectures
1. Park et al. "Generative Agents" (UIST 2023) — Memory stream with recency/relevance/importance
2. Shinn et al. "Reflexion" (NeurIPS 2023) — Verbal self-reflection as memory
3. Zhao et al. "ExpeL" (2023) — Experience extraction + insight learning
4. Chen et al. "AutoManual" (NeurIPS 2024) — Online rule learning, 97.4% ALFWorld
5. Majumder et al. "CLIN" (COLM 2024) — Continual causal abstractions
6. Fang et al. "Memp" (2025) — Procedural memory repository
7. Packer et al. "MemGPT/Letta" (2023) — OS-style hierarchical memory

### Temporal Memory
8. Rasmussen et al. "Zep/Graphiti" (2025) — Temporal knowledge graph for agents
9. "MapAgent" (2025) — Trajectory-constructed memory for planning
10. Zheng et al. "Synapse" (2023) — Trajectory-as-exemplar prompting

### Benchmarks
11. Wu et al. "LongMemEval" (ICLR 2025) — 5 memory abilities, temporal reasoning
12. Maharana et al. "LoCoMo" (ACL 2024) — Very long-term conversational memory
13. "MemBench" (ACL Findings 2025) — Comprehensive memory evaluation
14. "MemoryAgentBench" (ICLR 2026) — Incremental multi-turn interactions
15. "Mem2ActBench" (2025) — Memory to action grounding

### Surveys
16. "Memory in the Age of AI Agents" (2024) — Comprehensive taxonomy
17. "Memory for Autonomous LLM Agents" (2026) — Mechanisms, evaluation, frontiers

---

## Notebooks

### Completed (Proof of Concept — Ollama qwen2.5:7b)

| Notebook | Focus | Key Result |
|----------|-------|------------|
| E1_episodic_coding | Code gen with episodic retrieval | 77.8% HumanEval pass@1 |
| E2_episodic_alfworld | Plan quality from episodic retrieval | 0.709 semantic similarity |
| E3_interactive_alfworld | Step-by-step agent with CVX causal search | **3.3% → 20% completion (6×)** |
| E4_iterative_coding | Debug retry with error memory | 28% → 31% |

### In Progress (Scaling to Frontier Models)

| Notebook | Focus | Status |
|----------|-------|--------|
| E5_alfworld_gpt4o | ALFWorld with GPT-4o (critical experiment) | Planned |
| E6_alfworld_gpt4o_mini | Cost-effective scaling test | Planned |

### Planned

| Notebook | Focus | Benchmark |
|----------|-------|-----------|
| E7_longmemeval | Temporal reasoning evaluation | LongMemEval (ICLR 2025) |
| E8_mem2act | Memory-to-action grounding | Mem2ActBench (2025) |

---

## Related

- [Episodic Trace Memory — Concept](/chronos-vector/research/episodic-trace-memory)
- [Episodic Memory — Full Experimental Report](/chronos-vector/research/episodic-memory-experiments)
- [RFC-010: Temporal Graph Extension](/chronos-vector/rfc/rfc-010) (causal search infrastructure)
- [RFC-012 Part C: Architecture Gaps](/chronos-vector/rfc/rfc-012) (agent memory roadmap)
- [Episodic Memory Tutorial](/chronos-vector/tutorials/guides/episodic-memory) (code tutorial with synthetic data)
