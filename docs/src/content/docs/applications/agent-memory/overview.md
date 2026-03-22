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

## Experimental Results

### ALFWorld — Consolidated Results

All experiments use the same protocol: 336 expert trajectories from AgentInstruct indexed in CVX with episode encoding. The agent queries CVX at each step with its current observation and receives expert continuations from similar past states. 30 games per condition, `eval_out_of_distribution` split (134 games).

| Experiment | Model | NoMemory | CVX-Causal | Improvement |
|-----------|-------|----------|------------|-------------|
| **E3** (proof of concept) | qwen2.5:7b (Ollama) | 3.3% | 20.0% | **+16.7pp (6.0×)** |
| **E6** | GPT-4o-mini | 13.3% | 26.7% | **+13.4pp (2.0×)** |
| **E5** | GPT-4o | 20.0% | **43.3%** | **+23.3pp (2.2×)** |

**Key findings**:

1. **CVX memory improves performance across all model scales** — from 7B local to frontier GPT-4o
2. **The absolute improvement grows with model capability** — stronger models leverage the retrieved expert actions more effectively (+16.7pp at 7B, +23.3pp at GPT-4o)
3. **Memory partially compensates for model size** — CVX + GPT-4o-mini (26.7%) outperforms NoMemory + GPT-4o-mini (13.3%) and approaches NoMemory + GPT-4o (20.0%)
4. **Critical implementation detail**: the causal context must include **actual action text** from expert successors, not just similarity scores. Empty context (E5 v1) showed zero improvement

### Comparison with SOTA (ALFWorld, 1-shot, no retry)

| System | Success Rate | Model | Memory Type |
|--------|-------------|-------|-------------|
| ExpeL | ~59% | GPT-4 | Experience extraction + insights |
| **CVX-Causal** | **43.3%** | **GPT-4o** | **Temporal vector retrieval** |
| ReAct (no memory) | ~20% | GPT-4o | None |
| **CVX-Causal** | **26.7%** | **GPT-4o-mini** | **Temporal vector retrieval** |
| **CVX-Causal** | **20.0%** | **qwen:7b** | **Temporal vector retrieval** |
| No memory | 3.3% | qwen:7b | None |

CVX at 43.3% does not yet beat ExpeL (59%), but ExpeL uses **experience extraction + insight learning** (a complex multi-stage pipeline) while CVX uses **pure retrieval** from a temporal index — no post-processing, no rule extraction.

### Online Learning (E7b) — Self-Improving Memory

The agent plays multiple rounds, adding its own experience to CVX after each round. Successful episodes get reward=1.0, failures get reward=0.0. Expert trajectories that were retrieved but led to failure get a 10% reward decay.

**Key insight — context format matters for small models**: Including expert observations in the prompt (E7 verbose, 722 chars) degraded performance vs compact action chains (E7b, ~200 chars). Small models lose performance with long contexts. The best format combines **abstract strategy templates** with **compact action chains**:

```
Strategy: Find object, take it, go to sinkbasin, clean it, go to target, put it.
Expert action sequences:
  [1] go to drawer 2 -> open drawer 2 -> take soapbar 1 -> go to sinkbasin 1
  [2] go to cabinet 1 -> open cabinet 1 -> take cloth 2 -> go to sinkbasin 1
```

| Round | Index size | E7 (verbose) | E7b (compact) |
|-------|-----------|-------------|---------------|
| 1 (expert only) | 4,542 | 6.7% | 6.7% |
| 2 (+own experience) | ~5,400 | 6.7% | 13.3% |
| 3 (+reward decay) | ~6,300 | 16.7% | **26.7%** |

**E7b Round 3 (26.7%) beats the E3 baseline (20%)** — online learning + compact strategy templates outperform static expert retrieval.

### Learning Curve & Memory Dynamics (10 rounds, qwen:7b)

Three variants tested over 10 rounds to understand how memory quality
evolves with accumulated experience:

```
Round:   1     2     3     4     5     6     7     8     9     10    Mean R4-10
E7c:    6.7  10.0  20.0  13.3  16.7  16.7  13.3  10.0  16.7  16.7   14.8%
E7d:   10.0  23.3  26.7  13.3  13.3  16.7  13.3  26.7  13.3  23.3   17.1%
E7e:   10.0  23.3  30.0  16.7  16.7  23.3  13.3  26.7  13.3  26.7   19.5%
```

| Variant | Approach | Peak | Mean R4-10 |
|---------|----------|------|-----------|
| **E7c** | All experience in index, blind decay (10%) | 20.0% | 14.8% |
| **E7d** | Wins-only index, blind decay (15%) | 26.7% | 17.1% |
| **E7e** | Wins-only index, **context-aware decay** (25%) | **30.0%** | **19.5%** |

### Three Discoveries

**1. Memory contamination degrades retrieval** (E7c)

Each round adds ~25 failed episodes and ~4 successful ones. By Round 10,
only 59% of memory episodes are successful. Failed experiences are
semantically similar to new queries (same task types) and pollute retrieval.
Fix: never add failed episodes to the retrieval index (E7d).

**2. Blind reward decay destroys useful experts** (E7d)

An expert trajectory for `clean soapbar` retrieved during a failed
`cool lettuce` game gets penalized — even though the expert was
irrelevant to that failure. After several rounds, good experts in
unrelated task types lose reward unfairly.

**3. Context-aware decay preserves memory quality** (E7e)

Decay only when: (a) expert task type matches the failed game's task type,
AND (b) the agent actually followed the expert's suggested action.
This protects cross-task experts and experts whose advice was ignored.

| Scenario | Blind decay | Context-aware |
|----------|-----------|--------------|
| Expert `clean` retrieved during failed `cool` game | -15% | **No decay** |
| Expert action retrieved but agent chose differently | -15% | **No decay** |
| Expert followed AND agent failed at same task type | -15% | **-25% decay** |

See [RFC-013 Part E](/chronos-vector/rfc/rfc-013) for the full analysis
and [RFC-013 Part F](/chronos-vector/rfc/rfc-013) for the integrated
active memory architecture that combines these findings.

### Other Experiments

**E1: Code Generation** (MBPP → HumanEval) — 77.8% pass@1 with episodic retrieval (qwen:7b)

**E2: ALFWorld Plan Quality** — 0.709 semantic similarity with episodic retrieval

**E4: Iterative Debugging** (APPS) — 28% → 31% with error-to-fix memory (qwen:7b)

---

## Research Roadmap

### Phase 1: Scale to Frontier Models — DONE

E5 (GPT-4o) and E6 (GPT-4o-mini) completed. CVX-Causal shows consistent 2× improvement across model scales. The 43.3% result with GPT-4o is the current best.

### Phase 1b: Online Learning — DONE

E7b shows 4× improvement across 3 rounds with compact strategy templates + online reward annotation. The learning curve (6.7% → 13.3% → 26.7%) suggests further rounds may improve. Next: study the saturation curve to understand when and why improvement plateaus.

### Phase 2: Learning Curve & Saturation Analysis (In Progress)

| Experiment | Approach | Question |
|-----------|---------|----------|
| **E7c** | E7b with 10+ rounds | Where does the learning curve saturate? |
| **E7c analysis** | Per-task-type breakdown across rounds | Which task types improve most? Which plateau? |
| **Transfer analysis** | Compare retrieved vs actual actions | Why does retrieval fail for specific task types? |

### Phase 3: Structural Extensions (Under Investigation)

Open question: can auxiliary structures (knowledge graphs, HMMs, Bayesian networks) improve the memory beyond what pure vector retrieval provides? See [RFC-012 Part D](/chronos-vector/rfc/rfc-012).

### Phase 4: Temporal Reasoning Benchmarks

**LongMemEval — Temporal Reasoning Subtask**

CVX's temporal features (timestamps, ordering, causal_search) directly address temporal reasoning:
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

### Completed

| Notebook | Model | Focus | Key Result |
|----------|-------|-------|------------|
| E1_episodic_coding | qwen:7b | Code gen with episodic retrieval | 77.8% HumanEval pass@1 |
| E2_episodic_alfworld | qwen:7b | Plan quality from episodic retrieval | 0.709 semantic similarity |
| E3_interactive_alfworld | qwen:7b | Step-by-step agent with CVX causal search | **3.3% → 20.0% (6×)** |
| E4_iterative_coding | qwen:7b | Debug retry with error memory | 28% → 31% |
| E5_alfworld_gpt4o | **GPT-4o** | ALFWorld with frontier model | **20.0% → 43.3% (2.2×)** |
| E6_alfworld_gpt4o_mini | **GPT-4o-mini** | Cost-effective scaling test | **13.3% → 26.7% (2.0×)** |
| E7_online_learning | qwen:7b | Online learning with observation context | 6.7% → 16.7% (verbose) |
| E7b | qwen:7b | Compact strategy + online learning | 6.7% → 26.7% (3 rounds) |
| E7c | qwen:7b | 10-round saturation study | Peak 20%, plateau 14.8% |
| E7d | qwen:7b | Clean memory (wins-only index) | Peak 26.7%, plateau 17.1% |
| **E7e** | **qwen:7b** | **Context-aware reward decay** | **Peak 30%, plateau 19.5%** |

### Planned

| Notebook | Focus | Target |
|----------|-------|--------|
| E7c | Learning curve saturation (10+ rounds) | Find plateau point |
| E8_longmemeval | Temporal reasoning evaluation | LongMemEval (ICLR 2025) |
| E9_mem2act | Memory-to-action grounding | Mem2ActBench (2025) |

---

## Related

- [Episodic Trace Memory — Concept](/chronos-vector/research/episodic-trace-memory)
- [Episodic Memory — Full Experimental Report](/chronos-vector/research/episodic-memory-experiments)
- [RFC-010: Temporal Graph Extension](/chronos-vector/rfc/rfc-010) (causal search infrastructure)
- [RFC-012 Part C: Architecture Gaps](/chronos-vector/rfc/rfc-012) (agent memory roadmap)
- [Episodic Memory Tutorial](/chronos-vector/tutorials/guides/episodic-memory) (code tutorial with synthetic data)
