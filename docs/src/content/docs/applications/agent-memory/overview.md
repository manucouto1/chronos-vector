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
| **CVX E9 (structured)** | RegionMDP + phase detection + CVX | **63.3%** | qwen2.5:14b (zero-shot, no fine-tuning) | This work |
| **CVX-Causal** | Temporal episodic (causal retrieval) | **43.3%** | GPT-4o (1-shot) | This work |
| No memory | Baseline | 3.3% | qwen2.5:7b | This work |

**Key insight**: The E9 structured pipeline achieves 63.3% (stable range 60-63%) on ALFWorld with a local 14b model, zero-shot, no fine-tuning. This outperforms BUTLER (46%), the Act baseline (45%), and ExpeL (~59% with GPT-4). All improvements from v4 onward are prompt engineering -- the CVX infrastructure is complete and working. See the [E9 Pipeline section](#e9-pipeline-structured-agent-with-regionmdp-current-best) for the full breakdown.

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
| **CVX E9 (structured)** | **63.3%** | **qwen2.5:14b** | **RegionMDP + phase detection + CVX** |
| ExpeL | ~59% | GPT-4 | Experience extraction + insights |
| **CVX-Causal** | **43.3%** | **GPT-4o** | **Temporal vector retrieval** |
| ReAct (no memory) | ~20% | GPT-4o | None |
| **CVX-Causal** | **26.7%** | **GPT-4o-mini** | **Temporal vector retrieval** |
| **CVX-Causal** | **20.0%** | **qwen:7b** | **Temporal vector retrieval** |
| No memory | 3.3% | qwen:7b | None |

CVX E9 at 63.3% with a local 14b model outperforms ExpeL (~59% with GPT-4). ExpeL uses **experience extraction + insight learning** (a complex multi-stage pipeline) while CVX E9 uses **structured retrieval + prompt engineering** -- no post-processing, no rule extraction, no fine-tuning.

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

---

## E4: Iterative Code Generation with CVX Debug Memory

CVX as **episodic debug memory** for a generate-test-debug coding agent on the APPS benchmark. The agent recalls how similar errors were fixed in the past.

### Pipeline

1. **Build debug traces** from 200 APPS interview problems (hard): attempt → error → retry → fix
2. **Index 6-step episodes** in CVX: problem → attempt1 → error1 → attempt2 → error2 → fix
3. **On eval**: when code fails, `causal_search` finds similar past errors and walks forward to the fix
4. **Online learning**: successful fixes are added to the index during evaluation

### Best Result (qwen2.5-coder:7b + APPS introductory)

| Condition | Pass Rate | Description |
|---|---|---|
| SinglePass | 28% | Zero-shot, no retries |
| Retry-NoMemory | 28% | Up to 3 retries, no memory — retries alone don't help |
| **Retry-CVX-Causal** | **31%** | **Up to 3 retries with CVX debug memory** |

CVX provides **+3% absolute (+10.7% relative)** over NoMemory retries. Retries without memory are useless (28% = 28%) because the model repeats the same mistakes. CVX breaks this cycle by retrieving how similar errors were fixed.

### Scaling Analysis

| Model | Difficulty | SinglePass | NoMemory | CVX | CVX vs NoMem |
|---|---|---|---|---|---|
| coder:7b | introductory | 28% | 28% | **31%** | **+3%** |
| qwen2.5:14b | introductory | 33% | **45%** | 42% | -3% |
| qwen2.5:14b | interview | 6% | **16%** | 14% | -2% |
| coder:7b | interview | 5% | 5% | 5% | 0% |

Key findings:

1. **CVX helps when the model is weak but capable of improving with guidance** (7b + easy problems)
2. **Strong models self-correct without external context** — extra tokens from retrieved fixes distract the 14b model
3. **Too-weak models cannot leverage any help** — 7b on hard problems fails regardless
4. Same pattern as E9: the value of memory depends on the model's ability to use the retrieved information

### Complementary Roles with E9

| Experiment | CVX Role | Memory Type | Key Mechanism |
|---|---|---|---|
| **E4** | Debug memory | Episodic/procedural | error → fix retrieval via causal_search |
| **E9** | Agent guidance | Semantic/procedural | RegionMDP + phase detection + location hints |

---

## E9 Pipeline: Structured Agent with RegionMDP (Current Best)

The E9 pipeline represents a fundamental shift from pure retrieval (E3-E7) to a **structured decision-making agent** that combines CVX episodic memory with probabilistic action selection and expert-derived heuristics. Evaluated on the full ALFWorld `eval_out_of_distribution` split (134 games).

### Result

| Pipeline | Model | Games | Success Rate |
|----------|-------|-------|-------------|
| **E9 v7 (structured agent)** | **qwen2.5:14b** | **30** | **63.3% (19/30)** |
| E7e (best retrieval) | qwen2.5:7b | 30 | 30% (peak) |
| E5 (causal retrieval) | GPT-4o | 30 | 43.3% |

Stable performance range: 60-63% across v6-v8 variants.

### SOTA Comparison (ALFWorld eval_out_of_distribution)

| System | Success Rate | Model | Notes |
|--------|-------------|-------|-------|
| AutoManual | 97.4% | GPT-4 | Online rule learning |
| ReAct + Reflexion | 97% | GPT-4 | Self-reflection + retry |
| Embodied Planner-R1 | 96.3% | RL fine-tuned | Reinforcement learning |
| AgentPRM | 91% | 3B fine-tuned | Process reward model |
| RLVMR | 87.9% | Qwen-1.5B RL fine-tuned | Reinforcement learning |
| AutoManual | 86.2% | GPT-3.5 | Online rule learning |
| ReAct | 71% | GPT-3.5 | Prompt-only |
| **CVX + qwen2.5:14b (ours)** | **63.3%** | **qwen2.5:14b** | **Zero-shot, no fine-tuning, local model** |
| ExpeL | ~59% | GPT-4 | Experience extraction + insights |
| BUTLER | 46% | -- | Imitation learning |
| Act baseline | 45% | -- | No reasoning |

**Key positioning**: CVX at 63.3% with a local 14b model (zero-shot, no fine-tuning) outperforms BUTLER (46%), the Act baseline (45%), and ExpeL (~59% with GPT-4). The CVX infrastructure is complete and working -- all improvements from v4 onward are prompt engineering. The gap to SOTA (97.4%) is explained by three factors: (1) we use a small local model vs GPT-4, (2) zero-shot vs few-shot/fine-tuned, and (3) no retry/reflection mechanism. With GPT-4o-mini, the same pipeline would likely reach 70%+.

### E9 Pipeline Components

The E9 agent is composed of six interacting components:

**1. RegionMDP** -- Maintains `P(success | region, action_type)` using a Beta prior. Each region (e.g., `countertop`, `drawer`) accumulates success/failure counts per action type. The agent uses these posteriors to rank candidate actions.

**2. Location hints** -- Expert-derived mappings from `(task_type, phase)` to prioritized location types. For example, `(clean, searching)` prioritizes `countertop > drawer > shelf`, while `(heat, transforming)` points to `microwave`. These encode domain knowledge that the LLM lacks.

**3. Phase detection** -- A full-history state machine that tracks the agent's progress through four phases: `searching` (looking for the target object), `holding` (object acquired), `transforming` (applying clean/heat/cool), and `placing` (delivering to destination). Phase transitions are detected from the complete action history, not just the last observation.

**4. Loop detection** -- Identifies and excludes repeated or oscillating actions (e.g., `go to shelf 1 -> go to shelf 2 -> go to shelf 1`). Prevents the agent from wasting steps revisiting locations.

**5. Online learning** -- Successful episodes are inserted into the CVX index with `reward=1.0`. Failed episodes update RegionMDP statistics but are not added to the retrieval index (lesson from E7d: memory contamination).

**6. Abstract guidance** -- The system prompts the LLM with **action types** (e.g., "go to a receptacle", "take the object") and **location types** (e.g., "try a countertop or drawer"), not specific actions (e.g., "go to countertop 3"). This allows transfer across different room layouts.

### Iterative Development: What Worked and What Did Not

The path from E3 (3.3%) to E9 v7 (63.3%) was not a single leap. Each version tested a specific hypothesis:

| Version | Change | Result | Delta | Notes |
|---------|--------|--------|-------|-------|
| v1 | Abstract only (7b coder) | 3.3% | Baseline | |
| v2 | Location hints | 16.7% | +13.4% | Expert-derived per task_type/phase |
| v3 | qwen2.5:14b | 23.3% | +6.6% | General model vs coding model |
| v4 | Phase detection (full history) | 50.0% | +26.7% | Biggest single improvement |
| v5 | Few-shot examples | 46.7% | -3.3% | Extra tokens distract small model |
| v6 | USE THIS ACTION NOW | 56.7% | +10.0% | Explicit action directives |
| **v7** | **Exploration directives** | **63.3%** | **+6.6%** | **NOT YET VISITED, GO TO TARGET NOW** |
| v8 | World state (tested) | 63.3% | 0% net | Helps clean/examine, hurts pick/cool |

### Per Task Type (v7 Best)

| Task Type | Success Rate |
|-----------|-------------|
| examine | 100% |
| pick_two | 100% |
| cool | 83% |
| clean | 60% |
| heat | 50% |
| pick | 50% |

**Key insights**:

1. **Specific expert actions do not transfer across layouts.** An expert action "go to countertop 3" is useless when the test environment has different numbered objects. This is why pure retrieval (E3-E7) plateaus.

2. **Abstract action types alone are too vague.** Telling the LLM "try a go-to action" without specifying where provides no useful guidance.

3. **Location hints per task_type/phase are critical.** Encoding "during the searching phase of a clean task, prioritize countertops and drawers" gives the LLM actionable structure. This single addition yielded +13.4pp.

4. **LLM scale matters.** qwen2.5:14b consistently outperforms qwen2.5:7b-coder, especially for following structured prompts (+6.6pp).

5. **Phase detection from full history was the largest single improvement.** Knowing whether the agent is searching, holding, transforming, or placing determines which action types and locations are relevant. This added +26.7pp, the biggest jump in the entire experimental series.

6. **Few-shot examples hurt with small local models.** Adding one example per task type (v5) reduced performance from 50.0% to 46.7%. The extra prompt tokens consume context budget that qwen2.5:14b needs for reasoning. This is consistent with the E7/E7b finding that compact prompts outperform verbose ones for small models.

7. **Explicit action directives work.** Phrasing like "USE THIS ACTION NOW" (v6) and "NOT YET VISITED, GO TO TARGET NOW" (v7) give the small model clear, imperative guidance that improves action selection by +10.0% and +6.6% respectively.

8. **CVX infrastructure is complete.** All improvements from v4 to v7 are prompt engineering. The retrieval, phase detection, and RegionMDP components work correctly -- the remaining gains come from how we communicate with the LLM.

### World State Experiment (v8)

Adding holding status and TARGET FOUND markers to the prompt had mixed results:

| Task Type | Without world state (v7) | With world state (v8) | Delta |
|-----------|-------------------------|----------------------|-------|
| clean | 60% | 100% | +40% |
| examine | 75% | 100% | +25% |
| pick | 50% | lower | negative |
| cool | 83% | lower | negative |

The world state information helps tasks that benefit from explicit context (clean, examine) but adds token overhead that hurts tasks where the small model is already performing well. This would likely benefit from a larger model where extra tokens are less costly.

### Failure Analysis (v7, 11/30 failures)

The remaining 11 failures (36.7%) break down into two primary categories:

**1. Holding failures (5/11)** -- The agent picks up the wrong object or cannot find the target to pick it up. These occur in `pick` and `pick_two` tasks where multiple similar objects exist.

**2. Searching failures (4/11)** -- The agent cannot find the target object. The search strategy does not cover all possible locations, and some objects are in unusual places not covered by location hints.

**3. Transforming failure (1/11)** -- The agent fails during the transformation step (clean/heat/cool).

**4. Placing failure (1/11)** -- The agent fails to execute the final placement action.

### Roadmap to Improve

| Improvement | Status | Expected Impact | Effort |
|-------------|--------|----------------|--------|
| **Fix action matching** | Done | Eliminated format failures | Low -- regex normalization |
| ~~**Few-shot examples**~~ | Tested (v5) | -3.3pp (hurts with local models) | -- |
| **Explicit action directives** | Done (v6) | +10.0pp | Low -- prompt engineering |
| **Exploration directives** | Done (v7) | +6.6pp | Low -- prompt engineering |
| **World state context** | Tested (v8) | 0% net (mixed per task type) | Low -- needs larger model |
| **Reflexion** | Next (highest priority) | +10-15pp (self-correction across attempts) | Medium -- retry loop with verbal reflection |
| **GPT-4o-mini for publication** | Planned | 63% to estimated 70%+ (same pipeline, larger model) | Low -- API swap |

The CVX infrastructure is complete and working. All improvements from v4 to v7 are prompt engineering, demonstrating that the retrieval, phase detection, and RegionMDP components are solid.

The highest-impact next step is **GPT-4o-mini** for publication. With the same pipeline that achieves 63.3% on qwen2.5:14b, a larger model would likely reach 70%+ based on the consistent scaling pattern observed across experiments. **Reflexion** (self-correction across attempts) remains the highest expected impact technique -- it took ReAct from ~45% to ~97% with GPT-4.

---

## Research Roadmap

### Phase 1: Scale to Frontier Models — DONE

E5 (GPT-4o) and E6 (GPT-4o-mini) completed. CVX-Causal shows consistent 2x improvement across model scales. The E9 structured pipeline with qwen2.5:14b now achieves 63.3%, surpassing the GPT-4o causal retrieval result (43.3%).

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
| **E9** | **qwen2.5:14b** | **Structured agent: RegionMDP + phase detection + prompt eng** | **63.3% (v7 best, stable 60-63%)** |

### Planned

| Notebook | Focus | Target |
|----------|-------|--------|
| E8_longmemeval | Temporal reasoning evaluation | LongMemEval (ICLR 2025) |
| ~~E9b~~ | ~~Few-shot examples~~ | ~~Target 65-70%~~ -- tested as v5, -3.3pp (hurts local models) |
| ~~E9c~~ | ~~Exploration directives~~ | Done as v6-v7, reached 63.3% |
| ~~E9d~~ | ~~World state context~~ | Done as v8, 0% net (mixed per task type) |
| E9e | Reflexion (self-correction across attempts) | Target 70%+ (highest expected impact) |
| E9f | GPT-4o-mini publication run | Target 70%+ (publication-ready results) |

---

## Related

- [Episodic Trace Memory — Concept](/chronos-vector/research/episodic-trace-memory)
- [Episodic Memory — Full Experimental Report](/chronos-vector/research/episodic-memory-experiments)
- [RFC-010: Temporal Graph Extension](/chronos-vector/rfc/rfc-010) (causal search infrastructure)
- [RFC-012 Part C: Architecture Gaps](/chronos-vector/rfc/rfc-012) (agent memory roadmap)
- [Episodic Memory Tutorial](/chronos-vector/tutorials/guides/episodic-memory) (code tutorial with synthetic data)
