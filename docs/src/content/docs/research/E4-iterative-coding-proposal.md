---
title: "E4: Iterative Code Generation with CVX Memory (Proposal)"
description: "Making code generation interactive — CVX as prefrontal cortex for a generate-test-debug agent"
---

## Why E1 Failed and How to Fix It

E1 showed that static retrieval (find similar problems → inject solutions) doesn't differentiate CVX from flat cosine for code generation. The reason: **code generation was treated as a single-step task** — the query is always "here's a problem, generate code." There's no evolving state.

E3 showed that CVX shines when the **query changes at each step** based on real feedback. The fix for code generation is to make it **interactive**: a generate-test-debug loop where each iteration produces a new state (the error message, the failing test, the partial solution) that CVX can match against.

## The Iterative Agent Loop

```
┌─────────────────────────────────────────────────────────────┐
│  1. LLM generates code (attempt N)                          │
│  2. Execute tests → pass? → done ✓                          │
│  3. If fail: embed(error_message + failing_test + code)     │
│  4. CVX.search(error_embedding) → find similar past errors  │
│  5. Extract continuation: "this error was fixed by..."      │
│  6. LLM(code + error + CVX fix suggestions) → attempt N+1  │
│  7. → goto 2                                                │
└─────────────────────────────────────────────────────────────┘
```

## Episode Structure for Code Debug Traces

Each **debug episode** is a multi-step trajectory stored in CVX:

| Step | Embedding | Timestamp | Content |
|------|-----------|-----------|---------|
| 0 | problem description | t₀ | "Write a function that..." |
| 1 | attempt 1 (code) | t₁ | Generated code |
| 2 | error from attempt 1 | t₂ | "TypeError: list indices must be integers" |
| 3 | fix applied | t₃ | Diff or corrected code |
| 4 | attempt 2 (code) | t₄ | Updated code |
| 5 | pass ✓ | t₅ | Final working solution |

The **reward** encodes whether the episode ultimately succeeded. Failed debug traces (never resolved) get reward=0; successful ones get reward=1.

## What CVX Queries Look Like at Each Step

### Step 1: Initial generation (same as E1)
- Query: problem description
- CVX returns: similar problems and their solutions
- This is what E1 already does — baseline

### Step 2: First error (NEW — this is where CVX differentiates)
- Query: `embed("TypeError: list indices must be integers, not str" + code_context)`
- CVX searches ALL steps, finds similar errors in past debug traces
- Returns: "when this error occurred before, the fix was to cast index to int"
- **This is impossible with flat cosine** — it only has problem embeddings, not error embeddings

### Step 3: Subsequent errors
- Query: `embed(new_error + previous_fix_attempt)`
- CVX tracks the **trajectory of debugging** — if the agent is going in circles (same error repeatedly), CVX can detect this via `velocity()` and suggest a different approach

### Step 4: Stuck detection
- Use `drift()` between consecutive attempts — if the code isn't changing (low drift), the agent is stuck
- Use `hurst_exponent()` on the debug trajectory — anti-persistent trajectories (H < 0.5) indicate productive exploration, persistent ones (H > 0.5) indicate the agent is repeating the same mistakes

## Training Data: Where Do Debug Traces Come From?

### Option A: Synthetic from MBPP/HumanEval
1. Run the LLM on all MBPP problems with T=0.8 (intentionally imperfect)
2. Capture the errors when solutions fail
3. Fix each error (either by the LLM with feedback or by using the known solution)
4. Store the full trace: problem → bad code → error → fix → good code

### Option B: Real debug traces from development
- Instrument a coding workflow to capture edit → test → error → fix cycles
- Each commit-test cycle is a step in the episode
- Git history provides natural temporal ordering

### Option C: Multi-model ensemble
- Generate solutions from multiple models/temperatures
- Errors from weaker attempts + fixes from stronger models
- Creates diverse debug traces covering many error patterns

## CVX Features Exploited

| Feature | Usage | Why flat cosine can't |
|---------|-------|-----------------------|
| `search()` across all steps | Match on errors, not just problems | Flat store only has problem embeddings |
| Episode encoding | Group problem → error → fix as one trajectory | No episode concept |
| Timestamp ordering | Know that the fix came AFTER the error | No ordering |
| Continuation extraction | "This error was followed by this fix" | Can't extract "what came next" |
| `velocity()` | Detect productive vs stuck debugging | No trajectory concept |
| `drift()` | Measure if attempts are converging | No temporal structure |
| `detect_changepoints()` | Find where the debugging approach pivoted | No sequence |
| Reward filtering | Only retrieve from successful debug traces | Post-hoc only |

## Experimental Design

### Benchmark
- **MBPP** (384 problems) for building debug traces
- **HumanEval** (164 problems) for evaluation
- **LiveCodeBench** (if available) for harder problems where multi-step debugging matters more

### Conditions
| Condition | Description |
|-----------|-------------|
| NoMemory | Single-pass generation, no retry |
| Retry-NoMemory | Generate → test → retry with error (no CVX) |
| Retry-FlatCosine | Retry + flat cosine retrieval on error text |
| **Retry-CVX-Causal** | Retry + CVX causal retrieval (match error, return fix continuation) |

### Metrics
- **pass@1**: Single generation (baseline)
- **pass@1 after k retries**: With debug loop (primary metric)
- **Mean retries to pass**: Efficiency of debugging
- **Error diversity**: Does CVX help the agent try different fixes (not repeat)?

### Expected Results

The hypothesis is that CVX-Causal will differentiate most on **hard problems** that require debugging:
- Easy problems: all conditions pass on first attempt → no difference
- Medium problems: Retry-NoMemory may fix with generic error feedback
- **Hard problems**: Only CVX-Causal can retrieve specific fix patterns from similar past errors

## Implementation Plan

1. **Build debug trace corpus** (Option A): Run LLM on MBPP with T=0.8, capture traces
2. **Index in CVX**: Each trace is an episode with problem/attempt/error/fix steps
3. **Implement retry loop**: generate → test → embed error → search CVX → retry
4. **Evaluate on HumanEval**: Compare pass@k across conditions
5. **Analyze**: Which error types benefit most from CVX memory?

## Connection to Broader Vision

This positions CVX as a **prefrontal cortex** for coding agents:

- **Working memory**: Current problem + recent attempts (conversation context)
- **Episodic memory (CVX)**: Past debug experiences — "I've seen this error before, here's what worked"
- **Semantic memory**: General coding knowledge (LLM weights)

The iterative debug loop mirrors human problem-solving: try → fail → recall similar past failures → apply learned fix → retry. CVX provides the "recall" step that transforms a stateless LLM into a learning agent.
