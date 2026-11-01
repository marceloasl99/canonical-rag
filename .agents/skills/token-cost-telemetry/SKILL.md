---
name: token-cost-telemetry
description: >-
  Audits LLM token accounting, ephemeral prompt caching efficiency, token reduction estimators,
  hardware telemetry monitoring (CPU, RAM, Disk), and cost-optimization gateways.
  Use when analyzing LLM consumption, token metrics, cache accounting, or system telemetry.
---

# Token Economics & System Telemetry Skill

This skill provides methodologies for validating token reduction metrics, ephemeral prompt caching configurations, hardware telemetry sampling, and cost attribution in multi-provider LLM gateways.

## Core Audit Procedures

### 1. Token Counting & Pruning Verification
- **Token Estimation Models**: Check if token counts are estimated via character heuristics (`len(text) // 4`) or standard tokenizers (tiktoken, huggingface tokenizers). Note where heuristic drift occurs on non-English text.
- **Context Reduction Delta**: Verify that `tokens_saved_by_pruning` calculates the exact difference between raw retrieved candidate chunk volume and the post-pruning context passed to the LLM.
- **Cache Hit Accounting**: Validate that semantic cache hits increment the `tokens_saved_total` by the full cost of the cached response (input context + output completion), reflecting true zero-token economy.

### 2. Multi-Provider Ephemeral Prompt Caching
- **Anthropic Claude Cache Headers**: Verify the presence of `"cache_control": {"type": "ephemeral"}` on static system prompts and dense dossiers (> 1024 tokens) to trigger Anthropic's 90% prompt caching discount.
- **Google Gemini Context Caching**: Check whether repetitive system directives or corpus preambles leverage Gemini's context caching endpoints when prompt sizes exceed threshold limits.

### 3. Real-Time Hardware Telemetry Sampling
- **Non-Blocking Sampling**: Confirm that `psutil.cpu_percent()` is called with `interval=None` (non-blocking) rather than `interval=1.0` (which halts execution for 1 second).
- **Process vs System Metrics**: Distinguish between the process-level memory footprint (`psutil.Process().memory_info().rss`) and overall host RAM (`psutil.virtual_memory()`).
- **Telemetry Polling Cadence**: Ensure frontend telemetry polling (e.g. 2s interval) does not create connection churn or memory leaks in long-running browser tabs.
