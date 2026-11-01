---
name: rag-retrieval-audit
description: >-
  Audits and evaluates two-stage RAG pipelines, including lexical retrieval (SQLite FTS5),
  cross-encoder reranking (FlashRank ONNX), query normalization, semantic caching,
  sentence-level context pruning, and factual grounding verification. Use when reviewing
  RAG architectures, chunking strategies, ranking quality, or token pruning logic.
---

# RAG Retrieval & Grounding Audit Skill

This skill provides a systematic procedure for auditing, benchmarking, and identifying bugs and optimization opportunities in two-stage Retrieval-Augmented Generation (RAG) engines.

## Core Audit Procedures

### 1. Lexical Retrieval Audit (SQLite FTS5)
- **Tokenizer Configuration**: Check for `unicode61 remove_diacritics 2` or appropriate language-specific tokenization to ensure accents and case folding don't prevent match hits.
- **Query Sanitization**: Ensure user queries passed to FTS5 MATCH operators escape syntax operators (`"`, `*`, `NEAR`, `OR`, `NOT`, `-`) to prevent query syntax crashes while preserving intentional phrase quoting.
- **BM25 Weights**: Verify if title, headings, and body chunks have appropriate column weights (e.g., `bm25(title_weight, body_weight)`).
- **Proximity Operators**: Inspect proximity query thresholds (`NEAR(..., N)`). If N is too narrow (< 5 words), relevant technical phrases with intervening words will be dropped.

### 2. Cross-Encoder & Reranking Audit (FlashRank / ONNX)
- **Pairwise Cross-Attention**: Verify query-document pairs are fed properly without truncation of critical terms.
- **Threshold & Cutoff Logic**: Check whether passages below a minimum relevance score (e.g. < 0.15 or negative logits depending on the model) are discarded or logged.
- **CPU Threading & ONNX Performance**: Ensure ONNX Runtime session options (`intra_op_num_threads`, `inter_op_num_threads`) do not block the async event loop during inference.

### 3. Extractive Context Pruning & Dossier Construction
- **Sentence Segmentation**: Verify that regex or tokenizer splitting handles punctuation abbreviations (e.g., `art.`, `pág.`, `ex.`, `v.g.`, `1.5`) without premature truncation.
- **Informational Density**: Check how high-density technical sentences are identified vs generic filler (acknowledgments, table of contents lines, page headers).
- **Context Boundaries**: Confirm that source attribution (filename, page number, chunk index) is preserved alongside pruned sentences.

### 4. Semantic Cache Integrity
- **Canonical Hash Generation**: Check that whitespace trimming, lowercase conversion, and stopword sorting produce deterministic SHA-256 keys.
- **TTL and Invalidation**: Verify how cache entries are updated or evicted when underlying documents change.
- **Zero-Token Verification**: Ensure cache hits bypass external LLM calls entirely.
