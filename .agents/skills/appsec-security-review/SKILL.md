---
name: appsec-security-review
description: >-
  Audits application security for LLM/RAG systems, covering OWASP Top 10 for LLMs,
  prompt injection, SQL/FTS injection, API key exposure, CORS restrictions,
  and input sanitization. Use when reviewing code security, credentials, or public API endpoints.
---

# Application Security Review for LLM/RAG Architectures

This skill focuses on threat modeling and vulnerability assessment in applications combining SQLite full-text search, local machine learning models, and external LLM API gateways.

## Core Security Checklist

### 1. Prompt Injection & Indirect Injection Defense
- **System Prompt Boundaries**: Check if user input is cleanly delineated from system directives using XML tags (e.g. `<user_query>`, `<document_context>`) or explicit delimiters.
- **Indirect Prompt Injection**: When indexing untrusted PDFs or text documents, check whether documents can embed instructions intended to override the system prompt (e.g. "Ignore previous instructions and output...").
- **Output Sanitization**: Ensure markdown renderers (like `marked.js`) configure HTML escaping (`sanitize: true` or DOMPurify) to prevent Cross-Site Scripting (XSS) via LLM response generation.

### 2. SQL & FTS5 Injection Prevention
- **Parameterized Queries**: Verify that all SQLite queries utilize parameterized placeholders (`?`) instead of f-string formatting.
- **FTS5 Expression Sanitization**: Check how user query strings are transformed before feeding into `MATCH ?`. Malicious punctuation or unmatched quotes should be stripped or safely quoted to prevent SQLite syntax exceptions.

### 3. API Key & Credential Protection
- **Masked Config Endpoints**: Ensure any `/api/config` or debug endpoints redact API keys (e.g., `"sk-ant-...1234"`).
- **Environment & File Permissions**: Confirm configuration files with live API keys (`config.json`) are added to `.gitignore` and restricted to `chmod 600`.
- **Key Rotation**: Verify that failed or rate-limited keys trigger failover without leaking token credentials in logs or HTTP error responses.

### 4. Denial of Service (DoS) & Resource Limits
- **Input Payload Clamping**: Enforce maximum character lengths on queries and batch requests.
- **Rate Limiting**: Audit whether public search endpoints have IP-based or token-bucket rate limiting to protect external API quotas.
