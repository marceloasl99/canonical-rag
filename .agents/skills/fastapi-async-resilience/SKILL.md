---
name: fastapi-async-resilience
description: >-
  Audits FastAPI async backend applications for blocking I/O, event loop starvation,
  concurrency race conditions, thread pool offloading, timeout management, error middleware,
  and Pydantic validation resilience. Use when reviewing backend services, API endpoints,
  or asynchronous task execution.
---

# FastAPI Async & Concurrency Resilience Skill

This skill guides deep code reviews and performance audits for FastAPI asynchronous services, with particular emphasis on preventing event loop starvation when combining CPU-bound ML models and disk-bound databases.

## Core Review Areas

### 1. Event Loop Starvation & Blocking Calls
- **CPU-Bound Models (ONNX, Cross-Encoders)**: CPU inference functions (such as `rerank(...)`) are synchronous and computationally intensive. If called inside an `async def` endpoint without `asyncio.to_thread` or a `ThreadPoolExecutor`, they freeze the entire FastAPI event loop, blocking all concurrent HTTP requests.
- **SQLite / Disk I/O**: SQLite calls (`db.execute`, `fetchall`) perform blocking file I/O. In high-concurrency environments, audit whether database access should be wrapped in worker threads or an async driver (`aiosqlite`).
- **Synchronous HTTP Clients**: Ensure external API calls use `httpx.AsyncClient` or `aiohttp` instead of synchronous `requests` or blocking socket calls.

### 2. Connection Pooling & Resource Cleanup
- **HTTP Client Lifecycles**: Verify that `httpx.AsyncClient` instances are reused across requests (e.g. via FastAPI app `lifespan` context manager) rather than instantiated per-request, preventing socket exhaustion.
- **Database Connection Leaks**: Check that SQLite connections or cursor contexts use `with sqlite3.connect(...) as conn:` or deterministic `try ... finally: conn.close()` blocks.

### 3. Request Validation & Error Handling
- **Pydantic Model Constraints**: Validate that input fields have reasonable limits (e.g. `query: str = Field(..., max_length=2000)` to prevent DoS memory allocations).
- **Graceful Error Responses**: Ensure unhandled exceptions return structured JSON payloads (`{"error": "...", "detail": "..."}`) rather than generic 500 HTML error pages.
- **Cancellation & Timeouts**: Verify that long-running operations specify request timeouts (`httpx.Timeout(60.0)`) and respect client disconnection signals.
