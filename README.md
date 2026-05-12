# Forge Phase 2

Forge provides a fully local coding-agent backend with a production-oriented inference layer and a repository intelligence engine for autonomous codebase understanding.

## Features

- Fully local serving with `vLLM`
- OpenAI-compatible `POST /v1/chat/completions`
- Streaming token responses via Server-Sent Events
- Structured request tracing and model registry
- Centralized configuration with environment variables
- Async request handling and engine lifecycle management
- Repository scanning with incremental indexing support
- Tree-sitter AST extraction for Python, TypeScript, JavaScript, Go, and Rust
- Symbol graph construction with `networkx`
- Local embeddings with `BAAI/bge-small-en-v1.5`
- Local vector storage with Qdrant
- Hybrid retrieval, context building, and planning primitives
- Clean separation between API, config, core utilities, and LLM runtime
- Prepared boundaries for future orchestration, memory, and repo intelligence layers

## Structure

```text
backend/
  api/         HTTP routes and request/response schemas
  config/      Centralized settings
  core/        Logging and error handling
  llm/         vLLM engine lifecycle and chat service
  repointel/   Repository intelligence engine
  app.py       FastAPI application entrypoint
tests/
  test_chat_api.py
```

## Run

```bash
cp .env.example .env
.venv/bin/pip install -e '.[dev]'
.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

## Test

```bash
.venv/bin/pytest
```

## Example Requests

Non-streaming:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-coder",
    "agent_id": "planner-agent",
    "messages": [
      {"role": "system", "content": "You are a coding assistant."},
      {"role": "user", "content": "Write a Python function to merge two sorted lists."}
    ],
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 512
  }'
```

Streaming:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -N \
  -d '{
    "model": "deepseek-coder",
    "messages": [
      {"role": "user", "content": "Write a Rust function that validates a JWT."}
    ],
    "stream": true
  }'
```

Repository intelligence benchmark:

```bash
.venv/bin/python benchmarks/repointel_benchmark.py /path/to/repo "where is auth middleware initialized?"
```

Repository intelligence verification:

```bash
.venv/bin/python scripts/verify_repo_intelligence.py --repo-root /path/to/repo
```

Repository intelligence smoke test:

```bash
.venv/bin/python scripts/smoke_test_repo_intel.py --repo-root /path/to/repo
```

Multi-agent runtime smoke test:

```bash
.venv/bin/python scripts/smoke_test_multi_agent_runtime.py --repo-root /path/to/repo
```

Multi-agent runtime benchmark:

```bash
.venv/bin/python benchmarks/multi_agent_runtime_benchmark.py --task-count 10
```

Multi-agent runtime demo:

```bash
.venv/bin/python scripts/run_orchestration_demo.py
```

Multi-agent runtime architecture:

`backend/runtime/agents.py`: base agent abstraction, identity, capabilities, lifecycle hooks, structured execution.
`backend/runtime/messages.py`: typed inter-agent protocol payloads.
`backend/runtime/tasks.py`: task IDs, dependencies, retries, priorities, timestamps, status tracking.
`backend/runtime/events.py`: async event bus and runtime event types.
`backend/runtime/context.py`: shared context snapshots and token-aware context packaging with repository-intelligence integration.
`backend/runtime/patches.py`: structured patch and risk representation.
`backend/runtime/results.py`: candidate aggregation for future judge agents.
`backend/runtime/diagnostics.py`: agent/task/concurrency metrics.
`backend/runtime/orchestrator.py`: async scheduler, dependency resolution, cancellation, timeout handling, result/event aggregation.
`backend/runtime/mock_agents.py`: mock planner/coder/critic/context agents for orchestration verification.
`backend/runtime/api.py`: developer-facing runtime facade.

Canonical developer interfaces:

```python
import asyncio

from backend.repointel import (
    ContextBuilder,
    EmbeddingService,
    PlanningLayer,
    RepositoryIntelligenceEngine,
    TreeSitterParser,
)


async def main() -> None:
    parser = TreeSitterParser()
    print(parser.validate_languages())

    embeddings = EmbeddingService()
    print(await embeddings.validate_startup())

    engine = RepositoryIntelligenceEngine()
    print(await engine.verify_runtime())

    # After indexing a repository, the public subsystem interfaces are reusable.
    await engine.index_repository("/path/to/repo")
    context = await engine.context_builder.build("where is auth middleware initialized?")
    plan = await engine.planner.plan("add caching to auth middleware")
    print(context.related_files)
    print(plan.impacted_files)
    await engine.shutdown()


asyncio.run(main())
```
