# provider-contracts

> Shared abstract provider interfaces, Pydantic DTOs, and reusable concrete implementations for Python backend projects.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

---

## What is this?

**provider-contracts** is a Python package that defines a standard set of **abstract base classes (ABCs)** and **Pydantic DTOs** for common backend capabilities — LLMs, caching, vector stores, embeddings, PII detection, and more.

Instead of coupling your application to a specific vendor SDK, you code against these contracts. Swap providers by changing a single import — no application logic changes required.

```
┌─────────────────────────────────────────────────────────┐
│                   Your Application                      │
│                                                         │
│   from provider_contracts.llm import AbstractLLMProvider │
│   from provider_contracts.cache import AbstractCache... │
│                         │                               │
│                    (depends on ABCs only)                │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │  OpenAI    │ │  Ollama    │ │  Anthropic  │
   │  Provider  │ │  Provider  │ │  Provider   │
   └────────────┘ └────────────┘ └────────────┘
```

### Why?

- **Vendor independence** — switch from OpenAI to Ollama (or Anthropic, Gemini, Azure OpenAI) without touching business logic
- **Testability** — every provider ships a `Mock` or `InMemory` implementation for unit tests
- **Type safety** — `mypy --strict` compatible; fully typed ABCs and DTOs
- **Minimal core** — the base package only depends on `pydantic`; heavy SDKs are optional extras

---

## Installation

Requires **Python 3.11+**. Install with `uv` or `pip`:

```bash
# Core only (ABCs + DTOs + mock/in-memory providers)
uv pip install provider-contracts

# With specific backends
uv pip install "provider-contracts[openai,redis]"

# Everything
uv pip install "provider-contracts[all]"

# Editable install for development
uv pip install -e ".[all,dev]"
```

### Available Extras

| Extra | Providers unlocked | Key dependency |
|-------|-------------------|----------------|
| `openai` | OpenAI LLM, OpenAI Embeddings, Whisper API STT | `openai` |
| `anthropic` | Anthropic Claude LLM | `anthropic` |
| `gemini` | Google Gemini LLM | `google-genai` |
| `ollama` | Ollama local LLM | `httpx` |
| `redis` | Redis cache | `redis` |
| `r2` | Cloudflare R2 storage | `boto3` |
| `pgvector` | PostgreSQL vector store | `sqlalchemy`, `asyncpg`, `pgvector` |
| `elevenlabs` | ElevenLabs TTS | `httpx` |
| `whisper-local` | Local faster-whisper STT | `faster-whisper` |
| `webrtc-vad` | WebRTC VAD | `webrtcvad` |
| `celery` | Celery + Redis queue | `celery` |
| `qdrant` | Qdrant vector store | `qdrant-client` |
| `bge` | BGE / Sentence-Transformers embeddings | `sentence-transformers` |
| `presidio` | Microsoft Presidio PII detection | `presidio-analyzer`, `presidio-anonymizer` |
| `structlog` | Structured logging telemetry | `structlog` |
| `reranker` | Cross-encoder reranker | `sentence-transformers` |
| `all` | Everything above | — |
| `dev` | pytest, mypy, ruff | — |

---

## Quick Start

### 1. Import ABCs and DTOs (always available)

```python
from provider_contracts.llm import AbstractLLMProvider, LLMMessage, LLMResponse
from provider_contracts.cache import AbstractCacheProvider
from provider_contracts.embedding import AbstractEmbeddingProvider, EmbeddingResult
```

### 2. Use mock providers for testing

```python
from provider_contracts.llm.mock import MockLLMProvider
from provider_contracts.cache.memory import MemoryCacheProvider
from provider_contracts.vector_store.mock import MockVectorStoreProvider

# All mock providers work with zero configuration
llm = MockLLMProvider()
response = await llm.complete("Hello, world!")
```

### 3. Swap to a real provider — no logic changes

```python
# Development — local Ollama
from provider_contracts.llm.ollama import OllamaProvider  # needs [ollama]
llm = OllamaProvider(model="llama3.1:8b", base_url="http://localhost:11434")

# Production — OpenAI
from provider_contracts.llm.openai import OpenAIProvider  # needs [openai]
llm = OpenAIProvider(api_key="sk-...", model="gpt-4o")

# Same interface, same calling code
response = await llm.chat([
    LLMMessage(role="system", content="You are a helpful assistant."),
    LLMMessage(role="user", content="Explain provider patterns."),
])
print(response.content)
```

### 4. Type-safe tool calling

```python
from provider_contracts.llm import ToolDefinition, LLMMessage

tools = [
    ToolDefinition(
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
]

response = await llm.chat(
    messages=[LLMMessage(role="user", content="What's the weather in London?")],
    tools=tools,
)

for tool_call in response.tool_calls:
    print(f"Call: {tool_call.name}({tool_call.arguments})")
```

---

## Provider Catalogue

Every provider follows the same pattern: an **ABC** in `_base.py`, **Pydantic DTOs** for inputs/outputs, and one or more **concrete implementations**.

| Category | ABC | Concrete Implementations | Mock/Dev |
|----------|-----|--------------------------|----------|
| **LLM** | `AbstractLLMProvider` | `OpenAIProvider`, `AzureOpenAIProvider`, `AnthropicProvider`, `GeminiProvider`, `OllamaProvider` | `MockLLMProvider` |
| **Embedding** | `AbstractEmbeddingProvider` | `OpenAIEmbeddingProvider`, `AzureOpenAIEmbeddingProvider`, `SentenceTransformersProvider` | `MockEmbeddingProvider` |
| **Vector Store** | `AbstractVectorStoreProvider` | `QdrantProvider`, `PgVectorProvider`, `AzureAISearchProvider` | `MockVectorStoreProvider` |
| **Cache** | `AbstractCacheProvider` | `RedisProvider` | `MemoryCacheProvider` |
| **Storage** | `AbstractStorageProvider` | `R2StorageProvider`, `LocalStorageProvider` | `MockStorageProvider` |
| **Queue** | `AbstractQueueProvider` | `CeleryRedisProvider` | `InProcessQueueProvider` |
| **STT** | `AbstractSTTProvider` | `WhisperAPIProvider`, `WhisperLocalProvider` | `MockSTTProvider` |
| **TTS** | `AbstractTTSProvider` | `ElevenLabsProvider` | `MockTTSProvider` |
| **VAD** | `AbstractVADProvider` | `WebRTCVADProvider` | `MockVADProvider` |
| **PII** | `AbstractPiiProvider` | `PresidioProvider` | `MockPiiProvider` |
| **Content Safety** | `AbstractContentSafetyProvider` | `RuleBasedProvider` | `MockContentSafetyProvider` |
| **Audit Sink** | `AbstractAuditSinkProvider` | `JsonlAuditProvider` | `MockAuditSinkProvider` |
| **Telemetry** | `AbstractTelemetryProvider` | `ConsoleProvider` | `MockTelemetryProvider` |
| **Reranker** | `AbstractRerankerProvider` | `SentenceTransformersReranker` | — |
| **Session Store** | `AbstractSessionStoreProvider` | `MemorySessionStoreProvider` | `MockSessionStoreProvider` |
| **Email** | `AbstractEmailProvider` | — | `MockEmailProvider` |
| **Prompt Store** | `AbstractPromptStoreProvider` | — | — |
| **Secrets** | `AbstractSecretsProvider` | — | — |
| **Tools Client** | `AbstractToolsClientProvider` | — | — |

---

## Project Structure

```
provider-contracts/
├── pyproject.toml                    # Package metadata, extras, tool config
├── README.md
└── provider_contracts/
    ├── __init__.py                   # Public API — re-exports all ABCs + DTOs
    ├── _version.py                   # Semantic version
    ├── py.typed                      # PEP 561 marker
    │
    ├── llm/                          # LLM providers
    │   ├── _base.py                  #   ABC + LLMMessage, LLMResponse, ToolDefinition
    │   ├── openai.py                 #   OpenAI (GPT-4o, etc.)
    │   ├── azure_openai.py           #   Azure OpenAI
    │   ├── anthropic.py              #   Anthropic Claude
    │   ├── gemini.py                 #   Google Gemini
    │   ├── ollama.py                 #   Ollama (local)
    │   └── mock.py                   #   Mock for tests
    │
    ├── embedding/                    # Embedding providers
    │   ├── _base.py                  #   ABC + EmbeddingResult
    │   ├── openai.py                 #   OpenAI text-embedding-3
    │   ├── azure_openai.py           #   Azure OpenAI embeddings
    │   ├── sentence_transformers.py  #   Local sentence-transformers / BGE
    │   └── mock.py                   #   Mock for tests
    │
    ├── vector_store/                 # Vector store providers
    │   ├── _base.py                  #   ABC + VectorDocument, SearchResult
    │   ├── qdrant.py                 #   Qdrant
    │   ├── pgvector.py               #   PostgreSQL + pgvector
    │   ├── azure_ai_search.py        #   Azure AI Search
    │   └── mock.py                   #   Mock for tests
    │
    ├── cache/                        # Cache providers
    │   ├── _base.py                  #   ABC
    │   ├── redis.py                  #   Redis
    │   └── memory.py                 #   In-memory (dev/test)
    │
    ├── storage/                      # Object storage providers
    │   ├── _base.py                  #   ABC + StoredFile
    │   ├── r2.py                     #   Cloudflare R2 (S3-compatible)
    │   ├── local.py                  #   Local filesystem
    │   └── mock.py                   #   Mock for tests
    │
    ├── queue/                        # Task queue providers
    │   ├── _base.py                  #   ABC + EnqueuedTask
    │   ├── celery_redis.py           #   Celery + Redis
    │   └── in_process.py             #   In-process (dev/test)
    │
    ├── stt/                          # Speech-to-text providers
    │   ├── _base.py                  #   ABC + STTResult
    │   ├── whisper_api.py            #   OpenAI Whisper API
    │   ├── whisper_local.py          #   Local faster-whisper
    │   └── mock.py                   #   Mock for tests
    │
    ├── tts/                          # Text-to-speech providers
    │   ├── _base.py                  #   ABC + TTSResult
    │   ├── elevenlabs.py             #   ElevenLabs
    │   └── mock.py                   #   Mock for tests
    │
    ├── vad/                          # Voice activity detection
    │   ├── _base.py                  #   ABC + VADResult
    │   └── mock.py                   #   Mock for tests
    │
    ├── pii/                          # PII detection/anonymisation
    │   ├── _base.py                  #   ABC + PiiResult, PiiSpan
    │   ├── presidio.py               #   Microsoft Presidio
    │   └── mock.py                   #   Mock for tests
    │
    ├── content_safety/               # Content safety moderation
    │   ├── _base.py                  #   ABC + SafetyResult, SafetyVerdict
    │   ├── rule_based.py             #   Rule-based implementation
    │   └── mock.py                   #   Mock for tests
    │
    ├── audit_sink/                   # Audit logging
    │   ├── _base.py                  #   ABC + AuditEvent
    │   ├── jsonl.py                  #   JSONL file sink
    │   └── mock.py                   #   Mock for tests
    │
    ├── telemetry/                    # Observability / tracing
    │   ├── _base.py                  #   ABC + SpanContext
    │   ├── console.py                #   Console output
    │   └── mock.py                   #   Mock for tests
    │
    ├── reranker/                     # Reranking providers
    │   ├── _base.py                  #   ABC
    │   └── sentence_transformers.py  #   Cross-encoder reranker
    │
    ├── session_store/                # Session / conversation store
    │   ├── memory.py                 #   In-memory store
    │   └── mock.py                   #   Mock for tests
    │
    ├── email/                        # Email providers
    │   ├── _base.py                  #   ABC + EmailMessage, EmailResult
    │   └── mock.py                   #   Mock for tests (placeholder)
    │
    ├── prompt_store/                 # Prompt template storage
    │   └── _base.py                  #   ABC
    │
    ├── secrets/                      # Secret management
    │   └── _base.py                  #   ABC
    │
    └── tools_client/                 # External tool invocation
        └── _base.py                  #   ABC + ToolCall, ToolResult
```

---

## How to Add a New Provider

1. **Pick the right category** — find the ABC in `provider_contracts/<category>/_base.py`
2. **Create a new module** — e.g., `provider_contracts/cache/memcached.py`
3. **Subclass the ABC** and implement all `@abstractmethod`s
4. **Add the optional dependency** to `pyproject.toml` under `[project.optional-dependencies]`
5. **Re-export in `__init__.py`** of the sub-package if needed
6. **Run checks** — `mypy --strict` and `pytest`

```python
# provider_contracts/cache/memcached.py
from provider_contracts.cache._base import AbstractCacheProvider

class MemcachedProvider(AbstractCacheProvider):
    """Memcached-backed cache provider."""

    async def get(self, key: str) -> str | None:
        ...

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        ...

    async def delete(self, key: str) -> None:
        ...
```

---

## Development

```bash
# Clone the repo
git clone https://github.com/mohit-bohra-dev/provider_contracts.git
cd provider_contracts

# Create venv and install all deps
uv venv
uv pip install -e ".[all,dev]"

# Type checking
mypy --strict provider_contracts/

# Linting & formatting
ruff check provider_contracts/
ruff format provider_contracts/

# Run tests
pytest
```

### Code Quality

| Tool | Config | Purpose |
|------|--------|---------|
| **mypy** | `strict = true`, Python 3.11 | Static type checking |
| **ruff** | `line-length = 99`, select `E,F,I,UP,B,SIM` | Linting & formatting |
| **pytest** | `asyncio_mode = auto` | Testing with async support |

---

## License

[MIT](LICENSE)
