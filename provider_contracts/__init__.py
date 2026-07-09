"""
provider_contracts — shared abstract provider interfaces and reusable adapters.

Public API (ABCs + DTOs only):

    LLM
    ───
    from provider_contracts.llm import AbstractLLMProvider, LLMMessage, LLMResponse

    Cache
    ─────
    from provider_contracts.cache import AbstractCacheProvider

    Email
    ─────
    from provider_contracts.email import AbstractEmailProvider, EmailMessage, EmailResult

    Storage
    ───────
    from provider_contracts.storage import AbstractStorageProvider, StoredFile

    STT
    ───
    from provider_contracts.stt import AbstractSTTProvider, STTResult

    TTS
    ───
    from provider_contracts.tts import AbstractTTSProvider, TTSResult

    VAD
    ───
    from provider_contracts.vad import AbstractVADProvider, VADResult

    Vector Store
    ────────────
    from provider_contracts.vector_store import (
        AbstractVectorStoreProvider, VectorDocument, SearchResult
    )

    Queue
    ─────
    from provider_contracts.queue import AbstractQueueProvider, EnqueuedTask

    Embedding
    ─────────
    from provider_contracts.embedding import AbstractEmbeddingProvider, EmbeddingResult

    Audit Sink
    ──────────
    from provider_contracts.audit_sink import AbstractAuditSinkProvider, AuditEvent

    Content Safety
    ──────────────
    from provider_contracts.content_safety import (
        AbstractContentSafetyProvider, SafetyResult, SafetyVerdict
    )

    PII
    ───
    from provider_contracts.pii import AbstractPiiProvider, PiiResult, PiiSpan

    Prompt Store
    ────────────
    from provider_contracts.prompt_store import AbstractPromptStoreProvider

    Secrets
    ───────
    from provider_contracts.secrets import AbstractSecretsProvider

    Telemetry
    ─────────
    from provider_contracts.telemetry import AbstractTelemetryProvider, SpanContext

    Tools Client
    ────────────
    from provider_contracts.tools_client import (
        AbstractToolsClientProvider, ToolCall, ToolResult
    )

Concrete implementations live in each provider sub-package and require
their corresponding optional extras to be installed.
"""

from ._version import __version__

# ── ABCs ──────────────────────────────────────────────────────────────────────
from .audit_sink import AbstractAuditSinkProvider, AuditEvent
from .cache import AbstractCacheProvider
from .content_safety import AbstractContentSafetyProvider, SafetyResult, SafetyVerdict
from .email import AbstractEmailProvider, EmailMessage, EmailResult
from .embedding import AbstractEmbeddingProvider, EmbeddingResult
from .llm import AbstractLLMProvider, LLMMessage, LLMResponse
from .pii import AbstractPiiProvider, PiiResult, PiiSpan
from .prompt_store import AbstractPromptStoreProvider
from .queue import AbstractQueueProvider, EnqueuedTask
from .secrets import AbstractSecretsProvider
from .storage import AbstractStorageProvider, StoredFile
from .stt import AbstractSTTProvider, STTResult
from .telemetry import AbstractTelemetryProvider, SpanContext
from .tools_client import AbstractToolsClientProvider, ToolCall, ToolResult
from .tts import AbstractTTSProvider, TTSResult
from .vad import AbstractVADProvider, VADResult
from .vector_store import AbstractVectorStoreProvider, SearchResult, VectorDocument

__all__ = [
    "__version__",
    # LLM
    "AbstractLLMProvider",
    "LLMMessage",
    "LLMResponse",
    # Cache
    "AbstractCacheProvider",
    # Email
    "AbstractEmailProvider",
    "EmailMessage",
    "EmailResult",
    # Storage
    "AbstractStorageProvider",
    "StoredFile",
    # STT
    "AbstractSTTProvider",
    "STTResult",
    # TTS
    "AbstractTTSProvider",
    "TTSResult",
    # VAD
    "AbstractVADProvider",
    "VADResult",
    # Vector Store
    "AbstractVectorStoreProvider",
    "VectorDocument",
    "SearchResult",
    # Queue
    "AbstractQueueProvider",
    "EnqueuedTask",
    # Embedding
    "AbstractEmbeddingProvider",
    "EmbeddingResult",
    # Audit Sink
    "AbstractAuditSinkProvider",
    "AuditEvent",
    # Content Safety
    "AbstractContentSafetyProvider",
    "SafetyResult",
    "SafetyVerdict",
    # PII
    "AbstractPiiProvider",
    "PiiResult",
    "PiiSpan",
    # Prompt Store
    "AbstractPromptStoreProvider",
    # Secrets
    "AbstractSecretsProvider",
    # Telemetry
    "AbstractTelemetryProvider",
    "SpanContext",
    # Tools Client
    "AbstractToolsClientProvider",
    "ToolCall",
    "ToolResult",
]
