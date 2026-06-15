# Graph Report - .  (2026-06-14)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 846 nodes · 2092 edges · 19 communities (18 shown, 1 thin omitted)
- Extraction: 68% EXTRACTED · 32% INFERRED · 0% AMBIGUOUS · INFERRED: 677 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `14ec2743`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]

## God Nodes (most connected - your core abstractions)
1. `AbstractLLMProvider` - 75 edges
2. `LLMMessage` - 73 edges
3. `LLMResponse` - 73 edges
4. `ToolDefinition` - 71 edges
5. `ToolCallRequest` - 70 edges
6. `AbstractVectorStoreProvider` - 42 edges
7. `VectorDocument` - 39 edges
8. `SearchResult` - 39 edges
9. `AbstractStorageProvider` - 31 edges
10. `AbstractEmbeddingProvider` - 29 edges

## Surprising Connections (you probably didn't know these)
- `bool` --uses--> `AbstractCacheProvider`  [INFERRED]
  cache/redis.py → cache/_base.py
- `test_ttl_expiration()` --calls--> `InMemorySessionStoreProvider`  [INFERRED]
  session_store/tests/test_contract.py → session_store/memory.py
- `JsonlAuditSinkProvider` --uses--> `AbstractAuditSinkProvider`  [INFERRED]
  audit_sink/jsonl.py → audit_sink/_base.py
- `JsonlAuditSinkProvider` --uses--> `AuditEvent`  [INFERRED]
  audit_sink/jsonl.py → audit_sink/_base.py
- `str` --uses--> `AbstractAuditSinkProvider`  [INFERRED]
  audit_sink/jsonl.py → audit_sink/_base.py

## Import Cycles
- 1-file cycle: `queue/celery_redis.py -> queue/celery_redis.py`

## Communities (19 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (86): Content, AnthropicProvider, Any, bool, float, int, LLMMessage, LLMResponse (+78 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (62): AsyncEngine, AzureAISearchVectorStoreProvider, Any, float, int, SearchResult, str, VectorDocument (+54 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (39): AzureOpenAIEmbeddingProvider, EmbeddingResult, int, str, Azure OpenAI embedding adapter., Wraps the Azure OpenAI embeddings API.      Requires the ``openai`` optional e, Initialize the Azure OpenAI embedding provider.          Args:             en, Generate an embedding for a single text. (+31 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (39): AbstractStorageProvider, bool, bytes, int, str, Abstract base class for file storage providers., Contract that every storage adapter must satisfy., Upload raw bytes and return storage metadata.          Args:             data: F (+31 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (37): BaseModel, AbstractEmailProvider, EmailMessage, EmailResult, Abstract base class for email providers., Result returned after attempting to send an email., Contract that every email adapter must satisfy., Send an email.          Args:             message: The email to deliver. (+29 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (27): AbstractCacheProvider, Any, bool, int, str, Abstract base class for cache providers., Return the cached value for key, or None if missing / expired.          Args:, Store a value under key.          Args:             key: Cache key. (+19 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (34): AbstractSTTProvider, bytes, str, Abstract base class for speech-to-text providers., Contract that every STT adapter must satisfy., Transcribe raw audio bytes to text.          Args:             audio: Raw audio, Structured transcript from an STT provider., STTResult (+26 more)

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (29): Celery, AbstractQueueProvider, EnqueuedTask, Any, bool, int, str, Abstract base class for task/background-job queue providers. (+21 more)

### Community 8 - "Community 8"
Cohesion: 0.10
Nodes (27): AbstractTelemetryProvider, Any, float, str, Abstract base class for telemetry / observability providers., Lightweight handle to an active tracing span., Contract that every telemetry adapter must satisfy.      Concrete implementation, Create and activate a tracing span as an async context manager.          Overrid (+19 more)

### Community 9 - "Community 9"
Cohesion: 0.11
Nodes (25): AbstractToolsClientProvider, str, Abstract base class for tools client providers., The response from a tool endpoint invocation., Contract that every tools client adapter must satisfy.      Concrete implementat, Invoke a tool endpoint and return its result.          Args:             tool: S, Return the names of all available tools from the endpoint., A request to invoke a tool endpoint. (+17 more)

### Community 10 - "Community 10"
Cohesion: 0.10
Nodes (27): AbstractVADProvider, bytes, int, Abstract voice activity detection contract., Speech likelihood for a single analysed chunk., Contract implemented by concrete VAD adapters., Analyse *audio_chunk* for voiced content.          Args:             audio_chunk, VADResult (+19 more)

### Community 11 - "Community 11"
Cohesion: 0.10
Nodes (18): ABC, AbstractPromptStoreProvider, str, Abstract base class for prompt store providers., Retrieve a prompt template by name.          Args:             name: Logical pro, Return all known prompt names in this store., Contract that every prompt store adapter must satisfy.      A prompt store is a, FilePromptStoreProvider (+10 more)

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (23): AbstractTTSProvider, float, str, Abstract base class for text-to-speech providers., Contract that every TTS adapter must satisfy., Synthesize speech from text.          Args:             text: The text to conver, Structured audio output from a TTS provider., TTSResult (+15 more)

### Community 13 - "Community 13"
Cohesion: 0.09
Nodes (20): AbstractSessionStoreProvider, ConversationSession, ConversationTurn, InMemorySessionStoreProvider, int, str, In-memory session store provider., Delete a session entirely. (+12 more)

### Community 14 - "Community 14"
Cohesion: 0.11
Nodes (17): AbstractAuditSinkProvider, AuditEvent, Abstract base class for audit sink providers., A structured audit log entry., Contract that every audit sink adapter must satisfy., Emit a single audit event to the sink.          Implementations must be non-bloc, Flush any buffered events to the backing store.          Called on graceful shut, Audit sink provider interfaces. (+9 more)

### Community 15 - "Community 15"
Cohesion: 0.14
Nodes (21): AbstractContentSafetyProvider, str, Abstract base class for content safety providers., Structured result from a content safety check., Contract that every content safety adapter must satisfy., Analyse *text* for unsafe content.          Args:             text: The content, Outcome of a content safety check., SafetyResult (+13 more)

### Community 16 - "Community 16"
Cohesion: 0.27
Nodes (7): bool, float, int, str, Single-turn completion.          Args:             json_mode: When True, constra, Streaming chat — yields text chunks., Return embedding vector for the given text.

### Community 17 - "Community 17"
Cohesion: 0.33
Nodes (3): MockSessionStoreProvider, Mock session store provider for tests., Mock implementation of the SessionStoreProvider for testing.

## Knowledge Gaps
- **20 isolated node(s):** `bool`, `str`, `int`, `str`, `Any` (+15 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AbstractLLMProvider` connect `Community 0` to `Community 16`, `Community 11`, `Community 4`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `AbstractCacheProvider` connect `Community 5` to `Community 11`, `Community 4`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Why does `AbstractVectorStoreProvider` connect `Community 1` to `Community 11`, `Community 4`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Are the 60 inferred relationships involving `AbstractLLMProvider` (e.g. with `Content` and `AnthropicProvider`) actually correct?**
  _`AbstractLLMProvider` has 60 INFERRED edges - model-reasoned connections that need verification._
- **Are the 60 inferred relationships involving `LLMMessage` (e.g. with `Content` and `AnthropicProvider`) actually correct?**
  _`LLMMessage` has 60 INFERRED edges - model-reasoned connections that need verification._
- **Are the 60 inferred relationships involving `LLMResponse` (e.g. with `Content` and `AnthropicProvider`) actually correct?**
  _`LLMResponse` has 60 INFERRED edges - model-reasoned connections that need verification._
- **Are the 60 inferred relationships involving `ToolDefinition` (e.g. with `Content` and `AnthropicProvider`) actually correct?**
  _`ToolDefinition` has 60 INFERRED edges - model-reasoned connections that need verification._