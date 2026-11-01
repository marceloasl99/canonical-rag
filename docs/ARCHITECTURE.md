# CanonicalRAG — Technical Architecture Deep Dive

*Read this document in [English](#english-version) or [Português](#versão-em-português).*

---

<a name="english-version"></a>
## 🌐 English: Technical Architecture

CanonicalRAG is an ultra-lightweight (~40 MB RAM in idle), production-ready Two-Stage Retrieval-Augmented Generation engine designed for high-precision technical documentation (Engineering, Law, Medicine, IT, and Scientific Literature).

```
                       ┌─────────────────────────┐
                       │       User Query        │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Local Semantic Cache    │──(Hit: <10ms)──► Immediate Response
                       │ (SHA-256 SQLite B-Tree) │                   (0 LLM Tokens / $0.00)
                       └────────────┬────────────┘
                                    │ (Cache Miss)
                                    ▼
                       ┌─────────────────────────┐
                       │ Stage 1: Broad FTS5     │
                       │ (BM25 + NEAR Proximity) │──► Top 20 Candidates (< 15ms)
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Stage 2: Mini-LLM ONNX  │
                       │ (FlashRank Cross-Encoder│──► Top 5 Gold Excerpts (~0.8s on CPU)
                       │  ms-marco-MiniLM-L-12)  │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ UniversalDossierBuilder │
                       │ Sentence Pruning (-70%) │──► High-Density Context (~350 tok)
                       └────────────┬────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
        [ Direct Pure Mode ]              [ Multi-Slot LLM Gateway ]
         (Zero LLM Tokens /                (Claude / Gemini / OpenRouter)
          100% Documental)                 Ephemeral Caching + Synthesis
```

---

### 1. The Two-Stage Funnel vs Dense Vector Databases
Traditional enterprise RAG stacks rely heavily on dense vector embeddings (e.g. `text-embedding-3-small`, `bge-large`) stored in standalone vector databases (Milvus, Qdrant, Chroma, Pinecone). In technical domains, this architecture introduces two severe flaws:
1. **Semantic Dilution:** Compressing 800-character technical passages into a single 768-dimensional float vector causes exact technical terms, standard IDs (e.g., `ISO 27001`, `RFC 7519`, `ABNT NBR 6118`), article numbers, and formulas to lose statistical prominence against common conversational vocabulary.
2. **Infrastructure Bloat:** Running standalone vector DB containers requires 4 GB to 8 GB of RAM merely to initialize.

**The CanonicalRAG Solution:**
- **Stage 1 — Lexical Retrieval (SQLite FTS5):** Uses native SQLite FTS5 with the `unicode61 remove_diacritics 2` tokenizer. Scans 60,000+ chunks in `< 15ms` using a tiered search strategy:
  - Exact quoted phrases (`"quoted query"`)
  - Consecutive technical bigrams
  - Proximity search (`NEAR(term1 term2, 15)`)
  - Strict prefix queries (`term*`)
  - Returns the **Top 20 candidate chunks** with zero GPU usage and zero vector overhead.
- **Stage 2 — Mini-LLM Cross-Encoder (FlashRank ONNX):** The 20 candidates are evaluated in-process by the `ms-marco-MiniLM-L-12-v2` cross-encoder running on CPU via ONNX Runtime (~25 MB RAM). Unlike bi-encoders, the cross-encoder computes full token-to-token cross-attention between query and passage, outputting real relevance scores to select the **Top 5 gold excerpts**.

---

### 2. Extractive Sentence-Level Context Pruning
On average, 65% to 70% of chunks extracted from technical documentation or PDFs consist of generic preambles, acknowledgments, footnotes, or conversational transitions that inflate token consumption without adding factual density.

The `UniversalDossierBuilder`:
1. Decomposes each of the 5 selected chunks into discrete grammatical sentences.
2. Evaluates the lexical and conceptual density of each sentence against the query.
3. Discards low-signal sentences and reconstructs a compact passage retaining only the 3 to 4 highest-density sentences.
4. **Impact:** Reduces input context size from ~1,200 tokens down to **~350 tokens**, cutting LLM input costs by ~70% and preventing hallucinations.

---

### 3. Deterministic Local Semantic Caching (SQLite SHA-256)
To avoid redundant LLM invocations for identical or syntactically varied queries:
1. Queries undergo canonical normalization (stripping accents, punctuation, universal stopwords, and sorting conceptual word stems alphabetically).
2. For example, `"How to design reinforced concrete beams?"` and `"reinforced concrete beam design technical guide"` normalize to the exact same canonical key: `[beam, concrete, design, reinforce]`.
3. Computes a deterministic SHA-256 hash (`query_hash`) and performs an index lookup in SQLite.
4. Hits return the verified technical dossier in **`< 10ms` with 0 external API calls and $0.00 cost**.

---

### 4. Resilient Multi-LLM Gateway with Ephemeral Caching
- **Multi-Provider Support:** Anthropic Claude 3.5, Google Gemini 1.5/2.0, OpenRouter (including free tier `:free` models), and Cloudflare Workers AI.
- **Ephemeral Prompt Caching:** Adds provider-specific cache headers (`cache_control: {"type": "ephemeral"}` for Claude and context caching for Gemini) to reuse system instructions for free.
- **Key Pooling & Round-Robin:** Rotates requests across multiple registered API keys per provider to prevent rate-limit throttling (HTTP 429).
- **Automated Failover:** If an active model fails or times out, the gateway seamlessly fails over to the next configured model slot without user intervention.

---

### ⚖️ Architectural Comparison: CanonicalRAG vs Traditional Heavy Vector RAG

| Criterion / Feature | CanonicalRAG | Traditional Heavy Vector RAG (Milvus / Qdrant) |
| :--- | :--- | :--- |
| **Idle Memory (RAM)** | **~40 MB RAM** | 4,000 MB – 8,000 MB RAM |
| **Dedicated GPU** | **Not required (100% CPU on low-end VPS)** | Recommended or required for scale |
| **Fidelity for Acronyms & Formulas** | **~100% (BM25 + Proximity + Cross-Attention)** | Degraded by 768-dim vector embedding dilution |
| **Token Cost per Query** | **~350 tokens (or 0 with Cache Hit)** | 1,500 – 3,500 tokens (no sentence pruning) |
| **Cold Start Time** | **< 1 second** | 30 – 60 seconds (multi-container bootstrap) |
| **Database & Infrastructure** | **Single SQLite file (`.db`)** | Distributed cluster (Vector DB + PostgreSQL + Redis) |
| **Backups & Portability** | **Copy a single file** | Cluster snapshots, volume syncing |

---

<a name="versão-em-português"></a>
## 🇧🇷 Português: Arquitetura Técnica

O **CanonicalRAG** é uma engine de **Recuperação Aumentada por Geração (RAG) em Duas Etapas** de ultra-baixo consumo (~40 MB RAM em repouso), desenvolvida para operar com **fidelidade estrita a documentos técnicos** (Engenharia, Direito, Medicina, TI e Literatura Científica).

### 1. O Pipeline de Busca em Duas Etapas (Two-Stage Retrieval)
Em sistemas RAG tradicionais baseados puramente em vetores (Embeddings densos), termos técnicos específicos, siglas, números de normas e fórmulas frequentemente são diluídos pelo cálculo de similaridade de cosseno. 

O CanonicalRAG adota uma abordagem de **funil híbrido de precisão**:
1. **Busca Inicial Ampla (SQLite FTS5):**
   - Utiliza índice de texto completo FTS5 com tokenizador `unicode61 remove_diacritics 2`.
   - Executa busca em camadas: correspondência exata de termos compostos, bigramas técnicos consecutivos, proximidade léxica (`NEAR(termo1 termo2, 15)`) e prefixos estritos.
   - Retorna os **Top 20 candidatos** em menos de 15ms com **zero uso de GPU**.
2. **Reclassificação Semântica com Mini-LLM Local (FlashRank ONNX):**
   - Os 20 candidatos são submetidos ao modelo Cross-Encoder `ms-marco-MiniLM-L-12-v2` rodando na CPU via ONNX Runtime (~25 MB RAM).
   - O modelo avalia a interação real par-a-par entre a pergunta do usuário e o conteúdo de cada fragmento.
   - Atribui uma pontuação de relevância contínua (`rerank_score`) e seleciona apenas os **Top 5 fragmentos mais densos**.

### 2. Poda Extrativa de Sentenças (*Sentence-Level Context Pruning*)
Mesmo após o reranking, fragmentos típicos de 800 caracteres contêm introduções, agradecimentos, frases de transição e pontes irrelevantes que inflam a contagem de tokens de entrada.
- O `UniversalDossierBuilder` divide cada fragmento selecionado em orações semânticas completas, avalia a densidade conceitual de cada uma e descarta o ruído.
- **Resultado:** Reduz o tamanho do contexto injetado no prompt em **65% a 70%** (~350 tokens no total), acelerando a inferência e cortando custos de API.

### 3. Cache Semântico Local em SQLite (Consumo Zero)
- A consulta passa por uma normalização de texto (remoção de acentos, pontuações, caracteres especiais e stopwords universais).
- As palavras-chave resultantes são ordenadas alfabeticamente para gerar uma chave canônica única com hash SHA-256 (`query_hash`).
- Se o hash existir na tabela `query_cache`, a resposta é servida instantaneamente em **menos de 10ms**, sem nenhuma chamada de rede externa e com **0 tokens consumidos**.

### 4. Gateway LLM com Prompt Caching Efêmero
- Suporte nativo a Anthropic Claude (com Prompt Caching efêmero), Google Gemini, OpenRouter (modelos gratuitos `:free`) e Cloudflare Workers AI.
- Balanceamento automático (Round-Robin) entre chaves de API e alternância dinâmica entre slots ativos em caso de erro (*Failover automático*).
