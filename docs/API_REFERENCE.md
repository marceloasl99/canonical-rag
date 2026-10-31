# Guia de Endpoints da API REST (CanonicalRAG)

O **CanonicalRAG** expõe uma API REST assíncrona de alto desempenho desenvolvida em FastAPI.

---

## 1. Consulta RAG & Geração

### `POST /api/query`
Executa o pipeline de recuperação em duas etapas, poda e síntese.

**Corpo da Requisição (JSON):**
```json
{
  "query": "Como dimensionar a armadura longitudinal de uma viga submetida a flexão simples?",
  "limit_chunks": 5,
  "only_retrieve": false,
  "preferred_slot_id": "slot-1",
  "force_refresh": false
}
```

**Parâmetros:**
- `query` *(string, obrigatório)*: Pergunta ou texto a ser analisado.
- `limit_chunks` *(int, padrão: 5)*: Quantidade de trechos canônicos a utilizar após o reranking.
- `only_retrieve` *(bool, padrão: false)*: Quando `true`, opera no modo local sem acionar nenhuma LLM externa (0 tokens).
- `preferred_slot_id` *(string, opcional)*: Especifica o slot de modelo ativo desejado (`slot-1`, `slot-2`, etc.).
- `force_refresh` *(bool, padrão: false)*: Ignora o cache semântico e força nova consulta e geração.

**Resposta de Sucesso (`200 OK`):**
```json
{
  "query": "Como dimensionar a armadura longitudinal...",
  "found_chunks": [
    {
      "filename": "NBR_6118_Projeto_Estruturas_Concreto.pdf",
      "page_number": 42,
      "chunk_index": 3,
      "category": "engenharia",
      "content": "Para vigas sob flexão simples no Estádio III...",
      "rerank_score": 0.8924
    }
  ],
  "dossier_facts": {
    "query_original": "Como dimensionar a armadura...",
    "termos_exatos": [],
    "variaveis_principais": ["dimensionar", "armadura", "longitudinal", "viga", "flexao"]
  },
  "report": "## 1. Fundamento e Critérios Normativos\n\nO dimensionamento...",
  "llm_meta": {
    "slot_label": "Modelo Ativo 1 (Principal)",
    "provider": "claude",
    "model": "claude-haiku-5-5",
    "tokens_saved_by_pruning": 348,
    "reranker": "FlashRank MiniLM-L-12-v2"
  },
  "from_cache": false,
  "tokens_saved": 348
}
```

---

## 2. Telemetria e Monitoramento do Sistema

### `GET /api/system-telemetry`
Retorna métricas em tempo real de hardware e consumo de tokens para a barra superior da UI.

**Resposta de Sucesso (`200 OK`):**
```json
{
  "status": "ok",
  "cpu_percent": 12.4,
  "ram_used_mb": 450,
  "ram_total_mb": 2512,
  "ram_percent": 17.9,
  "disk_used_gb": 3.4,
  "disk_total_gb": 13.7,
  "disk_percent": 24.8,
  "tokens_saved_total": 15276,
  "cache_hits_total": 6,
  "cached_queries_count": 3,
  "total_documents": 283,
  "total_chunks": 59647
}
```

---

## 3. Catálogo de Documentos e Indexação

### `GET /api/documents`
Lista as obras e arquivos catalogados com suporte a busca e filtros por categoria.

**Parâmetros de Consulta (Query Params):**
- `search` *(string, opcional)*: Termo para filtrar pelo nome do arquivo.
- `category` *(string, opcional)*: Filtro por categoria documental (ex: `engenharia`, `direito`).

### `POST /api/index`
Dispara a varredura e indexação incremental de arquivos da pasta configurada em segundo plano.

---

## 4. Configuração e Chaves de API

### `GET /api/config`
Retorna as configurações atuais do sistema com chaves mascaradas para segurança.

### `POST /api/settings`
Atualiza os parâmetros operacionais globais (pasta de documentos, temperatura, idioma, etc.).

### `POST /api/active-models`
Atualiza a lista de slots de modelos ativos para geração em nuvem.
