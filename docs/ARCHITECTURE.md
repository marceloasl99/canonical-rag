# Arquitetura Técnica do CanonicalRAG

Este documento detalha o funcionamento interno dos 4 pilares do **CanonicalRAG**:

```
                       ┌─────────────────────────┐
                       │   Consulta do Usuário   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  Cache Semântico Local  │──(Hit: <10ms)──► Resposta Imediata
                       └────────────┬────────────┘                   (0 Tokens)
                                    │ (Miss)
                                    ▼
                       ┌─────────────────────────┐
                       │    Tiered Search FTS5   │
                       │   (Top 20 Candidatos)   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Mini-LLM Local FlashRank│
                       │ (Cross-Encoder CPU ONNX)│
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ UniversalDossierBuilder │
                       │(Poda Extrativa: -70% tok)
                       └────────────┬────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
      [ Modo Extração Pura ]              [ Gateway LLM Multi-Slot ]
       (Zero Tokens / 100%                 (Claude / Gemini / OpenRouter)
        Fidelidade Documental)             Prompt Caching + Síntese Fluida
```

---

## 1. O Pipeline de Busca em Duas Etapas (Two-Stage Retrieval)

Em sistemas RAG tradicionais baseados puramente em vetores (Embeddings densos), termos técnicos específicos, siglas, números de normas e fórmulas frequentemente são diluídos pelo cálculo de similaridade de cosseno. 

O CanonicalRAG adota uma abordagem de **funil híbrido de precisão**:

1. **Busca Inicial Ampla (SQLite FTS5):**
   - Utiliza índice de texto completo FTS5 com tokenizador `unicode61 remove_diacritics 2`.
   - Executa busca em camadas:
     * *Camada 1:* Correspondência exata de termos compostos entre aspas.
     * *Camada 2:* Bigramas técnicos consecutivos.
     * *Camada 3:* Proximidade léxica (`NEAR(termo1 termo2, 15)`).
     * *Camada 4:* Prefixo estrito com operador `AND`.
   - Retorna os **Top 20 candidatos** em menos de 15ms com **zero uso de GPU**.

2. **Reclassificação Semântica com Mini-LLM Local (FlashRank ONNX):**
   - Os 20 candidatos são submetidos ao modelo Cross-Encoder `ms-marco-MiniLM-L-12-v2`.
   - O modelo avalia a interação real par-a-par entre a pergunta do usuário e o conteúdo de cada fragmento.
   - Atribui uma pontuação de relevância contínua (`rerank_score`) e seleciona apenas os **Top 5 fragmentos mais densos**.

---

## 2. Poda Extrativa de Sentenças (*Sentence-Level Context Pruning*)

Mesmo após o reranking, fragmentos típicos de 800 caracteres contêm introduções, agradecimentos, frases de transição e pontes irrelevantes que inflam a contagem de tokens de entrada.

O `UniversalDossierBuilder`:
1. Divide cada fragmento selecionado em orações semânticas completas (`re.split`).
2. Avalia a densidade léxica de cada sentença em relação aos conceitos da consulta.
3. Descarta as sentenças de menor relevância e recompõe o fragmento com as 3 ou 4 sentenças de maior densidade informacional.
4. **Resultado:** Reduz o tamanho do contexto injetado no prompt em **65% a 70%**, acelerando o tempo de inferência e cortando custos de API.

---

## 3. Cache Semântico Local em SQLite (Consumo Zero)

Para evitar reprocessar perguntas idênticas ou com ligeiras variações sintáticas:
1. A consulta passa por uma normalização de texto (remoção de acentos, pontuações, caracteres especiais e stopwords universais).
2. As palavras-chave resultantes são ordenadas alfabeticamente para gerar uma chave canônica única.
3. É computado um hash SHA-256 da chave canônica (`query_hash`).
4. Se o hash existir na tabela `query_cache`, a resposta é servida instantaneamente em **menos de 10ms**, sem nenhuma chamada de rede externa e com **0 tokens consumidos**.

---

## 4. Gateway LLM com Prompt Caching Efêmero

Quando a consulta exige geração via LLM externa:
- As instruções de sistema fixas recebem marcadores de cache efêmero (`cache_control: {"type": "ephemeral"}` na Anthropic Claude e otimização de cabeçalho no Gemini).
- O gateway implementa balanceamento automático (Round-Robin) entre chaves de API e alternância dinâmica entre slots ativos em caso de erro (*Failover automático*).
