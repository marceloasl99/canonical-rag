# CanonicalRAG — High-Fidelity Two-Stage Universal RAG Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![FlashRank](https://img.shields.io/badge/Reranker-FlashRank%20ONNX-orange.svg)](https://github.com/PrithivirajDamodaran/FlashRank)
[![Memory Footprint](https://img.shields.io/badge/RAM%20Idle-~40MB-success.svg)](#)
[![Zero-GPU](https://img.shields.io/badge/GPU-Not%20Required-brightgreen.svg)](#)

> **CanonicalRAG** é uma engine de **RAG em Duas Etapas (Two-Stage Retrieval)** de ultra-baixo consumo (~40 MB RAM em repouso), projetada para operar com **fidelidade estrita sobre acervos documentais técnicos** (Engenharia, Direito, Medicina, TI, etc.). 
>
> Ao combinar busca de texto completo local com um **Mini-LLM Cross-Encoder via ONNX na CPU**, poda extrativa de contexto e cache semântico persistente, o CanonicalRAG reduz em até **70% a 100% o consumo de tokens em APIs de LLM externas**.

---

## 🎯 Por que o CanonicalRAG foi Criado?

A maioria dos frameworks modernos de RAG (como implementações pesadas de LangChain ou LlamaIndex) comete dois erros críticos em produção:
1. **Consumo Excessivo de Memória e Infraestrutura:** Exigem containers pesados de bancos vetoriais (Milvus, Qdrant, Elasticsearch) consumindo de 4 GB a 8 GB de RAM só para inicializar.
2. **Injeção de Ruído & Explosão de Tokens:** Injetam fragmentos inteiros e brutos de PDFs no prompt das LLMs. O modelo externo consome milhares de tokens de entrada tentando adivinhar o contexto, gerando custos elevados e alucinações.

O **CanonicalRAG** resolve esses problemas através de um **Pipeline em Funil**:
- **Consumo de Memória Leve:** Roda inteiramente em uma CPU modesta com menos de 100 MB de RAM.
- **Precisão Cirúrgica:** Recupera primeiro via SQLite FTS5, reclassifica com Cross-Encoder local e poda sentenças vazias antes de enviar qualquer caractere à LLM.

---

## 🏗️ Visão Geral da Arquitetura

```
[ Usuário / API / Interface Web ]
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. CAMADA DE CACHE SEMÂNTICO LOCAL (< 10ms | 0 Tokens LLM)  │
│ - Normalização de stopwords, ortografia e ordinais          │
│ - Busca por hash SHA-256 no SQLite local                    │
└──────────────────────────────┬──────────────────────────────┘
               │ (Miss)
               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. ETAPA 1: BUSCA AMPLA EM CAMADAS (SQLite FTS5 Nativo)     │
│ - Match exato entre aspas + Bigramas consecutivos           │
│ - Proximidade NEAR(...) + Prefixo AND + Ranking BM25        │
│ - Coleta instantânea de até 20 candidatos em < 15ms         │
└──────────────────────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. ETAPA 2: MINI-LLM LOCAL CROSS-ENCODER (FlashRank ONNX)   │
│ - Modelo ms-marco-MiniLM-L-12-v2 rodando 100% na CPU        │
│ - Avalia relevância cruzada real par-a-par                  │
│ - Descarta os 15 candidatos irrelevantes e isola o Top 5    │
└──────────────────────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. PODA EXTRATIVA DE SENTENÇAS & CONSTRUTOR DE DOSSIÊ       │
│ - Elimina parágrafos de transição e saudações               │
│ - Isola apenas as orações de alta densidade conceitual      │
│ - Redução de 70% nos tokens de entrada (~350 tokens total)  │
└──────────────────────────────┬──────────────────────────────┘
               │
       ┌───────┴──────────────────────────────┐
       ▼                                      ▼
[ MODO LOCAL DIRETO ]               [ GATEWAY MULTI-LLM ]
(0 Tokens Externos)                 (Claude / Gemini / OpenRouter)
- Laudo técnico imediato            - Prompt Caching efêmero
- 100% fiel à literatura            - Redação fluida e madura
```

---

## 📊 Benchmark Comparativo

| Métrica | RAG Tradicional (Vetorial) | CanonicalRAG |
| :--- | :---: | :---: |
| **Consumo de RAM em Idle** | 800 MB – 2.5 GB | **~35 MB – 50 MB** |
| **Exigência de GPU** | Frequentemente necessária | **Zero (100% CPU via ONNX)** |
| **Tokens de Entrada por Consulta** | ~2.500 a 4.000 tokens | **~350 a 500 tokens** (-70%) |
| **Consultas Frequentes / Repetidas** | Custo integral da API | **0 Tokens (Cache Hit < 10ms)** |
| **Re-ranking Semântico** | Via API paga ou PyTorch | **~0.8s localmente (sem custos)** |
| **Taxonomia Documental** | Depende de banco vetorial externo | **Nativa em SQLite com FTS5** |

---

## ⚡ Instalação e Execução

### 1. Pré-requisitos
- Python 3.10 ou superior
- Linux, macOS ou Windows
- Mínimo de 512 MB de RAM disponíveis

### 2. Passo a Passo

```bash
# 1. Clonar o repositório
git clone https://github.com/seu-usuario/canonical-rag.git
cd canonical-rag

# 2. Criar e ativar o ambiente virtual
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Configurar as credenciais (opcional para modo local)
cp config.example.json config.json

# 5. Iniciar o servidor
python3 -m backend.main
```

A interface web estará disponível em: `http://localhost:80` (ou na porta configurada via variável `PORT`).

---

## 🛠️ Configuração de Modelos e Provedores

O arquivo `config.json` permite configurar múltiplos provedores e até **3 Slots de Modelos Ativos** simultâneos com suporte a failover e round-robin:

- **Google Gemini:** `gemini-2.5-flash`, `gemini-2.5-pro`
- **Anthropic Claude:** `claude-haiku-5-5`, `claude-3-5-sonnet-latest` (com Prompt Caching efêmero)
- **OpenRouter:** Suporte nativo a modelos gratuitos (ex: `google/gemini-2.0-flash-exp:free`, `qwen/qwen-2.5-72b:free`)
- **Cloudflare Workers AI & Groq:** Para inferência de modelos open-source de alta velocidade

---

## 📚 Documentação Detalhada

- [📖 Arquitetura Técnica Completa](docs/ARCHITECTURE.md): Análise detalhada dos 4 pilares do sistema.
- [🔌 Guia de Endpoints da API](docs/API_REFERENCE.md): Documentação completa de todas as rotas REST.
- [🚀 Guia de Deploy e Produção](docs/DEPLOYMENT.md): Configuração com Systemd, Nginx e Docker.

---

## 📄 Licença
Distribuído sob a licença MIT. Consulte [LICENSE](LICENSE) para mais informações.
