import logging
import asyncio
from typing import Dict, Any, List, Optional
from .config import AppConfig
from .indexer import DocumentIndexer
from .llm_gateway import LLMGateway
from .dossier_builder import UniversalDossierBuilder

logger = logging.getLogger("RAGOrchestrator")

SYSTEM_PROMPT_CANONICAL = """Você é um especialista e redator de elite especializado em síntese técnica e fidelidade documental no idioma {language}.
Sua missão é responder com máxima clareza didática, precisão conceitual e rigor metodológico à PERGUNTA OU TEXTO apresentado, fundamentando a resposta EXCLUSIVAMENTE nos extratos canônicos fornecidos.

DIRETRIZES FUNDAMENTAIS:
1. RESPOSTA EXATA AO SOLICITADO:
   - Responda pontualmente ao que foi perguntado, cobrindo integralmente as variáveis e os requisitos da dúvida.

2. DIDÁTICA E DISCRIMINAÇÃO TÉCNICA:
   - Decomponha cada componente, premissa ou conceito mencionado.
   - Destaque dependências estruturais, fórmulas, restrições e regras aplicáveis contidas nos textos de apoio.

3. FIDELIDADE AO VIÉS E À NATUREZA DOS DOCUMENTOS:
   - Adote o vocabulário, o rigor e a metodologia das obras e documentos presentes no dossiê de apoio.
   - Não invente fatos ou regras externas que não estejam respaldadas pela documentação.

4. SEM CITAÇÕES BIBLIOGRÁFICAS OU METADADOS (REGRA ABSOLUTA):
   - Jamais mencione nomes de arquivos (ex: .pdf, .txt), autores, títulos de livros ou números de páginas no texto final.
   - Redija a análise diretamente para o leitor com voz própria de autoridade técnica."""

class RAGOrchestrator:
    def __init__(self, config: AppConfig, indexer: DocumentIndexer, gateway: LLMGateway):
        self.config = config
        self.indexer = indexer
        self.gateway = gateway
        self.dossier_builder = UniversalDossierBuilder()

    def update_config(self, config: AppConfig):
        self.config = config
        self.gateway.update_config(config)

    def format_direct_dossier_report(self, query: str, dossier: Dict[str, Any]) -> str:
        items = dossier["items"]
        lines = [
            f"# Dossiê Técnico Estruturado\n",
            f"> **Consulta:** {query}  ",
            f"> **Modo:** Processamento Local (Mini-LLM FlashRank Cross-Encoder + Poda Extrativa | 0 Tokens Externos)\n\n---\n"
        ]

        lines.append("## 1. Princípios e Fundamentos Canônicos (Extratos Reranqueados)\n")
        for it in items:
            page_txt = f" (pág. {it['page']})" if it.get("page") else ""
            lines.append(f"### Extrato [{it['index']}] — *{it['source']}{page_txt}*\n")
            lines.append(f"{it['content']}\n")
            lines.append("---\n")

        lines.append("\n*Dossiê compilado via recuperação em duas etapas (SQLite FTS5 + Mini-LLM FlashRank).*")
        return "\n".join(lines)

    async def query_and_draft_report(
        self,
        query: str,
        limit_chunks: int = 5,
        only_retrieve: bool = False,
        preferred_slot_id: Optional[str] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        if not only_retrieve and not force_refresh:
            cached = await asyncio.to_thread(self.indexer.get_cached_response, query)
            if cached:
                logger.info(f"[CACHE HIT]: Retornando resposta em cache para: '{query}'")
                return cached

        ranked_chunks = await asyncio.to_thread(self.indexer.search, query, limit=limit_chunks)
        
        if not ranked_chunks:
            return {
                "query": query,
                "found_chunks": [],
                "report": "Nenhum trecho correspondente foi localizado no acervo indexado. Verifique os termos ou consulte a aba de documentos.",
                "llm_meta": None,
                "from_cache": False,
                "tokens_saved": 0
            }

        extracted = self.indexer.extract_universal_query_terms(query)
        keywords = extracted["keywords"]
        dossier = self.dossier_builder.build_dossier(query, ranked_chunks, keywords)

        if only_retrieve:
            direct_report = self.format_direct_dossier_report(query, dossier)
            return {
                "query": query,
                "found_chunks": ranked_chunks,
                "dossier_facts": dossier["query_analysis"],
                "report": direct_report,
                "llm_meta": {
                    "provider": "flashrank_local",
                    "model": "ms-marco-MiniLM-L-12-v2",
                    "slot_label": "Mini-LLM Local (0 Tokens Externos)",
                    "key_used": "onnx_cpu"
                },
                "from_cache": False,
                "tokens_saved": 2500
            }

        context_blocks = []
        for it in dossier["items"]:
            context_blocks.append(f"--- EXTRATO CANÔNICO [{it['index']}] ---\n{it['content']}\n")
        canonical_context = "\n".join(context_blocks)

        raw_char_count = sum(len(c["content"]) for c in ranked_chunks)
        pruned_char_count = len(canonical_context)
        estimated_saved = int(max(0, raw_char_count - pruned_char_count) / 4)

        system_prompt = SYSTEM_PROMPT_CANONICAL.format(
            language=self.config.settings.language
        )

        user_prompt = f"""### PERGUNTA OU SOLICITAÇÃO A SER RESPONDIDA:
\"\"\"
{query}
\"\"\"

### DOCUMENTAÇÃO CANÔNICA DE APOIO (RERANQUEADA VIA CROSS-ENCODER):
{canonical_context}

### DIRETRIZES OBRIGATÓRIAS DE REDAÇÃO:
1. Responda DIRETAMENTE ao que foi solicitado, decompondo os componentes essenciais com clareza didática.
2. Fundamente rigorosamente as asserções nos extratos canônicos acima, preservando a exatidão conceitual dos documentos.
3. Não cite nomes de obras, arquivos (.pdf/.txt), autores ou páginas.
4. Redija de forma fluida e definitiva para o leitor."""

        llm_response = await self.gateway.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            preferred_slot_id=preferred_slot_id
        )

        report_text = llm_response["text"]
        llm_meta = {
            "slot_label": llm_response.get("slot_label"),
            "provider": llm_response["provider"],
            "model": llm_response["model"],
            "reasoning_effort": llm_response.get("reasoning_effort"),
            "key_used": llm_response["key_used"],
            "tokens_saved_by_pruning": estimated_saved,
            "reranker": "FlashRank MiniLM-L-12-v2"
        }

        tokens_for_cache = estimated_saved + int(len(report_text) / 4) + 1000
        await asyncio.to_thread(
            self.indexer.store_cached_response,
            raw_query=query,
            report=report_text,
            found_chunks=ranked_chunks,
            llm_meta=llm_meta,
            tokens_saved=tokens_for_cache
        )

        return {
            "query": query,
            "found_chunks": ranked_chunks,
            "dossier_facts": dossier["query_analysis"],
            "report": report_text,
            "llm_meta": llm_meta,
            "from_cache": False,
            "tokens_saved": estimated_saved
        }
