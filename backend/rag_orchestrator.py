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

ESTRUTURA MANDATÓRIA DA RESPOSTA (DUAS SEÇÕES INDEPENDENTES):
Sua resposta DEVE ser rigorosamente estruturada em duas seções principais independentes:

## 1. Síntese Técnica Contínua (Início, Meio e Fim)
- Redija uma dissertação coesa, fluida e articulada em parágrafos contínuos.
- INÍCIO: Introdução didática que contextualiza a dúvida, define o tema e estabelece a premissa de partida.
- MEIO: Desenvolvimento aprofundado encadeando logicamente os extratos canônicos, demonstrando causas, efeitos e dependências estruturais com conectivos naturais em português.
- FIM: Conclusão consistente que amarra a análise e apresenta a síntese definitiva.

---

## 2. Análise Estruturada em Tópicos
- Apresente a mesma fundamentação decomposta em tópicos analíticos claros e objetivos:
  - **📌 Conceito Central & Resposta Direta:** resposta direta e pontual.
  - **⚙️ Mecanismos & Fundamentação Técnica:** como atua, regras essenciais e dinâmicas conceituais dos extratos.
  - **💡 Variáveis, Condições e Pontos de Atenção:** fatores críticos, restrições e dependências de contexto.
  - **🎯 Diretriz Prática & Conclusão Executiva:** síntese executiva e aplicação concreta.

DIRETRIZES DE OURO:
1. RESPOSTA EXATA: Cubra com rigor todos os componentes citados na pergunta.
2. FIDELIDADE AOS DOCUMENTOS: Adote o vocabulário e rigor dos textos canônicos, sem alucinar fatos não documentados.
3. SEM CITAÇÕES BIBLIOGRÁFICAS OU METADADOS (REGRA ABSOLUTA): Jamais mencione nomes de arquivos (ex: .pdf, .txt), autores, títulos de livros ou páginas. Redija com autoridade própria direta para o leitor."""

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
        """
        Formata o Dossiê Técnico Estruturado com DUAS SEÇÕES INDEPENDENTES
        (1. Síntese Contínua com início, meio e fim; 2. Análise Estruturada em Tópicos)
        diretamente pelo motor com ZERO tokens de LLM.
        """
        return self.dossier_builder.format_dual_report(query, dossier)

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

### DIRETRIZES OBRIGATÓRIAS DE REDAÇÃO (DUAS SEÇÕES INDEPENDENTES):
1. Sua resposta DEVE conter rigorosamente as duas seções:
   - '## 1. Síntese Técnica Contínua (Início, Meio e Fim)': texto dissertativo contínuo, iniciando com introdução didática que contextualiza a dúvida, desenvolvimento aprofundado encadeando causas e efeitos com conectivos fluídos em português, e conclusão técnica definitiva.
   - '## 2. Análise Estruturada em Tópicos': decomposição analítica em marcadores objetivos (📌 Conceito Central & Resposta Direta, ⚙️ Mecanismos & Fundamentação Técnica, 💡 Variáveis, Condições e Pontos de Atenção, 🎯 Diretriz Prática & Conclusão Executiva).
2. Responda DIRETAMENTE ao que foi solicitado, decompondo os componentes essenciais com clareza didática.
3. Fundamente rigorosamente as asserções nos extratos canônicos acima, preservando a exatidão conceitual dos documentos.
4. Jamais cite nomes de obras, arquivos (.pdf/.txt), autores ou páginas.
5. Redija de forma fluida e definitiva para o leitor."""

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
