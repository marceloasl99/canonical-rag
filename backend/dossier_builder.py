import re
from typing import List, Dict, Any

class UniversalDossierBuilder:
    """
    Construtor de Dossiê Estruturado Agnóstico a Domínio.
    Extrai variáveis conceituais, poda sentenças acessórias e estrutura
    a informação para geração local ou envio de baixíssimo custo à LLM.
    """

    @staticmethod
    def extract_entities_and_variables(query: str) -> Dict[str, Any]:
        q = query.strip()
        quoted = re.findall(r'"([^"]+)"', q)
        cleaned = re.sub(r'"[^"]+"', ' ', q)
        
        words = re.findall(r'\b[a-zA-Z0-9áéíóúâêîôûãõçüÁÉÍÓÚÂÊÎÔÛÃÕÇÜ]{3,}\b', cleaned)
        
        return {
            "query_original": q,
            "termos_exatos": quoted,
            "variaveis_principais": words[:8]
        }

    @staticmethod
    def prune_sentences(content: str, keywords: List[str], max_sentences: int = 4) -> str:
        raw_sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        if len(raw_sentences) <= 2:
            return content.strip()

        scored = []
        kw_set = [k.lower() for k in keywords if len(k) > 2]

        for idx, s in enumerate(raw_sentences):
            s_clean = s.strip()
            if not s_clean:
                continue
            s_lower = s_clean.lower()
            score = sum(1 for kw in kw_set if kw in s_lower)
            if idx == 0 and score > 0:
                score += 0.5
            scored.append((score, idx, s_clean))

        scored_by_rel = sorted(scored, key=lambda x: x[0], reverse=True)
        selected = [item for item in scored_by_rel if item[0] > 0][:max_sentences]
        if not selected:
            selected = scored[:2]

        selected.sort(key=lambda x: x[1])
        return " ".join([item[2] for item in selected])

    def build_dossier(self, query: str, ranked_chunks: List[Dict[str, Any]], keywords: List[str]) -> Dict[str, Any]:
        analysis = self.extract_entities_and_variables(query)
        items = []
        for i, c in enumerate(ranked_chunks, 1):
            pruned_text = self.prune_sentences(c["content"], keywords, max_sentences=4)
            score_txt = f" [Score: {c.get('rerank_score', 0):.2f}]" if "rerank_score" in c else ""
            items.append({
                "index": i,
                "source": f"{c.get('filename')}{score_txt}",
                "filename": c.get("filename"),
                "page": c.get("page_number"),
                "content": pruned_text
            })

        return {
            "query_analysis": analysis,
            "items": items,
            "raw_chunks": ranked_chunks
        }

    def format_dual_report(self, query: str, dossier: Dict[str, Any]) -> str:
        """
        Gera relatório técnico com DUAS SEÇÕES INDEPENDENTES:
        1. Síntese Técnica Contínua (Início, Meio e Fim)
        2. Análise Estruturada em Tópicos
        3. Extratos Canônicos Fundamentais (Fontes Reranqueadas)
        100% local, autônomo e com zero consumo de tokens de LLM.
        """
        analysis = dossier.get("query_analysis", {})
        items = dossier.get("items", [])

        sentences = []
        for it in items:
            raw_s = re.split(r'(?<=[.!?])\s+', it["content"].strip())
            for s in raw_s:
                s_clean = s.strip().rstrip('.')
                if len(s_clean) > 25 and s_clean not in sentences:
                    sentences.append(s_clean)

        # SEÇÃO 1: SÍNTESE TÉCNICA CONTÍNUA (INÍCIO, MEIO E FIM)
        variables = analysis.get("variaveis_principais", [])
        exact_terms = analysis.get("termos_exatos", [])

        intro_phrases = []
        if exact_terms:
            terms_str = ", ".join([f"'{t}'" for t in exact_terms])
            intro_phrases.append(f"A análise documental referente à consulta **'{query}'** estabelece, como critério preliminar, o enquadramento direto dos termos de referência {terms_str}.")
        elif variables:
            vars_str = ", ".join(variables[:3])
            intro_phrases.append(f"A análise documental referente a **'{query}'** estabelece, como premissa de partida, a fundamentação conceitual articulada em torno de **{vars_str}**.")
        else:
            intro_phrases.append(f"A análise documental referente a **'{query}'** estabelece, como premissa de partida, a estrita correlação entre os registros canônicos indexados e os requisitos da consulta.")

        if sentences:
            intro_phrases.append(f"Sob os registros primários de referência, verifica-se inicialmente que {sentences[0][0].lower() + sentences[0][1:]}.")

        intro_text = " ".join(intro_phrases)

        # Meio (Desenvolvimento Coeso com Conectivos Técnicos em Português)
        connectives = [
            "Sob a ótica da fundamentação documental, ",
            "Nesse sentido, a análise aprofundada esclarece que ",
            "Ademais, no que tange aos mecanismos operacionais e princípios estruturais, observa-se que ",
            "Em complemento às diretrizes apontadas, verifica-se que ",
            "Por conseguinte, no desdobramento técnico, "
        ]

        body_phrases = []
        for i, s in enumerate(sentences[1:5]):
            conn = connectives[i % len(connectives)]
            s_formatted = s[0].lower() + s[1:] if len(s) > 1 else s
            body_phrases.append(f"{conn}{s_formatted}.")

        body_text = " ".join(body_phrases) if body_phrases else "Os registros documentados consolidam as definições e critérios normativos fundamentais para o tema."

        # Fim (Conclusão Técnica & Fechamento)
        conclusion_text = f"Em síntese, a integração articulada desses fatores atende conclusivamente à solicitação referente a '{query}', consolidando uma interpretação coesa e estritamente respaldada no acervo canônico documentado."

        continuous_section = f"{intro_text}\n\n{body_text}\n\n{conclusion_text}"

        # SEÇÃO 2: ANÁLISE ESTRUTURADA EM TÓPICOS
        topic_points = []
        
        # Conceito Central & Resposta Direta
        first_premise = sentences[0] if sentences else query
        topic_points.append(f"- **📌 Conceito Central & Resposta Direta:** {first_premise}.")

        # Mecanismos & Fundamentação Técnica
        mech_sentences = sentences[1:3] if len(sentences) > 2 else sentences[:1]
        if mech_sentences:
            topic_points.append(f"- **⚙️ Mecanismos & Fundamentação Técnica:** {mech_sentences[0]}.")
            if len(mech_sentences) > 1:
                topic_points.append(f"- **⚙️ Dinâmica e Diretriz Estrutural:** {mech_sentences[1]}.")

        # Variáveis, Condições e Pontos de Atenção
        if exact_terms:
            topic_points.append(f"- **💡 Termos e Condições Mandatórias:** Requisito de correspondência com {', '.join([f'`{t}`' for t in exact_terms])}.")
        if len(sentences) > 3:
            topic_points.append(f"- **💡 Ponto de Atenção & Nuance:** {sentences[3]}.")
        else:
            topic_points.append(f"- **💡 Ponto de Atenção & Nuance:** Requer avaliação integrada do acervo de referência, evitando conclusões isoladas fora do contexto documental.")

        # Diretriz Prática & Conclusão Executiva
        topic_points.append(f"- **🎯 Diretriz Prática & Conclusão Executiva:** Aplicação fundamentada dos princípios técnicos analisados, assegurando conformidade com a base de conhecimento canônica.")

        topics_section = "\n".join(topic_points)

        # SEÇÃO 3: EXTRATOS FONTES
        excerpt_blocks = []
        for it in items:
            page_txt = f" (pág. {it['page']})" if it.get("page") else ""
            excerpt_blocks.append(f"### Extrato [{it['index']}] — *{it['source']}{page_txt}*\n{it['content']}\n")

        excerpts_section = "\n---\n".join(excerpt_blocks)

        report = f"""# Dossiê Técnico Canônico

> **Consulta:** {query}  
> **Modo de Processamento:** Motor Local Autônomo (FlashRank Cross-Encoder + Poda Extrativa | 0 Tokens Externos)

---

## 1. Síntese Técnica Contínua (Início, Meio e Fim)

{continuous_section}

---

## 2. Análise Estruturada em Tópicos

{topics_section}

---

## 3. Extratos Canônicos Fundamentais (Fontes Reranqueadas)

{excerpts_section}

*Dossiê compilado pelo motor canônico via busca semântica em duas etapas (SQLite FTS5 + Mini-LLM FlashRank).*"""

        return report
