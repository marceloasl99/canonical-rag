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
