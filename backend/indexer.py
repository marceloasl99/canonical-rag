import os
import sqlite3
import re
import unicodedata
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("DocumentIndexer")

DB_PATH = os.getenv("INDEX_DB_PATH", "documents_index.db")
MODELS_CACHE_DIR = os.getenv("MODELS_CACHE_DIR", "models_cache")

# Instância preguiçosa do Reranker FlashRank (Mini-LLM local ONNX)
_ranker_instance = None

def get_ranker():
    global _ranker_instance
    if _ranker_instance is None:
        try:
            from flashrank import Ranker
            os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
            _ranker_instance = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=MODELS_CACHE_DIR)
            logger.info("FlashRank Reranker (ms-marco-MiniLM-L-12-v2) inicializado.")
        except Exception as e:
            logger.warning(f"FlashRank não disponível: {e}. Usando BM25 nativo.")
            _ranker_instance = False
    return _ranker_instance

STOPWORDS = {
    "me", "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com",
    "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas",
    "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito",
    "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso",
    "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem",
    "nas", "meu", "esse", "eles", "estão", "você", "tinha", "foram", "essa", "num",
    "nem", "suas", "meu", "às", "minha", "têm", "numa", "pelos", "elas", "havia",
    "seja", "qual", "será", "nós", "tenho", "lhe", "deles", "essas", "esses", "pelas",
    "este", "fosse", "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas",
    "teu", "tua", "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "dela", "delas",
    "esta", "estes", "estas", "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo",
    "estou", "está", "estamos", "estão", "estive", "esteve", "estivemos", "estiveram",
    "explique", "diga", "sobre", "analise", "resumo", "significa", "significado", "qual",
    "quais", "como", "funciona", "o que", "oque", "quero", "faca", "faça"
}

class DocumentIndexer:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE,
                    filepath TEXT,
                    filesize INTEGER,
                    pages_count INTEGER,
                    category TEXT DEFAULT 'general',
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    doc_id UNINDEXED,
                    filename,
                    page_number UNINDEXED,
                    chunk_index UNINDEXED,
                    category UNINDEXED,
                    content,
                    tokenize = 'unicode61 remove_diacritics 2'
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE,
                    normalized_query TEXT,
                    raw_query TEXT,
                    report TEXT,
                    found_chunks_json TEXT,
                    llm_meta_json TEXT,
                    tokens_saved INTEGER DEFAULT 0,
                    hits INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_stats (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    total_tokens_saved INTEGER DEFAULT 0,
                    total_cache_hits INTEGER DEFAULT 0,
                    total_api_calls INTEGER DEFAULT 0
                )
            """)
            cursor.execute("INSERT OR IGNORE INTO cache_stats (id, total_tokens_saved, total_cache_hits, total_api_calls) VALUES (1, 0, 0, 0)")
            conn.commit()

    @staticmethod
    def normalize_query_for_cache(query: str) -> str:
        nfkd = unicodedata.normalize('NFKD', query.lower())
        without_accents = "".join([c for c in nfkd if not unicodedata.combining(c)])
        tokens = re.findall(r'\b[a-z0-9]+\b', without_accents)
        filtered = [t for t in tokens if t not in STOPWORDS]
        return " ".join(sorted(filtered))

    @staticmethod
    def compute_query_hash(normalized_query: str) -> str:
        return hashlib.sha256(normalized_query.encode('utf-8')).hexdigest()

    def get_cached_response(self, raw_query: str) -> Optional[Dict[str, Any]]:
        norm_q = self.normalize_query_for_cache(raw_query)
        if not norm_q:
            return None
        q_hash = self.compute_query_hash(norm_q)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM query_cache WHERE query_hash = ?", (q_hash,))
            row = cursor.fetchone()
            if row:
                cursor.execute("""
                    UPDATE query_cache 
                    SET hits = hits + 1, last_accessed = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (row["id"],))
                cursor.execute("""
                    UPDATE cache_stats 
                    SET total_cache_hits = total_cache_hits + 1,
                        total_tokens_saved = total_tokens_saved + ?
                    WHERE id = 1
                """, (row["tokens_saved"],))
                conn.commit()

                return {
                    "query": raw_query,
                    "found_chunks": json.loads(row["found_chunks_json"]),
                    "report": row["report"],
                    "llm_meta": json.loads(row["llm_meta_json"]) if row["llm_meta_json"] else None,
                    "from_cache": True,
                    "tokens_saved": row["tokens_saved"],
                    "cache_hits": row["hits"] + 1
                }
        return None

    def store_cached_response(self, raw_query: str, report: str, found_chunks: List[Dict[str, Any]], llm_meta: Dict[str, Any], tokens_saved: int = 1500):
        norm_q = self.normalize_query_for_cache(raw_query)
        if not norm_q:
            return
        q_hash = self.compute_query_hash(norm_q)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO query_cache (query_hash, normalized_query, raw_query, report, found_chunks_json, llm_meta_json, tokens_saved)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(query_hash) DO UPDATE SET
                    report = excluded.report,
                    found_chunks_json = excluded.found_chunks_json,
                    llm_meta_json = excluded.llm_meta_json,
                    last_accessed = CURRENT_TIMESTAMP
            """, (
                q_hash,
                norm_q,
                raw_query,
                report,
                json.dumps(found_chunks, ensure_ascii=False),
                json.dumps(llm_meta, ensure_ascii=False) if llm_meta else None,
                tokens_saved
            ))
            cursor.execute("UPDATE cache_stats SET total_api_calls = total_api_calls + 1 WHERE id = 1")
            conn.commit()

    @staticmethod
    def extract_universal_query_terms(query: str) -> Dict[str, Any]:
        q = query.strip()
        exact_phrases = re.findall(r'"([^"]+)"', q)
        remaining_q = re.sub(r'"[^"]+"', ' ', q).lower()
        words = [w for w in re.findall(r'\b[a-zA-Z0-9áéíóúâêîôûãõçüÁÉÍÓÚÂÊÎÔÛÃÕÇÜ]{2,}\b', remaining_q)]
        meaningful = [w for w in words if w.lower() not in STOPWORDS]
        bigrams = [f"{meaningful[i]} {meaningful[i+1]}" for i in range(len(meaningful) - 1)]
        return {
            "exact_phrases": exact_phrases,
            "keywords": meaningful,
            "bigrams": bigrams
        }

    def determine_category(self, filepath: str) -> str:
        parts = filepath.replace("\\", "/").split("/")
        if len(parts) >= 2:
            parent = parts[-2].lower()
            if parent not in ["documentos", "docs", "files", "acervo", "data"]:
                return parent
        return "general"

    def scan_and_index_folder(self, folder_path: str, chunk_size: int = 800, chunk_overlap: int = 150) -> Dict[str, Any]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            return {"status": "ok", "message": f"Pasta criada: {folder_path}.", "indexed_files": 0, "total_chunks": 0}

        supported_extensions = (".pdf", ".txt", ".md")
        files_to_process = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    files_to_process.append(os.path.join(root, file))

        indexed_files = 0
        total_chunks = 0

        for filepath in files_to_process:
            filename = os.path.basename(filepath)
            filesize = os.path.getsize(filepath)
            category = self.determine_category(filepath)

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, filesize FROM documents WHERE filename = ?", (filename,))
                row = cursor.fetchone()
                if row and row["filesize"] == filesize:
                    continue

                if row:
                    doc_id = row["id"]
                    cursor.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (str(doc_id),))
                    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                    conn.commit()

            pages = self._extract_text(filepath)
            if not pages:
                continue

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO documents (filename, filepath, filesize, pages_count, category)
                    VALUES (?, ?, ?, ?, ?)
                """, (filename, filepath, filesize, len(pages), category))
                doc_id = cursor.lastrowid

                file_chunks = 0
                for page_num, text in pages:
                    chunks = self._chunk_text(text, chunk_size, chunk_overlap)
                    for c_idx, chunk in enumerate(chunks):
                        cursor.execute("""
                            INSERT INTO chunks_fts (doc_id, filename, page_number, chunk_index, category, content)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (str(doc_id), filename, page_num, c_idx, category, chunk))
                        file_chunks += 1

                conn.commit()
                indexed_files += 1
                total_chunks += file_chunks

        return {
            "status": "ok",
            "indexed_files": indexed_files,
            "total_chunks": total_chunks,
            "total_files_found": len(files_to_process)
        }

    def _extract_text(self, filepath: str) -> List[tuple]:
        ext = os.path.splitext(filepath)[1].lower()
        pages = []
        if ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(filepath)
                for i, page in enumerate(doc):
                    text = page.get_text("text").strip()
                    if text:
                        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
                        pages.append((i + 1, text))
                doc.close()
                return pages
            except Exception:
                pass

            try:
                from pypdf import PdfReader
                reader = PdfReader(filepath)
                for i, page in enumerate(reader.pages):
                    text = reader.pages[i].extract_text()
                    if text and text.strip():
                        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text.strip())
                        pages.append((i + 1, text))
                return pages
            except Exception as e:
                logger.error(f"Erro ao ler PDF {filepath}: {e}")
                return []

        elif ext in [".txt", ".md"]:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                    if content:
                        pages.append((1, content))
            except Exception as e:
                logger.error(f"Erro ao ler texto {filepath}: {e}")
        return pages

    def _chunk_text(self, text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        raw_paras = re.split(r'\n{2,}|\n(?=[A-Z0-9\-\•\*\t\s]{2,}\b)', text)
        paras = [p.strip() for p in raw_paras if p.strip()]

        chunks = []
        curr_chunk = []
        curr_len = 0
        for p in paras:
            p_len = len(p)
            if curr_len + p_len > chunk_size and curr_chunk:
                chunks.append("\n\n".join(curr_chunk))
                if chunk_overlap > 0 and len(curr_chunk[-1]) < chunk_overlap * 2:
                    curr_chunk = [curr_chunk[-1]]
                    curr_len = len(curr_chunk[0])
                else:
                    curr_chunk = []
                    curr_len = 0
            curr_chunk.append(p)
            curr_len += p_len

        if curr_chunk:
            chunks.append("\n\n".join(curr_chunk))
        return chunks if chunks else [text]

    def search_universal(self, raw_query: str, max_total_chunks: int = 20, category_filter: str = None) -> List[Dict[str, Any]]:
        extracted = self.extract_universal_query_terms(raw_query)
        exact_phrases = extracted["exact_phrases"]
        keywords = extracted["keywords"]
        bigrams = extracted["bigrams"]

        with self._get_connection() as conn:
            results = []
            seen = set()

            def run_subquery(fts_expr: str, max_take: int, score_multiplier: float = 1.0):
                try:
                    cursor = conn.cursor()
                    sql = "SELECT filename, page_number, chunk_index, category, content, rank FROM chunks_fts WHERE chunks_fts MATCH ?"
                    params = [fts_expr]
                    if category_filter and category_filter != "all":
                        sql += " AND category = ?"
                        params.append(category_filter)
                    sql += f" ORDER BY rank LIMIT {max_take * 2}"
                    cursor.execute(sql, tuple(params))
                    for r in cursor.fetchall():
                        key = f"{r['filename']}_{r['page_number']}_{r['chunk_index']}"
                        if key not in seen:
                            seen.add(key)
                            results.append({
                                "filename": r["filename"],
                                "page_number": r["page_number"],
                                "chunk_index": r["chunk_index"],
                                "category": r["category"],
                                "content": r["content"],
                                "score": r["rank"] * score_multiplier
                            })
                            if len(results) >= max_total_chunks:
                                return True
                except Exception:
                    pass
                return False

            for ep in exact_phrases:
                if run_subquery(f'"{ep}"', max_take=4, score_multiplier=3.0):
                    return results[:max_total_chunks]

            for bg in bigrams[:3]:
                if run_subquery(f'"{bg}"', max_take=3, score_multiplier=2.5):
                    return results[:max_total_chunks]

            if len(keywords) >= 2:
                t1, t2 = keywords[0], keywords[1]
                run_subquery(f'NEAR("{t1}" "{t2}", 15)', max_take=4, score_multiplier=2.0)

            if len(keywords) >= 2 and len(results) < max_total_chunks:
                and_query = " AND ".join(f'"{t}"*' for t in keywords[:5])
                run_subquery(and_query, max_take=max_total_chunks - len(results), score_multiplier=1.5)

            if len(results) < max_total_chunks and keywords:
                or_query = " OR ".join(f'"{t}"*' for t in keywords[:4])
                run_subquery(or_query, max_take=max_total_chunks - len(results), score_multiplier=1.0)

            return results[:max_total_chunks]

    def rerank_passages(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidate_chunks or len(candidate_chunks) <= 1:
            return candidate_chunks[:top_k]

        ranker = get_ranker()
        if not ranker:
            return candidate_chunks[:top_k]

        try:
            from flashrank import RerankRequest
            passages = [
                {"id": idx, "text": c["content"], "meta": c}
                for idx, c in enumerate(candidate_chunks)
            ]
            rerank_req = RerankRequest(query=query, passages=passages)
            ranked_results = ranker.rerank(rerank_req)

            sorted_chunks = []
            for r in ranked_results[:top_k]:
                orig_meta = r.get("meta") or candidate_chunks[r["id"]]
                orig_meta["rerank_score"] = float(r.get("score", 0.0))
                sorted_chunks.append(orig_meta)

            return sorted_chunks
        except Exception as e:
            logger.warning(f"Erro no rerank FlashRank: {e}")
            return candidate_chunks[:top_k]

    def search(self, query: str, limit: int = 5, category_filter: str = None) -> List[Dict[str, Any]]:
        candidates = self.search_universal(query, max_total_chunks=20, category_filter=category_filter)
        return self.rerank_passages(query, candidates, top_k=limit)

    def get_documents(self, search: str = None, category: str = None, limit: int = 1000) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            conditions = []
            params = []
            if search:
                conditions.append("filename LIKE ?")
                params.append(f"%{search.strip()}%")
            if category and category != "all":
                conditions.append("category = ?")
                params.append(category)

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            sql = f"""
                SELECT id, filename, filepath, filesize, pages_count, category, indexed_at
                FROM documents
                {where_clause}
                ORDER BY filename COLLATE NOCASE ASC
                LIMIT ?
            """
            params.append(limit)
            cursor.execute(sql, tuple(params))
            return [dict(r) for r in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            doc_count = cursor.fetchone()["count"]
            cursor.execute("SELECT COUNT(*) as count FROM chunks_fts")
            chunk_count = cursor.fetchone()["count"]
            cursor.execute("SELECT category, COUNT(*) as count FROM documents GROUP BY category")
            by_cat = {r["category"]: r["count"] for r in cursor.fetchall()}
            cursor.execute("SELECT id, filename, pages_count, filesize, category, indexed_at FROM documents ORDER BY filename COLLATE NOCASE ASC")
            all_docs = [dict(r) for r in cursor.fetchall()]
            
            cursor.execute("SELECT total_tokens_saved, total_cache_hits, total_api_calls FROM cache_stats WHERE id = 1")
            row_cs = cursor.fetchone()
            cache_stats = dict(row_cs) if row_cs else {"total_tokens_saved": 0, "total_cache_hits": 0, "total_api_calls": 0}

            cursor.execute("SELECT COUNT(*) as cached_queries FROM query_cache")
            cache_stats["cached_queries_count"] = cursor.fetchone()["cached_queries"]

        return {
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "documents_by_category": by_cat,
            "documents": all_docs,
            "cache_stats": cache_stats
        }
