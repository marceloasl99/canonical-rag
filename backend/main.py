import os
import uuid
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import load_config, save_config, AppConfig, APIKeyEntry, ActiveModelSlot
from .llm_gateway import LLMGateway
from .indexer import DocumentIndexer
from .rag_orchestrator import RAGOrchestrator

app = FastAPI(title="Canonical RAG Gateway", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialização dos serviços
config = load_config()
gateway = LLMGateway(config)
indexer = DocumentIndexer()
orchestrator = RAGOrchestrator(config, indexer, gateway)

# Modelos de entrada para as rotas
class QueryRequest(BaseModel):
    query: str
    limit_chunks: int = 5
    only_retrieve: bool = False
    preferred_slot_id: Optional[str] = None

class AddKeyRequest(BaseModel):
    key: str
    label: str = ""

class FetchModelsRequest(BaseModel):
    key: Optional[str] = None
    key_id: Optional[str] = None

class UpdateSettingsRequest(BaseModel):
    documents_folder: str
    default_provider: str
    round_robin_enabled: bool
    fallback_enabled: bool
    temperature: float
    max_tokens: int
    language: str

class UpdateSlotRequest(BaseModel):
    id: str
    label: str
    provider: str
    model_id: str
    api_key_id: Optional[str] = None
    reasoning_effort: str = "none"
    temperature: float = 0.1
    active: bool = True

class UpdateProviderConfigRequest(BaseModel):
    account_id: Optional[str] = None
    base_url: Optional[str] = None

@app.post("/api/providers/{provider}/config")
async def update_provider_config(provider: str, req: UpdateProviderConfigRequest):
    if provider not in config.providers:
        raise HTTPException(status_code=404, detail="Provedor não encontrado")
    if req.account_id is not None:
        config.providers[provider].account_id = req.account_id
    if req.base_url is not None:
        config.providers[provider].base_url = req.base_url
    save_config(config)
    orchestrator.update_config(config)
    return {"status": "ok", "provider": config.providers[provider]}

@app.get("/api/config")
async def get_config():
    cfg_dict = config.model_dump()
    for p in cfg_dict["providers"].values():
        for k in p["keys"]:
            original = k["key"]
            if len(original) > 8:
                k["masked_key"] = f"{original[:4]}...{original[-4:]}"
            else:
                k["masked_key"] = "***"
    return cfg_dict

@app.post("/api/settings")
async def update_settings(req: UpdateSettingsRequest):
    config.settings.documents_folder = req.documents_folder
    config.settings.default_provider = req.default_provider
    config.settings.round_robin_enabled = req.round_robin_enabled
    config.settings.fallback_enabled = req.fallback_enabled
    config.settings.temperature = req.temperature
    config.settings.max_tokens = req.max_tokens
    config.settings.language = req.language
    save_config(config)
    orchestrator.update_config(config)
    return {"status": "ok", "settings": config.settings}

# Buscar modelos existentes com a chave de API
@app.post("/api/providers/{provider}/fetch-models")
async def fetch_provider_models(provider: str, req: FetchModelsRequest):
    if provider not in config.providers:
        raise HTTPException(status_code=404, detail="Provedor não encontrado")

    api_key = req.key
    if not api_key and req.key_id:
        for k in config.providers[provider].keys:
            if k.id == req.key_id:
                api_key = k.key
                break

    if not api_key:
        active_keys = [k for k in config.providers[provider].keys if k.active and k.key.strip()]
        if active_keys:
            api_key = active_keys[0].key

    if not api_key:
        raise HTTPException(status_code=400, detail="Nenhuma chave de API informada ou cadastrada para consultar os modelos")

    try:
        models = await gateway.fetch_available_models(provider, api_key)
        return {"status": "ok", "provider": provider, "models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar modelos na API do provedor: {e}")

# Gerenciar slots de modelos ativos
@app.post("/api/active-models")
async def update_active_models(slots: List[UpdateSlotRequest]):
    new_slots = []
    for s in slots:
        new_slots.append(ActiveModelSlot(
            id=s.id,
            label=s.label,
            provider=s.provider,
            model_id=s.model_id,
            api_key_id=s.api_key_id if s.api_key_id != "" else None,
            reasoning_effort=s.reasoning_effort,
            temperature=s.temperature,
            active=s.active
        ))
    config.active_models = new_slots
    save_config(config)
    orchestrator.update_config(config)
    return {"status": "ok", "active_models": config.active_models}

@app.post("/api/active-models/{slot_id}/toggle")
async def toggle_active_model_slot(slot_id: str):
    for s in config.active_models:
        if s.id == slot_id:
            s.active = not s.active
            save_config(config)
            orchestrator.update_config(config)
            return {"status": "ok", "slot_id": slot_id, "active": s.active}
    raise HTTPException(status_code=404, detail="Slot de modelo não encontrado")

@app.post("/api/providers/{provider}/toggle")
async def toggle_provider(provider: str):
    if provider not in config.providers:
        raise HTTPException(status_code=404, detail="Provedor não encontrado")
    config.providers[provider].active = not config.providers[provider].active
    save_config(config)
    orchestrator.update_config(config)
    return {"status": "ok", "active": config.providers[provider].active}

@app.post("/api/providers/{provider}/keys")
async def add_api_key(provider: str, req: AddKeyRequest):
    if provider not in config.providers:
        raise HTTPException(status_code=404, detail="Provedor não encontrado")
    
    clean_key = req.key.strip()
    if not clean_key:
        raise HTTPException(status_code=400, detail="Chave não pode ser vazia")

    new_entry = APIKeyEntry(
        id=str(uuid.uuid4())[:8],
        key=clean_key,
        label=req.label or f"Chave {len(config.providers[provider].keys) + 1}",
        active=True
    )
    config.providers[provider].keys.append(new_entry)
    save_config(config)
    orchestrator.update_config(config)
    return {"status": "ok", "key_id": new_entry.id}

@app.delete("/api/providers/{provider}/keys/{key_id}")
async def delete_api_key(provider: str, key_id: str):
    if provider not in config.providers:
        raise HTTPException(status_code=404, detail="Provedor não encontrado")
    keys = config.providers[provider].keys
    config.providers[provider].keys = [k for k in keys if k.id != key_id]
    save_config(config)
    orchestrator.update_config(config)
    return {"status": "ok"}

@app.post("/api/providers/{provider}/keys/{key_id}/toggle")
async def toggle_api_key(provider: str, key_id: str):
    if provider not in config.providers:
        raise HTTPException(status_code=404, detail="Provedor não encontrado")
    for k in config.providers[provider].keys:
        if k.id == key_id:
            k.active = not k.active
            save_config(config)
            orchestrator.update_config(config)
            return {"status": "ok", "active": k.active}
    raise HTTPException(status_code=404, detail="Chave não encontrada")

@app.post("/api/index")
async def trigger_indexing(background_tasks: BackgroundTasks):
    folder = config.settings.documents_folder
    def run_index():
        indexer.scan_and_index_folder(
            folder_path=folder,
            chunk_size=config.settings.chunk_size,
            chunk_overlap=config.settings.chunk_overlap
        )
    background_tasks.add_task(run_index)
    return {"status": "ok", "message": f"Indexação iniciada em segundo plano para: {folder}"}

@app.get("/api/system-telemetry")
async def get_system_telemetry():
    import psutil
    import shutil
    try:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        disk = shutil.disk_usage("/")
        stats = indexer.get_stats()
        cache_stats = stats.get("cache_stats", {})
        
        return {
            "status": "ok",
            "cpu_percent": round(cpu, 1),
            "ram_used_mb": mem.used // (1024 * 1024),
            "ram_total_mb": mem.total // (1024 * 1024),
            "ram_percent": round(mem.percent, 1),
            "disk_used_gb": round(disk.used / (1024 ** 3), 1),
            "disk_total_gb": round(disk.total / (1024 ** 3), 1),
            "disk_percent": round(disk.used / disk.total * 100, 1),
            "tokens_saved_total": cache_stats.get("total_tokens_saved", 0),
            "cache_hits_total": cache_stats.get("total_cache_hits", 0),
            "cached_queries_count": cache_stats.get("cached_queries_count", 0),
            "total_documents": stats.get("total_documents", 0),
            "total_chunks": stats.get("total_chunks", 0)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/stats")
async def get_stats():
    return indexer.get_stats()

@app.get("/api/documents")
async def list_documents(search: Optional[str] = None, doc_type: Optional[str] = None):
    return indexer.get_documents(search=search, doc_type=doc_type)

@app.post("/api/query")
async def execute_query(req: QueryRequest):
    import logging
    log = logging.getLogger("APIQuery")
    log.info(f"--> [QUERY RECEBIDA]: '{req.query}' (slot: {req.preferred_slot_id}, only_retrieve: {req.only_retrieve})")
    try:
        result = await orchestrator.query_and_draft_report(
            query=req.query,
            limit_chunks=req.limit_chunks,
            only_retrieve=req.only_retrieve,
            preferred_slot_id=req.preferred_slot_id
        )
        log.info(f"<-- [QUERY CONCLUIDA]: {len(result.get('found_chunks', []))} chunks recuperados.")
        return result
    except Exception as e:
        log.error(f"[!] ERRO NA QUERY '{req.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Servir a página Web na raiz "/"
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def serve_index():
    index_path = os.getenv("INDEX_HTML_PATH", "frontend/index.html")
    if not os.path.exists(index_path):
        index_path = "/opt/canonical-rag/frontend/index.html"
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "80"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
