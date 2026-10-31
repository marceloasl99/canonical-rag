import json
import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

CONFIG_FILE = os.getenv("CONFIG_PATH", "/opt/canonical-rag/config.json")

class APIKeyEntry(BaseModel):
    id: str
    key: str
    label: str = ""
    active: bool = True
    usage_count: int = 0
    errors_count: int = 0

class ProviderConfig(BaseModel):
    name: str # "gemini", "claude", "groq", "openrouter", "cloudflare", "openai"
    label: str
    active: bool = True
    default_model: str
    available_models: List[str] = []
    base_url: str = ""
    account_id: str = "" # Necessário para Cloudflare Workers AI
    keys: List[APIKeyEntry] = []

class ActiveModelSlot(BaseModel):
    id: str                          # "slot-1", "slot-2", "slot-3", etc.
    label: str                       # ex: "Modelo Ativo 1 (Principal)"
    provider: str                    # "gemini", "claude", "groq", "openrouter", "cloudflare", "openai"
    model_id: str                    # ex: "gemini-2.5-flash", "google/gemini-2.0-flash-exp:free"
    api_key_id: Optional[str] = None # None significa "usar qualquer chave ativa do provedor"
    reasoning_effort: str = "none"   # "none", "low", "medium", "high"
    temperature: float = 0.1
    active: bool = True
    usage_count: int = 0
    errors_count: int = 0

class SystemSettings(BaseModel):
    documents_folder: str = "/dados/documentos"
    chunk_size: int = 800
    chunk_overlap: int = 150
    default_provider: str = "gemini"
    round_robin_enabled: bool = True
    fallback_enabled: bool = True
    strict_grounding: bool = True
    temperature: float = 0.1
    max_tokens: int = 4000
    reasoning_effort: str = "none" # "none", "low", "medium", "high"
    language: str = "pt-BR"

class AppConfig(BaseModel):
    settings: SystemSettings = SystemSettings()
    # 3 modelos ativos configuráveis pelo usuário
    active_models: List[ActiveModelSlot] = [
        ActiveModelSlot(
            id="slot-1",
            label="Modelo Ativo 1 (Principal)",
            provider="gemini",
            model_id="gemini-2.5-flash",
            reasoning_effort="none",
            active=True
        ),
        ActiveModelSlot(
            id="slot-2",
            label="Modelo Ativo 2 (Secundário)",
            provider="openrouter",
            model_id="google/gemini-2.0-flash-exp:free",
            reasoning_effort="none",
            active=True
        ),
        ActiveModelSlot(
            id="slot-3",
            label="Modelo Ativo 3 (Reserva/Alternativo)",
            provider="claude",
            model_id="claude-3-5-haiku-latest",
            reasoning_effort="none",
            active=True
        )
    ]
    providers: Dict[str, ProviderConfig] = {
        "gemini": ProviderConfig(
            name="gemini",
            label="Google Gemini",
            active=True,
            default_model="gemini-2.5-flash",
            available_models=[
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-1.5-flash",
                "gemini-1.5-pro"
            ],
            keys=[]
        ),
        "openrouter": ProviderConfig(
            name="openrouter",
            label="OpenRouter (Centenas de Modelos & Gratuitos)",
            active=True,
            default_model="google/gemini-2.0-flash-exp:free",
            available_models=[
                "google/gemini-2.0-flash-exp:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "deepseek/deepseek-r1:free",
                "deepseek/deepseek-chat:free",
                "anthropic/claude-3.5-haiku",
                "google/gemini-2.5-flash"
            ],
            base_url="https://openrouter.ai/api/v1",
            keys=[]
        ),
        "cloudflare": ProviderConfig(
            name="cloudflare",
            label="Cloudflare Workers AI",
            active=False,
            default_model="@cf/meta/llama-3.3-70b-instruct",
            available_models=[
                "@cf/meta/llama-3.3-70b-instruct",
                "@cf/meta/llama-3.1-8b-instruct",
                "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
                "@cf/qwen/qwen2.5-7b-instruct",
                "@cf/mistral/mistral-7b-instruct-v0.2"
            ],
            base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
            account_id="",
            keys=[]
        ),
        "claude": ProviderConfig(
            name="claude",
            label="Anthropic Claude",
            active=True,
            default_model="claude-3-5-haiku-latest",
            available_models=[
                "claude-3-5-haiku-latest",
                "claude-3-5-sonnet-latest",
                "claude-3-haiku-20240307",
                "claude-3-opus-20240229"
            ],
            keys=[]
        ),
        "groq": ProviderConfig(
            name="groq",
            label="Groq (Llama 3 / DeepSeek Gratuito/Rápido)",
            active=False,
            default_model="llama-3.3-70b-versatile",
            available_models=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "deepseek-r1-distill-llama-70b"
            ],
            base_url="https://api.groq.com/openai/v1",
            keys=[]
        ),
        "openai": ProviderConfig(
            name="openai",
            label="OpenAI / DeepSeek / Custom",
            active=False,
            default_model="gpt-4o-mini",
            available_models=["gpt-4o-mini", "gpt-4o", "deepseek-chat"],
            base_url="https://api.openai.com/v1",
            keys=[]
        )
    }

def load_config() -> AppConfig:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                config = AppConfig(**data)
                # Garantir que novos provedores existam se o config.json for antigo
                for default_p_name, default_p in AppConfig().providers.items():
                    if default_p_name not in config.providers:
                        config.providers[default_p_name] = default_p
                return config
        except Exception as e:
            print(f"Erro ao carregar {CONFIG_FILE}: {e}. Usando configuração padrão.")
    config = AppConfig()
    save_config(config)
    return config

def save_config(config: AppConfig) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2, ensure_ascii=False)
