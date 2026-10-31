import httpx
import logging
import asyncio
from typing import Dict, Any, Optional, Tuple, List
from .config import AppConfig, ProviderConfig, APIKeyEntry, ActiveModelSlot, save_config

logger = logging.getLogger("LLMGateway")

class LLMGateway:
    def __init__(self, config: AppConfig):
        self.config = config
        self._provider_indices: Dict[str, int] = {}
        self._active_model_index: int = 0

    def update_config(self, config: AppConfig):
        self.config = config

    def _get_key_for_slot(self, slot: ActiveModelSlot) -> Optional[Tuple[APIKeyEntry, ProviderConfig]]:
        provider = self.config.providers.get(slot.provider)
        if not provider or not provider.active:
            return None

        # Se o slot tem chave específica definida
        if slot.api_key_id:
            for k in provider.keys:
                if k.id == slot.api_key_id and k.active and k.key.strip():
                    return k, provider

        # Caso contrário, pega a próxima chave ativa do provedor (Round-Robin)
        active_keys = [k for k in provider.keys if k.active and k.key.strip()]
        if not active_keys:
            return None

        idx = self._provider_indices.get(provider.name, 0)
        selected_key = active_keys[idx % len(active_keys)]
        if self.config.settings.round_robin_enabled:
            self._provider_indices[provider.name] = (idx + 1) % len(active_keys)

        return selected_key, provider

    async def fetch_available_models(self, provider_name: str, api_key: str) -> List[str]:
        """Consulta dinamicamente na API do provedor quais modelos existem para aquela chave."""
        provider = self.config.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Provedor {provider_name} desconhecido")

        models: List[str] = []

        async with httpx.AsyncClient(timeout=20.0) as client:
            if provider_name == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
                resp = await client.get(url)
                if resp.status_code != 200:
                    raise RuntimeError(f"Erro Gemini ({resp.status_code}): {resp.text}")
                data = resp.json()
                for m in data.get("models", []):
                    methods = m.get("supportedGenerationMethods", [])
                    if "generateContent" in methods:
                        m_name = m.get("name", "").replace("models/", "")
                        if m_name and not m_name.startswith("text-embedding") and not m_name.startswith("aqa"):
                            models.append(m_name)

            elif provider_name == "openrouter":
                url = "https://openrouter.ai/api/v1/models"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "http://10.0.0.151",
                    "X-Title": "CanonicalRAG"
                }
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    raise RuntimeError(f"Erro OpenRouter ({resp.status_code}): {resp.text}")
                data = resp.json()
                for m in data.get("data", []):
                    m_id = m.get("id")
                    if m_id:
                        models.append(m_id)
                # Ordenar modelos gratuitos (:free) no topo
                models = sorted(models, key=lambda x: (not ":free" in x, not "flash" in x, x))

            elif provider_name == "cloudflare":
                account_id = provider.account_id.strip()
                if account_id:
                    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search?task=Text%20Generation"
                    headers = {"Authorization": f"Bearer {api_key}"}
                    try:
                        resp = await client.get(url, headers=headers)
                        if resp.status_code == 200:
                            data = resp.json()
                            for m in data.get("result", []):
                                m_name = m.get("name")
                                if m_name:
                                    models.append(m_name)
                    except Exception as e:
                        logger.warning(f"Erro ao buscar modelos Cloudflare: {e}")

                if not models:
                    models = [
                        "@cf/meta/llama-3.3-70b-instruct",
                        "@cf/meta/llama-3.1-8b-instruct",
                        "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
                        "@cf/qwen/qwen2.5-7b-instruct",
                        "@cf/mistral/mistral-7b-instruct-v0.2",
                        "@cf/google/gemma-7b-it"
                    ]

            elif provider_name == "claude":
                url = "https://api.anthropic.com/v1/models"
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                }
                try:
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        for m in data.get("data", []):
                            m_id = m.get("id")
                            if m_id:
                                models.append(m_id)
                except Exception:
                    pass

                if not models:
                    models = [
                        "claude-3-5-haiku-latest",
                        "claude-3-5-sonnet-latest",
                        "claude-3-haiku-20240307",
                        "claude-3-sonnet-20240229",
                        "claude-3-opus-20240229"
                    ]

            elif provider_name in ["groq", "openai"]:
                base_url = provider.base_url.rstrip("/")
                url = f"{base_url}/models"
                headers = {"Authorization": f"Bearer {api_key}"}
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    raise RuntimeError(f"Erro {provider.label} ({resp.status_code}): {resp.text}")
                data = resp.json()
                for m in data.get("data", []):
                    m_id = m.get("id")
                    if m_id:
                        models.append(m_id)

        if models:
            provider.available_models = models
            save_config(self.config)

        return models

    async def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        preferred_slot_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executa a geração usando os Modelos Ativos configurados.
        """
        active_slots = [s for s in self.config.active_models if s.active]
        if not active_slots:
            raise ValueError("Nenhum modelo ativo configurado no sistema. Ative ao menos um modelo na aba de configurações.")

        slots_to_try = []
        if preferred_slot_id:
            for s in active_slots:
                if s.id == preferred_slot_id:
                    slots_to_try.append(s)
                    break

        if self.config.settings.round_robin_enabled and not preferred_slot_id:
            idx = self._active_model_index % len(active_slots)
            slots_to_try = active_slots[idx:] + active_slots[:idx]
            self._active_model_index = (idx + 1) % len(active_slots)
        else:
            for s in active_slots:
                if s not in slots_to_try:
                    slots_to_try.append(s)

        last_error = None
        for slot in slots_to_try:
            key_tuple = self._get_key_for_slot(slot)
            if not key_tuple:
                logger.warning(f"Slot {slot.label} ({slot.model_id}) não possui chave ativa válida.")
                continue

            key_entry, provider = key_tuple
            logger.info(f"Tentando Slot: {slot.label} | Provedor: {slot.provider} | Modelo: {slot.model_id}")

            try:
                if slot.provider == "gemini":
                    text = await self._call_gemini(
                        api_key=key_entry.key,
                        model=slot.model_id,
                        reasoning_effort=slot.reasoning_effort,
                        temperature=slot.temperature,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt
                    )
                elif slot.provider == "claude":
                    text = await self._call_claude(
                        api_key=key_entry.key,
                        model=slot.model_id,
                        temperature=slot.temperature,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt
                    )
                elif slot.provider == "openrouter":
                    text = await self._call_openrouter(
                        api_key=key_entry.key,
                        model=slot.model_id,
                        reasoning_effort=slot.reasoning_effort,
                        temperature=slot.temperature,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt
                    )
                elif slot.provider == "cloudflare":
                    text = await self._call_cloudflare(
                        provider=provider,
                        api_key=key_entry.key,
                        model=slot.model_id,
                        temperature=slot.temperature,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt
                    )
                elif slot.provider in ["groq", "openai"]:
                    text = await self._call_openai_compatible(
                        provider=provider,
                        api_key=key_entry.key,
                        model=slot.model_id,
                        reasoning_effort=slot.reasoning_effort,
                        temperature=slot.temperature,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt
                    )
                else:
                    raise ValueError(f"Provedor {slot.provider} não suportado")

                slot.usage_count += 1
                key_entry.usage_count += 1
                save_config(self.config)

                return {
                    "text": text,
                    "slot_label": slot.label,
                    "slot_id": slot.id,
                    "provider": slot.provider,
                    "model": slot.model_id,
                    "reasoning_effort": slot.reasoning_effort,
                    "key_used": key_entry.label or key_entry.id
                }

            except Exception as e:
                logger.warning(f"Falha no slot {slot.label} ({slot.model_id}): {e}")
                slot.errors_count += 1
                key_entry.errors_count += 1
                save_config(self.config)
                last_error = e

                if not self.config.settings.fallback_enabled:
                    break
                continue

        raise RuntimeError(f"Todos os modelos ativos configurados falharam. Último erro: {last_error}")

    async def _call_gemini(
        self,
        api_key: str,
        model: str,
        reasoning_effort: str,
        temperature: float,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        gen_config: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": self.config.settings.max_tokens,
        }

        if reasoning_effort == "none":
            gen_config["thinkingConfig"] = {"thinkingBudget": 0}
        elif reasoning_effort == "low":
            gen_config["thinkingConfig"] = {"thinkingBudget": 1024}
        elif reasoning_effort == "medium":
            gen_config["thinkingConfig"] = {"thinkingBudget": 2048}
        elif reasoning_effort == "high":
            gen_config["thinkingConfig"] = {"thinkingBudget": 4096}

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                }
            ],
            "generationConfig": gen_config
        }
        
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 400 and "thinking" in resp.text.lower():
                payload["generationConfig"].pop("thinkingConfig", None)
                resp = await client.post(url, json=payload)

            if resp.status_code != 200:
                raise RuntimeError(f"Gemini API Error {resp.status_code}: {resp.text}")
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"Resposta vazia da API do Gemini: {data}")
            parts = candidates[0].get("content", {}).get("parts", [])
            # Extrai apenas os blocos de resposta final, ignorando pensamentos se houver
            text_parts = [p["text"] for p in parts if "text" in p and not p.get("thought", False)]
            if not text_parts:
                text_parts = [p["text"] for p in parts if "text" in p]
            return "\n".join(text_parts) if text_parts else str(parts)

    async def _call_openrouter(
        self,
        api_key: str,
        model: str,
        reasoning_effort: str,
        temperature: float,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://10.0.0.151",
            "X-Title": "CanonicalRAG"
        }
        payload: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": self.config.settings.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        if reasoning_effort != "none":
            payload["reasoning_effort"] = reasoning_effort

        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter API Error {resp.status_code}: {resp.text}")
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"Resposta vazia da OpenRouter: {data}")
            return choices[0]["message"]["content"]

    async def _call_cloudflare(
        self,
        provider: ProviderConfig,
        api_key: str,
        model: str,
        temperature: float,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        account_id = provider.account_id.strip()
        if not account_id:
            raise ValueError("Cloudflare Workers AI requer o 'Account ID' configurado.")

        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": self.config.settings.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Cloudflare Workers AI Error {resp.status_code}: {resp.text}")
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"Resposta vazia da Cloudflare: {data}")
            return choices[0]["message"]["content"]

    async def _call_claude(
        self,
        api_key: str,
        model: str,
        temperature: float,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": model,
            "max_tokens": self.config.settings.max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }
        
        # Modelos mais recentes (como claude-haiku-5-5) descontinuaram o parâmetro temperature
        if "5-" not in model and "-5" not in model:
            payload["temperature"] = temperature

        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            # Se a Anthropic reclamar de temperature, retenta sem o campo
            if resp.status_code == 400 and "temperature" in resp.text.lower():
                payload.pop("temperature", None)
                resp = await client.post(url, headers=headers, json=payload)

            if resp.status_code != 200:
                raise RuntimeError(f"Claude API Error {resp.status_code}: {resp.text}")
            data = resp.json()
            content = data.get("content", [])
            if not content:
                raise RuntimeError(f"Resposta vazia da Claude API: {data}")

            # Itera sobre os blocos de conteúdo para pegar os blocos de texto (suporta thinking blocks)
            text_blocks = [block["text"] for block in content if block.get("type") == "text" and "text" in block]
            if text_blocks:
                return "\n".join(text_blocks)

            for block in content:
                if "text" in block:
                    return block["text"]

            raise RuntimeError(f"Nenhum bloco de texto retornado pela Claude API: {data}")

    async def _call_openai_compatible(
        self,
        provider: ProviderConfig,
        api_key: str,
        model: str,
        reasoning_effort: str,
        temperature: float,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        base_url = provider.base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": self.config.settings.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        if reasoning_effort != "none":
            payload["reasoning_effort"] = reasoning_effort

        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 400 and "reasoning" in resp.text.lower():
                payload.pop("reasoning_effort", None)
                resp = await client.post(url, headers=headers, json=payload)

            if resp.status_code != 200:
                raise RuntimeError(f"{provider.label} API Error {resp.status_code}: {resp.text}")
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"Resposta vazia da API: {data}")
            return choices[0]["message"]["content"]
