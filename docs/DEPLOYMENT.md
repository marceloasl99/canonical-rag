# Guia de Deploy e Produção do CanonicalRAG

Este guia orienta a configuração do **CanonicalRAG** para execução contínua em servidores Linux (Ubuntu, Debian, CentOS) com gerenciamento de processo, reinício automático e proxy reverso.

---

## 1. Deploy Nativo com Systemd (Recomendado)

O uso do `systemd` nativo consome o mínimo de recursos possível (~40 MB RAM), ideal para VPS e instâncias de 1 a 2 GB de RAM.

### Passo 1: Criar o arquivo de serviço
Crie o arquivo `/etc/systemd/system/canonical-rag.service`:

```ini
[Unit]
Description=CanonicalRAG Document Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/canonical-rag
ExecStart=/opt/canonical-rag/venv/bin/python3 -m backend.main
Restart=always
RestartSec=3
Environment="PORT=80"
Environment="CONFIG_PATH=/opt/canonical-rag/config.json"
Environment="INDEX_DB_PATH=/opt/canonical-rag/documents_index.db"
Environment="MODELS_CACHE_DIR=/opt/canonical-rag/models_cache"

[Install]
WantedBy=multi-user.target
```

### Passo 2: Habilitar e iniciar
```bash
sudo systemctl daemon-reload
sudo systemctl enable canonical-rag
sudo systemctl start canonical-rag
sudo systemctl status canonical-rag
```

### Passo 3: Monitorar logs em tempo real
```bash
journalctl -u canonical-rag -f
```

---

## 2. Deploy com Nginx (Proxy Reverso & HTTPS)

Para expor a aplicação com certificado SSL gratuito (Let's Encrypt):

### Configuração do Nginx (`/etc/nginx/sites-available/canonical-rag`):
```nginx
server {
    listen 80;
    server_name seudominio.com.br;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

Para ativar SSL com Certbot:
```bash
sudo certbot --nginx -d seudominio.com.br
```

---

## 3. Deploy com Docker e Docker Compose

Se preferir isolamento em contêiner:

### `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

CMD ["python", "-m", "backend.main"]
```

### `docker-compose.yml`:
```yaml
version: '3.8'

services:
  canonical-rag:
    build: .
    container_name: canonical-rag
    ports:
      - "80:80"
    volumes:
      - ./data:/app/data
      - ./config.json:/app/config.json
      - ./documents_index.db:/app/documents_index.db
      - ./models_cache:/app/models_cache
    restart: always
```
