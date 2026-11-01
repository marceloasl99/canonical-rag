#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "Instalando CanonicalRAG no Ubuntu (Porta 80 - 100% Leve)"
echo "=========================================================="

APP_DIR="/opt/canonical-rag"
DOCS_DIR="/dados/documentos"

# 1. Atualizar e instalar dependências básicas do sistema
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git curl

# 2. Criar diretórios
sudo mkdir -p "$APP_DIR"
sudo mkdir -p "$DOCS_DIR"

# Copiar arquivos do projeto para /opt/canonical-rag se executado da pasta clonada
if [ -f "requirements.txt" ]; then
    echo "Copiando arquivos da aplicação para $APP_DIR..."
    sudo cp -r ./* "$APP_DIR/"
fi

cd "$APP_DIR"

# 3. Criar ambiente virtual Python
if [ ! -d "venv" ]; then
    echo "Criando ambiente virtual Python..."
    sudo python3 -m venv venv
fi

# 4. Instalar bibliotecas
echo "Instalando bibliotecas Python leves..."
sudo ./venv/bin/pip install --upgrade pip
sudo ./venv/bin/pip install -r requirements.txt

# 5. Criar serviço systemd para rodar na porta 80 automaticamente no boot
echo "Criando serviço de inicialização automática (systemd)..."
sudo bash -c "cat <<EOF > /etc/systemd/system/canonical-rag.service
[Unit]
Description=CanonicalRAG Document Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/python3 -m backend.main
Restart=always
RestartSec=5
Environment=PORT=80
Environment=CONFIG_PATH=$APP_DIR/config.json
Environment=INDEX_DB_PATH=$APP_DIR/documents_index.db

[Install]
WantedBy=multi-user.target
EOF"

# 6. Recarregar e iniciar o serviço
sudo systemctl daemon-reload
sudo systemctl enable canonical-rag
sudo systemctl restart canonical-rag

echo "=========================================================="
echo "Tudo pronto! O CanonicalRAG já está rodando na porta 80."
echo "Acesse no seu navegador: http://$(hostname -I | awk '{print $1}')"
echo "Coloque seus arquivos e PDFs na pasta: $DOCS_DIR"
echo "=========================================================="
