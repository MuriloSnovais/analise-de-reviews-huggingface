FROM python:3.11-slim

# Evita .pyc e garante logs sem buffer (aparecem na hora no `docker logs`)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface

# Boa prática de segurança: não rodar como root
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# 1) Dependências primeiro (aproveita o cache de camadas do Docker)
COPY requirements.txt .

# Instalamos o PyTorch versão CPU ANTES: é bem menor que a versão com CUDA.
# Quando o pip processar o requirements.txt, ele verá que o torch já está instalado.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# 2) Código da aplicação
COPY --chown=appuser:appuser main.py .

USER appuser

# 3) Baixa o modelo durante o BUILD, e não na primeira execução.
#    Assim o container sobe rápido e funciona até sem internet.
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment')"

EXPOSE 8000

# 0.0.0.0 é necessário para a API ser acessível de fora do container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]