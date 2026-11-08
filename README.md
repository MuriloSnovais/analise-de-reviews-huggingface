# 🍔 API de Análise de Sentimento para Avaliações de Delivery

API REST que classifica automaticamente avaliações de restaurantes e pedidos de
delivery, retornando a nota prevista (1 a 5 estrelas) e o sentimento geral
(positivo, neutro ou negativo).

Construída com **Python, FastAPI e Hugging Face Transformers**, totalmente
conteinerizada com **Docker**.

---

## 🎯 Problema de negócio

Plataformas de delivery recebem milhares de avaliações em texto livre todos os dias.
Lê-las manualmente é inviável, e muitas vezes a nota em estrelas não reflete o que
o cliente escreveu (ex.: nota 5 com um comentário reclamando da entrega).

**Esta API permite a triagem automática das avaliações para:**

- 🚨 **Priorizar** avaliações negativas para o time de atendimento e parcerias;
- 📊 **Monitorar** a satisfação por restaurante, região ou período;
- 🔍 **Detectar** inconsistências entre a nota dada e o texto escrito;
- ⚡ **Reduzir** o tempo de resposta a clientes insatisfeitos.

---

## 🧠 Como funciona

```
Cliente ──POST /analisar-review──▶ FastAPI ──▶ Pipeline Hugging Face ──▶ Resposta JSON
                                (validação      (BERT multilíngue
                                 Pydantic)       nlptown)
```

- **Modelo:** [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment),
  um BERT multilíngue (inclui português) que prevê de 1 a 5 estrelas.
- **Regra de sentimento:**

| Estrelas | Sentimento |
|:--------:|:----------:|
| 1–2      | negativo   |
| 3        | neutro     |
| 4–5      | positivo   |

---

## 🚀 Como rodar

### Pré-requisito
- [Docker](https://docs.docker.com/get-docker/) instalado.

### 1. Construir a imagem
```bash
docker build -t sentimento-delivery .
```
> ⏳ O primeiro build demora (baixa o PyTorch e o modelo). Os próximos são rápidos.

### 2. Executar o container
```bash
docker run --rm -p 8000:8000 sentimento-delivery
```

### 3. Testar

**Documentação interativa (Swagger):** http://localhost:8000/docs

**Via terminal:**
```bash
curl -X POST http://localhost:8000/analisar-review \
  -H "Content-Type: application/json" \
  -d '{"review": "A comida estava ótima, mas a entrega demorou."}'
```

**Exemplo de resposta:**
```json
{
  "review": "A comida estava ótima, mas a entrega demorou.",
  "estrelas": 4,
  "confianca": 0.4821,
  "sentimento": "positivo",
  "probabilidades": {"1": 0.0312, "2": 0.0954, "3": 0.2417, "4": 0.4821, "5": 0.1496}
}
```
> Os valores acima são ilustrativos; o resultado real pode variar.

---

## 📡 Endpoints

| Método | Rota                | Descrição                               |
|--------|---------------------|-----------------------------------------|
| POST   | `/analisar-review`  | Analisa o sentimento de uma avaliação   |
| GET    | `/health`           | Verifica se a API e o modelo estão ok   |

**Códigos de resposta:** `200` sucesso · `422` entrada inválida · `500` erro interno.

---

## 💻 Rodando sem Docker (desenvolvimento)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## ⚠️ Limitações conhecidas

- O modelo classifica a avaliação **como um todo**: textos com sentimentos mistos
  (comida boa, entrega ruim) tendem a cair em notas intermediárias.
- Foi treinado em avaliações de produtos, não especificamente de delivery brasileiro;
  gírias e ironia podem reduzir a precisão.
- Inferência em CPU: adequada para demonstração, não para alto volume.

## 🔭 Próximos passos

- [ ] Análise de sentimento **por aspecto** (comida, entrega, atendimento)
- [ ] Endpoint de **processamento em lote**
- [ ] **Fine-tuning** com avaliações em português brasileiro
- [ ] Testes automatizados (`pytest`) e CI com GitHub Actions
- [ ] Cache de resultados e métricas (Prometheus)

## 🛠️ Stack

Python 3.11 · FastAPI · Pydantic · Hugging Face Transformers · PyTorch · Docker