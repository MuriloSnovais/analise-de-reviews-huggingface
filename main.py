"""
API de Análise de Sentimento para Avaliações de Delivery.

Recebe o texto de uma avaliação e devolve a classificação em estrelas (1-5)
e o sentimento geral (positivo, neutro ou negativo), usando um modelo
BERT multilíngue do Hugging Face.
"""

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from transformers import pipeline

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MAX_TOKENS = 512  # limite do BERT: textos maiores precisam ser truncados

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("sentimento-api")

Sentimento = Literal["positivo", "neutro", "negativo"]


# ----------------------------------------------------------------------------
# Ciclo de vida da aplicação: carrega o modelo UMA vez, na inicialização
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Carregando o modelo '%s'...", MODEL_NAME)
    # Se o modelo não carregar, deixamos a exceção subir (fail fast):
    # é melhor o container falhar logo do que subir "doente".
    app.state.classifier = pipeline("text-classification", model=MODEL_NAME)
    logger.info("Modelo carregado com sucesso.")
    yield
    logger.info("Encerrando a aplicação.")
    app.state.classifier = None


app = FastAPI(
    title="API de Análise de Sentimento - Delivery",
    description="Classifica avaliações de restaurantes em estrelas e sentimento.",
    version="1.0.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------------
# Schemas (contratos de entrada e saída)
# ----------------------------------------------------------------------------
class ReviewRequest(BaseModel):
    # Remove espaços no início/fim ANTES de validar o tamanho mínimo
    model_config = ConfigDict(str_strip_whitespace=True)

    review: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Texto da avaliação feita pelo cliente.",
        examples=["A comida estava ótima, mas a entrega demorou."],
    )


class ReviewResponse(BaseModel):
    review: str
    estrelas: int = Field(..., ge=1, le=5, description="Nota prevista (1 a 5).")
    confianca: float = Field(..., ge=0, le=1, description="Probabilidade da nota prevista.")
    sentimento: Sentimento
    probabilidades: dict[str, float] = Field(
        ..., description="Probabilidade do modelo para cada nota (1 a 5)."
    )


# ----------------------------------------------------------------------------
# Regras de negócio
# ----------------------------------------------------------------------------
def estrelas_para_sentimento(estrelas: int) -> Sentimento:
    """Converte a nota (1-5) em um sentimento geral."""
    if estrelas <= 2:
        return "negativo"
    if estrelas == 3:
        return "neutro"
    return "positivo"


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/health", tags=["Infra"])
def health(request: Request):
    """Usado por orquestradores (Docker, Kubernetes) para checar se a API está viva."""
    return {
        "status": "ok",
        "modelo_carregado": request.app.state.classifier is not None,
    }


@app.post("/analisar-review", response_model=ReviewResponse, tags=["Análise"])
def analisar_review(payload: ReviewRequest, request: Request) -> ReviewResponse:
    """Analisa uma avaliação e retorna estrelas, confiança e sentimento geral."""
    classifier = request.app.state.classifier

    try:
        # top_k=None -> retorna a probabilidade de TODAS as classes (1 a 5 estrelas)
        # truncation/max_length -> evita erro com textos acima do limite do BERT
        resultado = classifier(
            payload.review,
            top_k=None,
            truncation=True,
            max_length=MAX_TOKENS,
        )
    except Exception:
        # logger.exception registra o stack trace completo NO SERVIDOR,
        # mas o cliente recebe uma mensagem genérica (sem vazar detalhes internos).
        logger.exception("Falha na inferência do modelo.")
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao processar a avaliação.",
        )

    # Dependendo da versão do transformers, o retorno pode vir aninhado ([[...]])
    if resultado and isinstance(resultado[0], list):
        resultado = resultado[0]

    # Labels do modelo vêm como "1 star", "2 stars", ... -> pegamos o número
    probs = {int(item["label"].split()[0]): item["score"] for item in resultado}
    estrelas = max(probs, key=probs.get)

    return ReviewResponse(
        review=payload.review,
        estrelas=estrelas,
        confianca=round(probs[estrelas], 4),
        sentimento=estrelas_para_sentimento(estrelas),
        probabilidades={str(k): round(v, 4) for k, v in sorted(probs.items())},
    )