"""
API de Análise de Sentimento para Avaliações de Delivery.

Endpoints:
- POST /analisar-review   -> analisa UMA avaliação
- POST /analisar-reviews  -> analisa VÁRIAS avaliações de uma vez (lote)
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from transformers import pipeline

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MAX_TOKENS = 512            # limite do BERT
MAX_REVIEWS_POR_LOTE = 50   # protege a API contra requisições gigantes
TAMANHO_BATCH_MODELO = 8    # quantos textos o modelo processa juntos por vez

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("sentimento-api")

Sentimento = Literal["positivo", "neutro", "negativo"]

# Tipo reutilizável: string limpa (sem espaços nas pontas) e com tamanho válido
TextoReview = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=2000),
]


# ----------------------------------------------------------------------------
# Ciclo de vida: carrega o modelo UMA vez
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Carregando o modelo '%s'...", MODEL_NAME)
    app.state.classifier = pipeline("text-classification", model=MODEL_NAME)
    logger.info("Modelo carregado com sucesso.")
    yield
    logger.info("Encerrando a aplicação.")
    app.state.classifier = None


app = FastAPI(
    title="API de Análise de Sentimento - Delivery",
    description="Classifica avaliações de restaurantes em estrelas e sentimento.",
    version="1.1.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------
class ReviewRequest(BaseModel):
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


class BatchRequest(BaseModel):
    reviews: list[TextoReview] = Field(
        ...,
        min_length=1,
        max_length=MAX_REVIEWS_POR_LOTE,
        description=f"Lista de avaliações (de 1 a {MAX_REVIEWS_POR_LOTE}).",
        examples=[["Pedido chegou frio.", "Adorei, estava delicioso!"]],
    )


class ResumoLote(BaseModel):
    total: int
    positivos: int
    neutros: int
    negativos: int
    media_estrelas: float


class BatchResponse(BaseModel):
    resumo: ResumoLote
    resultados: list[ReviewResponse]  # mesma ordem da lista enviada


# ----------------------------------------------------------------------------
# Regras de negócio e funções auxiliares
# ----------------------------------------------------------------------------
def estrelas_para_sentimento(estrelas: int) -> Sentimento:
    """Converte a nota (1-5) em um sentimento geral."""
    if estrelas <= 2:
        return "negativo"
    if estrelas == 3:
        return "neutro"
    return "positivo"


def construir_resposta(review: str, saida_modelo: list) -> ReviewResponse:
    """Transforma a saída bruta do modelo para UM texto em ReviewResponse."""
    # Dependendo da versão do transformers, a saída pode vir aninhada ([[...]])
    if saida_modelo and isinstance(saida_modelo[0], list):
        saida_modelo = saida_modelo[0]

    # Labels vêm como "1 star", "2 stars", ... -> pegamos o número
    probs = {int(item["label"].split()[0]): item["score"] for item in saida_modelo}
    estrelas = max(probs, key=probs.get)

    return ReviewResponse(
        review=review,
        estrelas=estrelas,
        confianca=round(probs[estrelas], 4),
        sentimento=estrelas_para_sentimento(estrelas),
        probabilidades={str(k): round(v, 4) for k, v in sorted(probs.items())},
    )


def executar_modelo(classifier, entrada):
    """Chama o pipeline com tratamento de erro centralizado."""
    try:
        return classifier(
            entrada,
            top_k=None,                    # probabilidade de todas as classes
            truncation=True,               # corta textos acima do limite
            max_length=MAX_TOKENS,
            batch_size=TAMANHO_BATCH_MODELO,
        )
    except Exception:
        # Stack trace vai pro log do servidor; o cliente recebe mensagem genérica.
        logger.exception("Falha na inferência do modelo.")
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao processar a(s) avaliação(ões).",
        )


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/health", tags=["Infra"])
def health(request: Request):
    """Checa se a API está viva e se o modelo foi carregado."""
    return {
        "status": "ok",
        "modelo_carregado": request.app.state.classifier is not None,
    }


@app.post("/analisar-review", response_model=ReviewResponse, tags=["Análise"])
def analisar_review(payload: ReviewRequest, request: Request) -> ReviewResponse:
    """Analisa UMA avaliação."""
    saida = executar_modelo(request.app.state.classifier, payload.review)
    return construir_resposta(payload.review, saida)


@app.post("/analisar-reviews", response_model=BatchResponse, tags=["Análise"])
def analisar_reviews(payload: BatchRequest, request: Request) -> BatchResponse:
    """Analisa VÁRIAS avaliações em uma única requisição (lote)."""
    # Passando uma LISTA, o pipeline processa os textos em batches internos
    saidas = executar_modelo(request.app.state.classifier, payload.reviews)

    if len(saidas) != len(payload.reviews):
        logger.error("Modelo retornou %d resultados para %d textos.",
                     len(saidas), len(payload.reviews))
        raise HTTPException(status_code=500, detail="Erro interno ao processar o lote.")

    resultados = [
        construir_resposta(texto, saida)
        for texto, saida in zip(payload.reviews, saidas)
    ]

    total = len(resultados)
    resumo = ResumoLote(
        total=total,
        positivos=sum(r.sentimento == "positivo" for r in resultados),
        neutros=sum(r.sentimento == "neutro" for r in resultados),
        negativos=sum(r.sentimento == "negativo" for r in resultados),
        media_estrelas=round(sum(r.estrelas for r in resultados) / total, 2),
    )

    return BatchResponse(resumo=resumo, resultados=resultados)