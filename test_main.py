"""
Testes da API de Análise de Sentimento.

Usamos um modelo FALSO (FakeClassifier) para os testes rodarem em segundos,
sem baixar nem carregar o BERT. Aqui testamos a NOSSA lógica (validação,
formato da resposta, regras de negócio), não a qualidade do modelo.
"""

import pytest
from fastapi.testclient import TestClient

import main

URL_UNICO = "/analisar-review"
URL_LOTE = "/analisar-reviews"


# ----------------------------------------------------------------------------
# Modelo falso: imita o formato de saída do pipeline do Hugging Face
# ----------------------------------------------------------------------------
class FakeClassifier:
    """Devolve estrelas previsíveis com base em palavras-chave do texto."""

    def _estrelas(self, texto: str) -> int:
        texto = texto.lower()
        if any(p in texto for p in ("adorei", "ótimo", "excelente")):
            return 5
        if any(p in texto for p in ("péssimo", "horrível", "frio")):
            return 1
        return 3

    def _resultado(self, texto: str) -> list[dict]:
        vencedora = self._estrelas(texto)
        resultado = []
        for n in range(1, 6):
            label = "1 star" if n == 1 else f"{n} stars"  # igual ao modelo real
            resultado.append({"label": label, "score": 0.8 if n == vencedora else 0.05})
        return resultado

    def __call__(self, entrada, **kwargs):
        # Texto único -> lista de dicts | Lista de textos -> lista de listas
        if isinstance(entrada, list):
            return [self._resultado(t) for t in entrada]
        return self._resultado(entrada)


class ModeloQuebrado:
    """Simula uma falha na inferência."""

    def __call__(self, *args, **kwargs):
        raise RuntimeError("detalhe interno secreto")


# ----------------------------------------------------------------------------
# Fixture: cliente de teste com o modelo falso no lugar do real
# ----------------------------------------------------------------------------
@pytest.fixture
def client(monkeypatch):
    # Troca o `pipeline` usado em main.py pelo falso ANTES do startup
    monkeypatch.setattr(main, "pipeline", lambda *args, **kwargs: FakeClassifier())
    # O `with` dispara o lifespan (startup/shutdown) da aplicação
    with TestClient(main.app) as c:
        yield c


# ----------------------------------------------------------------------------
# Regra de negócio (teste unitário puro)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "estrelas, esperado",
    [(1, "negativo"), (2, "negativo"), (3, "neutro"), (4, "positivo"), (5, "positivo")],
)
def test_estrelas_para_sentimento(estrelas, esperado):
    assert main.estrelas_para_sentimento(estrelas) == esperado


# ----------------------------------------------------------------------------
# /health
# ----------------------------------------------------------------------------
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "modelo_carregado": True}


# ----------------------------------------------------------------------------
# /analisar-review (uma avaliação)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "texto, estrelas, sentimento",
    [
        ("Adorei! Chegou rápido e estava delicioso.", 5, "positivo"),
        ("Pedido chegou frio e o entregador foi grosseiro.", 1, "negativo"),
        ("Comida normal, nada de especial.", 3, "neutro"),
    ],
)
def test_analisar_review_sentimentos(client, texto, estrelas, sentimento):
    resp = client.post(URL_UNICO, json={"review": texto})
    assert resp.status_code == 200

    corpo = resp.json()
    assert corpo["review"] == texto
    assert corpo["estrelas"] == estrelas
    assert corpo["sentimento"] == sentimento
    assert corpo["confianca"] == 0.8
    assert set(corpo["probabilidades"]) == {"1", "2", "3", "4", "5"}


def test_analisar_review_remove_espacos_das_pontas(client):
    resp = client.post(URL_UNICO, json={"review": "   Adorei a comida!   "})
    assert resp.status_code == 200
    assert resp.json()["review"] == "Adorei a comida!"


@pytest.mark.parametrize(
    "payload",
    [
        {"review": ""},            # vazio
        {"review": "     "},       # só espaços
        {"review": "ok"},          # menor que o mínimo (3)
        {"review": "a" * 2001},    # maior que o máximo (2000)
        {},                        # campo ausente
        {"review": 123},           # tipo errado
    ],
)
def test_analisar_review_entradas_invalidas(client, payload):
    resp = client.post(URL_UNICO, json=payload)
    assert resp.status_code == 422


def test_analisar_review_erro_interno_nao_vaza_detalhes(client):
    client.app.state.classifier = ModeloQuebrado()

    resp = client.post(URL_UNICO, json={"review": "Adorei a comida!"})

    assert resp.status_code == 500
    assert "detalhe interno secreto" not in resp.text  # segurança!


# ----------------------------------------------------------------------------
# /analisar-reviews (lote)
# ----------------------------------------------------------------------------
def test_lote_mantem_ordem_e_calcula_resumo(client):
    textos = [
        "Adorei! Estava delicioso.",        # 5 -> positivo
        "Pedido chegou frio.",              # 1 -> negativo
        "Comida normal, nada de especial.", # 3 -> neutro
        "Excelente atendimento!",           # 5 -> positivo
    ]
    resp = client.post(URL_LOTE, json={"reviews": textos})
    assert resp.status_code == 200
    corpo = resp.json()

    # A ordem dos resultados é a mesma da lista enviada
    assert [r["review"] for r in corpo["resultados"]] == textos
    assert [r["estrelas"] for r in corpo["resultados"]] == [5, 1, 3, 5]

    assert corpo["resumo"] == {
        "total": 4,
        "positivos": 2,
        "neutros": 1,
        "negativos": 1,
        "media_estrelas": 3.5,  # (5 + 1 + 3 + 5) / 4
    }


def test_lote_com_um_unico_item(client):
    resp = client.post(URL_LOTE, json={"reviews": ["Adorei a comida!"]})
    assert resp.status_code == 200
    assert resp.json()["resumo"]["total"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"reviews": []},                                          # lista vazia
        {"reviews": ["Adorei a comida!", ""]},                    # um item vazio
        {"reviews": ["Adorei a comida!", "   "]},                 # um item só de espaços
        {"reviews": ["texto válido"] * (main.MAX_REVIEWS_POR_LOTE + 1)},  # acima do limite
        {"reviews": "isto não é uma lista"},                      # tipo errado
        {},                                                       # campo ausente
    ],
)
def test_lote_entradas_invalidas(client, payload):
    resp = client.post(URL_LOTE, json=payload)
    assert resp.status_code == 422


def test_lote_no_limite_maximo_e_aceito(client):
    textos = ["Adorei a comida!"] * main.MAX_REVIEWS_POR_LOTE
    resp = client.post(URL_LOTE, json={"reviews": textos})
    assert resp.status_code == 200
    assert resp.json()["resumo"]["total"] == main.MAX_REVIEWS_POR_LOTE


def test_lote_erro_interno(client):
    client.app.state.classifier = ModeloQuebrado()

    resp = client.post(URL_LOTE, json={"reviews": ["Adorei a comida!"]})

    assert resp.status_code == 500
    assert "detalhe interno secreto" not in resp.text