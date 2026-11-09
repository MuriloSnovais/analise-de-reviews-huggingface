import json
import urllib.request

URL = "http://localhost:8000/analisar-reviews"

with open("reviews.json", encoding="utf-8") as f:
    textos = [item["review"] for item in json.load(f)]

req = urllib.request.Request(
    URL,
    data=json.dumps({"reviews": textos}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    dados = json.load(resp)

print("RESUMO:", dados["resumo"], "\n")
for r in dados["resultados"]:
    print(f'{r["estrelas"]}★ | {r["sentimento"]:<8} | {r["confianca"]:.2f} | {r["review"]}')