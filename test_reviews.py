import json
import urllib.request

URL = "http://localhost:8000/analisar-review"

with open("reviews.json", encoding="utf-8") as f:
    reviews = json.load(f)

for item in reviews:
    req = urllib.request.Request(
        URL,
        data=json.dumps(item).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        r = json.load(resp)
    print(f'{r["estrelas"]}★ | {r["sentimento"]:<8} | {r["review"]}')