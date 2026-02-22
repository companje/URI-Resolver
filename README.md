# resolvable-uris

Resolver-service voor RDF resources onder `/id/...` met `pyoxigraph` als opslag.

## Doel

Deze service maakt URI's resolvebaar binnen een namespace, met:
- default output als JSON-LD;
- Turtle via content negotiation of extensie (`.ttl`);
- consistente foutresponses in hetzelfde opgevraagde formaat;
- eenvoudige HTTP-API voor database, graph en resource-niveau.

## URI-model

Alle resolvable URI's zitten onder `/id`:
- `/id/<db>`: database-resource (metadata in default graph)
- `/id/<db>/<graph>`: named graph
- `/id/<db>/<graph>/<resource>`: resource binnen die graph

Trailing slashes worden genegeerd:
- `/id/mijndb/` en `/id/mijndb` zijn equivalent
- idem voor graph/resource paden

## API

### `GET /id/<db>`
Leest triples over de database uit de default graph.
Bevat o.a. links naar bestaande graphs via predicate:
`https://kvan-todb.hualab.nl/def/hasGraph`.

### `POST /id/<db>`
Maakt database aan als die nog niet bestaat en voegt (minimaal) een label toe.

Body (optioneel):
```json
{
  "label": "My Database"
}
```

### `GET /id/<db>/<graph>`
Leest alle triples uit de named graph.

### `POST /id/<db>/<graph>`
Maakt graph aan als die nog niet bestaat en voegt (minimaal) een label toe.
Als database nog niet bestaat, wordt die ook automatisch aangemaakt.

Body (optioneel):
```json
{
  "label": "My Graph"
}
```

### `GET /id/<db>/<graph>/<resource>`
Leest triples van het subject `<resource>` binnen de opgegeven graph.

### `POST /id/<db>/<graph>/<resource>`
Voegt een triple toe voor die resource.

Als db en/of graph nog niet bestaan, worden ze automatisch aangemaakt.

Body (verplicht):
```json
{
  "p": "http://www.w3.org/2000/01/rdf-schema#label",
  "o": "hoi"
}
```

Ondersteunde `o`-vormen:
- string: plain literal
- object literal:
```json
{
  "type": "literal",
  "value": "hallo",
  "lang": "nl"
}
```
- object IRI:
```json
{
  "type": "iri",
  "value": "https://example.org/page"
}
```

## Content Negotiation

Representatiekeuze:
- default: `application/ld+json`
- `Accept: text/turtle` => Turtle
- `.ttl` extensie => Turtle
- `.json` extensie => JSON(-LD)

Dit geldt ook voor foutmeldingen.

Voorbeelden:
- `GET /id/mijndb/mijngraph/mijnresource` -> JSON-LD (default)
- `GET /id/mijndb/mijngraph/mijnresource` + `Accept: text/turtle` -> Turtle
- `GET /id/mijndb/mijngraph/mijnresource.ttl` -> Turtle
- `GET /id/mijndb/mijngraph/mijnresource.json` -> JSON

## Foutresponses

Fouten worden teruggegeven als:
- JSON (default / `.json`)
- Turtle (`Accept: text/turtle` of `.ttl`)

JSON-fouten gebruiken dit schema:
```json
{
  "error": {
    "code": "...",
    "message": "...",
    "details": {}
  }
}
```

## JSON formatting

Alle JSON-responses (inclusief fouten en JSON-LD) worden geformatteerd met `indent=2`.

## Testmodus: schrijven via GET

Voor test/doelmatig gebruik kun je writes via `GET` toestaan op basis van query parameters.

Activeren:
```bash
TEST_MODE_GET_WRITE=true uv run uvicorn app.main:app --reload
```

Gedrag in deze modus:
- Query params worden vertaald naar triples.
- Elke key/value wordt een triple met object als literal string.
- Zonder prefix wordt predicate: `https://kvan-todb.hualab.nl/def/<key>`.
- Met prefix (`prefix:local`) wordt predicate opgelost via prefix map.
- Als db (of graph/resource context) nog niet bestaat, wordt die bij deze write flow aangemaakt.

Voorbeelden:
```text
GET /id/mijndb?x=5
```
maakt (in default graph) een triple:
- subject: `https://kvan-todb.hualab.nl/id/mijndb`
- predicate: `https://kvan-todb.hualab.nl/def/x`
- object: `"5"`

```text
GET /id/mijndb/mijngraph/mijnresource?rdfs:label=hoi
```
maakt een triple in named graph `.../id/mijndb/mijngraph` met:
- predicate: `http://www.w3.org/2000/01/rdf-schema#label`
- object: `"hoi"`

Ondersteunde prefixes (kleine standaardlijst):
- `rdf`
- `rdfs`
- `xsd`
- `foaf`
- `skos`
- `sdo`

## Runnen met uv

```bash
uv venv
source .venv/bin/activate
uv sync
uv run uvicorn app.main:app --reload
```

Of kort:
```bash
uv run uvicorn app.main:app --reload
```

## Caddy lokaal

`caddy/Caddyfile` verwacht host `kvan-todb.hualab.nl` en proxyt naar `127.0.0.1:8000`.

1. Zorg voor hosts-entry:
```bash
echo "127.0.0.1 kvan-todb.hualab.nl" | sudo tee -a /etc/hosts
```

2. Start Caddy:
```bash
caddy run --config caddy/Caddyfile
```

3. Vertrouw lokale CA:
```bash
sudo caddy trust
```

Daarna kun je testen op:
- `https://kvan-todb.hualab.nl/id/...`

## Snelle smoke test

Triple aanmaken (maakt db/graph aan indien nodig):
```bash
curl -i -X POST 'https://kvan-todb.hualab.nl/id/mijndb/mijngraph/mijnresource' \
  -H 'Content-Type: application/json' \
  -d '{"p":"http://www.w3.org/2000/01/rdf-schema#label","o":"hoi"}'
```

Resource als Turtle:
```bash
curl -i 'https://kvan-todb.hualab.nl/id/mijndb/mijngraph/mijnresource.ttl'
```

Niet-bestaande graph als Turtle-fout:
```bash
curl -i 'https://kvan-todb.hualab.nl/id/mijndb/bestaatniet.ttl'
```
