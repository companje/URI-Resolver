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
- `/id`: systeemresource met overzicht van databases
- `/id/<db>`: database-resource (metadata in default graph)
- `/id/<db>/<graph>`: named graph
- `/id/<db>/<graph>/<resource>`: resource binnen die graph

Vocabulary/definitie URI's zitten onder `/def`:
- `/def`: overzicht van definities
- `/def/<term>`: definitie-resource (bijv. `/def/Database`)

Trailing slashes worden genegeerd:
- `/id/mijndb/` en `/id/mijndb` zijn equivalent
- idem voor graph/resource paden

## API

### `GET /id/<db>`
Leest triples over de database uit de default graph.
Bevat o.a. links naar bestaande graphs via predicate:
`https://schema.org/hasPart` (`sdo:hasPart`).

### `GET /id`
Leest systeemmetadata en bevat links naar databases via predicate:
`https://schema.org/hasPart` (`sdo:hasPart`).
Deze system-resource wordt on-the-fly opgebouwd en bevat ook:
`rdf:type https://kvan-todb.hualab.nl/def/System`.

### `POST /id`
Voegt een triple toe aan de systeemresource (`https://kvan-todb.hualab.nl/id`).

Body:
```json
{
  "p": "http://www.w3.org/2000/01/rdf-schema#label",
  "o": "Nieuwe label"
}
```

### `DELETE /id`
Verwijdert triples op de systeemresource.

Body:
```json
{
  "p": "http://www.w3.org/2000/01/rdf-schema#label",
  "o": "Resolvable URI System"
}
```

Gedrag:
- met `p` + `o`: verwijder exact matchende triple(s)
- met alleen `p`: verwijder alle waarden voor die predicate

### `GET /def`
Leest alle definitie-triples uit de beheerde `def` graph.

### `GET /def/<term>`
Leest triples van een specifieke definitie, bijvoorbeeld:
`/def/Database`.

### `POST /def/<term>`
Voegt triples toe aan een definitie-resource in de `def` graph.

Body (verplicht):
```json
{
  "p": "http://www.w3.org/2000/01/rdf-schema#label",
  "o": "Database"
}
```

### `DELETE /def/<term>`
Verwijdert triples op een definitie-resource in de `def` graph.

Body:
```json
{
  "p": "http://www.w3.org/2000/01/rdf-schema#label",
  "o": "Database"
}
```

Gedrag:
- met `p` + `o`: verwijder exact matchende triple(s)
- met alleen `p`: verwijder alle waarden voor die predicate

### `POST /def` en `DELETE /def`
Zelfde triple-create/triple-delete maar op subject `https://kvan-todb.hualab.nl/def`.

### `POST /id/<db>`
Maakt database aan als die nog niet bestaat en voegt (minimaal) een label toe.
Bij nieuwe database wordt ook gezet:
`rdf:type https://kvan-todb.hualab.nl/def/Database`.
En:
`sdo:isPartOf https://kvan-todb.hualab.nl/id`.

Je kunt ook direct een custom triple toevoegen op het db-subject met:
```json
{
  "p": "http://www.w3.org/2000/01/rdf-schema#label",
  "o": "xNL-NmRAN_70"
}
```

### `DELETE /id/<db>`
Ondersteunt twee modi:
- Database verwijderen (cascade): verwijdert de volledige database inclusief alle graphs/triples.
- Triple verwijderen op database-subject: met `p` (en optioneel `o`) verwijdert triples van subject `/id/<db>` in de default graph.

Body (optioneel):
```json
{
  "cascade": true
}
```

Opmerking:
- Voor database-drop is expliciet vereist:
  - `"action": "drop_db"`
- Zonder deze action wordt geen database verwijderd.
- `cascade` default is `true`.

Triple-delete voorbeeld op db-level:
```json
{
  "p": "http://www.w3.org/2000/01/rdf-schema#label",
  "o": { "type": "literal", "value": "oude naam", "lang": "nl" }
}
```

Als `o` wordt weggelaten, worden alle waarden voor die `p` verwijderd.

Body (optioneel):
```json
{
  "label": "My Database"
}
```

### `GET /id/<db>/<graph>`
Leest alle triples uit de named graph.

Optioneel resolve-mechanisme (JSON-LD uitbreiden met gerelateerde nodes):
- `resolve`: JSON array van predicate IRIs
- `resolve_depth`: default `1`
- `resolve_direction`: `out|in|both` (default `out`)
- `resolve_limit`: default `1000`
- `resolve_include_root`: default `true`

Voorbeeld:
```text
/id/bs2324-deel1/scans?resolve=["https://schema.org/hasPart","https://schema.org/isPartOf","https://schema.org/text"]&resolve_direction=both&resolve_depth=4
```

Gedrag:
- response volgt normale content negotiation (`.ttl`, `.json`, `Accept`), ook met `resolve`;
- extra nodes/triples worden server-side toegevoegd op basis van de opgegeven predicates.

### `POST /id/<db>/<graph>`
Maakt graph aan als die nog niet bestaat en voegt (minimaal) een label toe.
Als database nog niet bestaat, wordt die ook automatisch aangemaakt.
Bij nieuwe graph wordt ook gezet:
`rdf:type https://kvan-todb.hualab.nl/def/Graph`.
En:
`sdo:isPartOf https://kvan-todb.hualab.nl/id/<db>`.

### `DELETE /id/<db>/<graph>`
Verwijdert de volledige graph (alle triples in die named graph) en haalt de
`sdo:hasPart` link vanaf de database weg.

Body (optioneel):
```json
{
  "action": "drop_graph",
  "cascade": true
}
```

Let op:
- Zonder `"action": "drop_graph"` wordt géén graph verwijderd.
- Met body `{ "p": "...", "o": ... }` doet deze route alleen triple-delete op het graph-subject.

Body (optioneel):
```json
{
  "label": "My Graph"
}
```

### `GET /id/<db>/<graph>/<resource>`
Leest triples van het subject `<resource>` binnen de opgegeven graph.
Ondersteunt dezelfde `resolve*` query-params als `GET /id/<db>/<graph>`.

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

### `DELETE /id/<db>/<graph>/<resource>`
Verwijdert triples op een resource-subject binnen de named graph.

Body:
```json
{
  "p": "http://www.w3.org/2000/01/rdf-schema#label",
  "o": "oude waarde"
}
```

Gedrag:
- met `p` + `o`: verwijder exact matchende triple(s)
- met alleen `p`: verwijder alle waarden voor die predicate op deze resource

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

Opmerking: in JSON-LD kan `rdf:type` verschijnen als `@type`.

Viewer redirect:
- Als er geen expliciete RDF content negotiation is (en geen `.ttl`/`.json`),
  redirectt de resolver naar de viewer met `303 See Other`.
- Target is: `VIEWER_BASE?uri=<de-opgevraagde-id-uri>`.
- `VIEWER_BASE` default: `https://kvan-todb.hualab.nl/viewer` (configureerbaar via env var).
- In `TEST_MODE_GET_WRITE=true` is deze redirect uitgeschakeld.

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
- Extra: `GET /id/<db>` zonder query params maakt in testmodus ook direct de database aan.

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

Definitie label toevoegen:
```bash
curl -i -X POST 'https://kvan-todb.hualab.nl/def/Database' \
  -H 'Content-Type: application/json' \
  -d '{"p":"http://www.w3.org/2000/01/rdf-schema#label","o":{"type":"literal","value":"Database","lang":"nl"}}'
```

Definitie opvragen als Turtle:
```bash
curl -i 'https://kvan-todb.hualab.nl/def/Database.ttl'
```

## Export all graphs

Exporteer alle databases met per database-map Turtle-bestanden voor:
- default graph (`default.ttl`)
- elke named graph (`<graphnaam>.ttl`)

Run:
```bash
uv run python scripts/export_all_graphs.py --data-dir app/data --output-dir exports
```

Optioneel interne system-store meenemen:
```bash
uv run python scripts/export_all_graphs.py --data-dir app/data --output-dir exports --include-system-db
```

## Relations API (voor viewer)

### `GET /api/relations/incoming`
Query params:
- `uri` (verplicht)
- `predicates` (optioneel, comma-separated IRIs)
- `graph` (optioneel)
- `offset` (default `0`)
- `limit` (default `100`, max `1000`)

Response bevat altijd `graph` per hit.

### `GET /api/relations/neighbors`
Query params:
- `uri` (verplicht)
- `direction` = `out|in|both` (default `both`)
- `predicates` (optioneel)
- `graph` (optioneel)
- `offset`, `limit`

Response bevat `direction` en `graph` per hit.

Defaults voor predicates (als `predicates` ontbreekt):
- `https://schema.org/hasPart`
- `https://schema.org/isPartOf`
