from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import Response
from pyoxigraph import DefaultGraph, Literal, NamedNode, Quad, Store
from rdflib import BNode, Graph, Literal as RdfLiteral, Namespace, URIRef


BASE_PUBLIC = os.getenv("PUBLIC_BASE", "https://kvan-todb.hualab.nl").rstrip("/")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
TEST_MODE_GET_WRITE = os.getenv("TEST_MODE_GET_WRITE", "false").strip().lower() in {"1", "true", "yes", "on"}
_STORE_CACHE: dict[str, Store] = {}
_STORE_CACHE_LOCK = Lock()
COMMON_PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "sdo": "https://schema.org/",
}
HAS_GRAPH_PREDICATE = f"{BASE_PUBLIC}/def/hasGraph"


@dataclass
class APIError(Exception):
    status_code: int
    code: str
    message: str
    details: dict[str, Any] | None = None


def _db_path(db: str) -> Path:
    return DATA_DIR / db


def _db_exists(db: str) -> bool:
    return _db_path(db).exists()


def _open_store(db: str) -> Store:
    with _STORE_CACHE_LOCK:
        cached = _STORE_CACHE.get(db)
        if cached is not None:
            return cached
    try:
        store = Store(str(_db_path(db)))
        with _STORE_CACHE_LOCK:
            _STORE_CACHE[db] = store
        return store
    except Exception as exc:
        raise APIError(
            500,
            "store_open_failed",
            f"Database `{db}` kon niet worden geopend",
            {"db": db, "cause": str(exc)},
        ) from exc


def _ensure_store(db: str) -> Store:
    with _STORE_CACHE_LOCK:
        cached = _STORE_CACHE.get(db)
        if cached is not None:
            return cached
    try:
        _db_path(db).mkdir(parents=True, exist_ok=True)
        store = Store(str(_db_path(db)))
        with _STORE_CACHE_LOCK:
            _STORE_CACHE[db] = store
        return store
    except Exception as exc:
        raise APIError(
            500,
            "store_create_failed",
            f"Database `{db}` kon niet worden aangemaakt/geopend",
            {"db": db, "cause": str(exc)},
        ) from exc


def _db_iri(db: str) -> str:
    return f"{BASE_PUBLIC}/id/{db}"


def _graph_iri(db: str, graph: str) -> str:
    return f"{BASE_PUBLIC}/id/{db}/{graph}"


def _resource_iri(db: str, graph: str, resource: str) -> str:
    return f"{BASE_PUBLIC}/id/{db}/{graph}/{resource}"


def _term_to_rdflib(term: Any) -> Any:
    cls_name = term.__class__.__name__
    if cls_name == "NamedNode":
        value = getattr(term, "value", str(term).strip("<>"))
        return URIRef(value)
    if cls_name == "BlankNode":
        value = getattr(term, "value", str(term))
        return BNode(value)
    if cls_name == "Literal":
        value = getattr(term, "value", str(term))
        language = getattr(term, "language", None)
        datatype = getattr(term, "datatype", None)
        dt_value = getattr(datatype, "value", None) if datatype else None
        if language:
            return RdfLiteral(value, lang=language)
        if dt_value:
            return RdfLiteral(value, datatype=URIRef(dt_value))
        return RdfLiteral(value)
    return URIRef(str(term).strip("<>"))


def _parse_object(payload_o: Any) -> Any:
    if isinstance(payload_o, str):
        return Literal(payload_o)
    if not isinstance(payload_o, dict):
        raise APIError(400, "invalid_payload", "`o` moet string of object zijn")

    object_type = payload_o.get("type", "literal")
    value = payload_o.get("value")
    if not value:
        raise APIError(400, "invalid_payload", "`o.value` ontbreekt")

    if object_type == "iri":
        return NamedNode(value)
    if object_type != "literal":
        raise APIError(400, "invalid_payload", "`o.type` moet `literal` of `iri` zijn")

    lang = payload_o.get("lang")
    datatype = payload_o.get("datatype")
    if lang:
        return Literal(value, language=lang)
    if datatype:
        return Literal(value, datatype=NamedNode(datatype))
    return Literal(value)


def _determine_format(request: Request) -> str:
    forced = getattr(request.state, "forced_format", None)
    if forced == "turtle":
        return "turtle"

    accept = (request.headers.get("accept") or "").lower()
    if "text/turtle" in accept or "application/x-turtle" in accept:
        return "turtle"
    return "json-ld"


async def _json_or_empty(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _serialize_graph(graph: Graph, output_format: str) -> tuple[str, str]:
    if output_format == "turtle":
        ttl = graph.serialize(format="turtle")
        return ttl, "text/turtle; charset=utf-8"
    jsonld = graph.serialize(format="json-ld")
    try:
        jsonld = json.dumps(json.loads(jsonld), indent=2, ensure_ascii=False)
    except Exception:
        pass
    return jsonld, "application/ld+json; charset=utf-8"


def _pretty_json_response(content: Any, status_code: int = 200, media_type: str = "application/json") -> Response:
    return Response(
        content=json.dumps(content, indent=2, ensure_ascii=False),
        status_code=status_code,
        media_type=f"{media_type}; charset=utf-8",
    )


def _serialize_error(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> Response:
    fmt = _determine_format(request)
    payload_details = details or {}
    if fmt == "turtle":
        g = Graph()
        err = BNode()
        EX = Namespace("https://kvan-todb.hualab.nl/vocab/error#")
        g.add((err, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), EX.Error))
        g.add((err, EX.code, RdfLiteral(code)))
        g.add((err, EX.message, RdfLiteral(message)))
        g.add((err, EX.statusCode, RdfLiteral(status_code)))
        if payload_details:
            g.add((err, EX.details, RdfLiteral(str(payload_details))))
        body = g.serialize(format="turtle")
        return Response(content=body, media_type="text/turtle; charset=utf-8", status_code=status_code)

    return _pretty_json_response(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": payload_details,
            }
        },
    )


def _resource_graph(store: Store, db: str, graph: str, resource: str) -> Graph:
    g = Graph()
    graph_node = NamedNode(_graph_iri(db, graph))
    subject = NamedNode(_resource_iri(db, graph, resource))

    for quad in store.quads_for_pattern(subject, None, None, graph_node):
        g.add((_term_to_rdflib(quad.subject), _term_to_rdflib(quad.predicate), _term_to_rdflib(quad.object)))
    return g


def _named_graph(store: Store, db: str, graph: str) -> Graph:
    g = Graph()
    graph_node = NamedNode(_graph_iri(db, graph))

    for quad in store.quads_for_pattern(None, None, None, graph_node):
        g.add((_term_to_rdflib(quad.subject), _term_to_rdflib(quad.predicate), _term_to_rdflib(quad.object)))
    return g


def _db_graph(store: Store, db: str) -> Graph:
    g = Graph()
    subject = NamedNode(_db_iri(db))
    for quad in store.quads_for_pattern(subject, None, None, DefaultGraph()):
        g.add((_term_to_rdflib(quad.subject), _term_to_rdflib(quad.predicate), _term_to_rdflib(quad.object)))
    return g


def _graph_exists(store: Store, db: str, graph: str) -> bool:
    graph_node = NamedNode(_graph_iri(db, graph))
    for _ in store.quads_for_pattern(None, None, None, graph_node):
        return True
    return False


def _insert_triple(
    store: Store,
    *,
    subject: str,
    predicate: str,
    obj: Any,
    graph_iri: str | None,
) -> None:
    graph_name = NamedNode(graph_iri) if graph_iri else DefaultGraph()
    quad = Quad(NamedNode(subject), NamedNode(predicate), _parse_object(obj), graph_name)
    store.add(quad)


def _link_db_graph(store: Store, db: str, graph: str) -> None:
    _insert_triple(
        store,
        subject=_db_iri(db),
        predicate=HAS_GRAPH_PREDICATE,
        obj={"type": "iri", "value": _graph_iri(db, graph)},
        graph_iri=None,
    )


def _predicate_from_query_key(key: str) -> str:
    if not key:
        raise APIError(400, "invalid_query_key", "Lege query key is niet toegestaan")
    if ":" in key:
        prefix, local = key.split(":", 1)
        base = COMMON_PREFIXES.get(prefix)
        if not base:
            raise APIError(
                400,
                "unknown_prefix",
                f"Onbekende prefix `{prefix}`",
                {
                    "key": key,
                    "known_prefixes": COMMON_PREFIXES,
                    "how_to_extend": (
                        "Voeg een entry toe aan COMMON_PREFIXES in app/main.py, "
                        "bijvoorbeeld: \"ex\": \"https://example.org/vocab/\""
                    ),
                },
            )
        if not local:
            raise APIError(400, "invalid_query_key", f"Prefix key `{key}` mist local part")
        return f"{base}{local}"
    return f"{BASE_PUBLIC}/def/{quote(key, safe='')}"


def _apply_get_test_writes(request: Request, *, db: str, graph: str | None = None, resource: str | None = None) -> None:
    if not TEST_MODE_GET_WRITE:
        return
    params = list(request.query_params.multi_items())
    if not params:
        return

    store = _ensure_store(db)
    graph_created = False
    if graph is not None:
        graph_created = not _graph_exists(store, db, graph)
    if graph is None:
        subject = _db_iri(db)
        graph_iri = None
    elif resource is None:
        subject = _graph_iri(db, graph)
        graph_iri = _graph_iri(db, graph)
    else:
        subject = _resource_iri(db, graph, resource)
        graph_iri = _graph_iri(db, graph)

    for key, value in params:
        _insert_triple(
            store,
            subject=subject,
            predicate=_predicate_from_query_key(key),
            obj=value,
            graph_iri=graph_iri,
        )
    if graph_created:
        _link_db_graph(store, db, graph)


app = FastAPI(title="Resolvable URIs", version="0.1.0")


@app.middleware("http")
async def ttl_extension_override(request: Request, call_next):
    path = request.scope.get("path", "")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
        request.scope["path"] = path
        request.scope["raw_path"] = path.encode("utf-8")
    if path.endswith(".ttl"):
        request.scope["path"] = path[:-4]
        request.scope["raw_path"] = path[:-4].encode("utf-8")
        request.state.forced_format = "turtle"
    if path.endswith(".json"):
        request.scope["path"] = path[:-5]
        request.scope["raw_path"] = path[:-5].encode("utf-8")
        request.state.forced_format = "json-ld"
    return await call_next(request)


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return _serialize_error(
        request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    return _serialize_error(
        request,
        status_code=500,
        code="internal_server_error",
        message="Onverwachte serverfout",
        details={"cause": str(exc)},
    )


@app.get("/id/{db}")
def get_db(db: str, request: Request):
    _apply_get_test_writes(request, db=db)
    if not _db_exists(db):
        raise APIError(404, "db_not_found", f"Database `{db}` bestaat niet")

    g = _db_graph(_open_store(db), db)
    if len(g) == 0:
        raise APIError(404, "db_description_not_found", "Geen triples gevonden voor database in default graph")

    body, media_type = _serialize_graph(g, _determine_format(request))
    return Response(content=body, media_type=media_type)


@app.post("/id/{db}")
async def post_db(db: str, request: Request):
    payload = await _json_or_empty(request)
    label = payload.get("label", db)

    created = not _db_exists(db)
    store = _ensure_store(db)

    _insert_triple(
        store,
        subject=_db_iri(db),
        predicate="http://www.w3.org/2000/01/rdf-schema#label",
        obj={"type": "literal", "value": label},
        graph_iri=None,
    )

    status = 201 if created else 200
    return _pretty_json_response({"db": db, "created": created}, status_code=status)


@app.get("/id/{db}/{graph}")
def get_graph(db: str, graph: str, request: Request):
    _apply_get_test_writes(request, db=db, graph=graph)
    if not _db_exists(db):
        raise APIError(404, "db_not_found", f"Database `{db}` bestaat niet")

    store = _open_store(db)
    g = _named_graph(store, db, graph)
    if len(g) == 0:
        raise APIError(404, "graph_not_found", f"Graph `{graph}` bestaat niet of bevat geen triples")

    body, media_type = _serialize_graph(g, _determine_format(request))
    return Response(content=body, media_type=media_type)


@app.post("/id/{db}/{graph}")
async def post_graph(db: str, graph: str, request: Request):
    payload = await _json_or_empty(request)
    label = payload.get("label", graph)

    db_created = not _db_exists(db)
    store = _ensure_store(db)
    existed = _graph_exists(store, db, graph)

    _insert_triple(
        store,
        subject=_graph_iri(db, graph),
        predicate="http://www.w3.org/2000/01/rdf-schema#label",
        obj={"type": "literal", "value": label},
        graph_iri=_graph_iri(db, graph),
    )
    if not existed:
        _link_db_graph(store, db, graph)

    status = 200 if existed else 201
    return _pretty_json_response(
        {"db": db, "graph": graph, "db_created": db_created, "created": not existed},
        status_code=status,
    )


@app.get("/id/{db}/{graph}/{resource:path}")
def get_resource(db: str, graph: str, resource: str, request: Request):
    _apply_get_test_writes(request, db=db, graph=graph, resource=resource)
    if not _db_exists(db):
        raise APIError(404, "db_not_found", f"Database `{db}` bestaat niet")

    store = _open_store(db)
    g = _resource_graph(store, db, graph, resource)
    if len(g) == 0:
        raise APIError(404, "resource_not_found", f"Resource `{resource}` niet gevonden")

    body, media_type = _serialize_graph(g, _determine_format(request))
    return Response(content=body, media_type=media_type)


@app.post("/id/{db}/{graph}/{resource:path}")
async def post_resource(db: str, graph: str, resource: str, request: Request):
    payload = await request.json()
    if not isinstance(payload, dict):
        raise APIError(400, "invalid_payload", "Payload moet JSON object zijn")

    p = payload.get("p")
    o = payload.get("o")
    if not p or o is None:
        raise APIError(400, "invalid_payload", "Payload vereist minimaal `p` en `o`")

    db_created = not _db_exists(db)
    store = _ensure_store(db)
    graph_created = not _graph_exists(store, db, graph)

    _insert_triple(
        store,
        subject=_resource_iri(db, graph, resource),
        predicate=p,
        obj=o,
        graph_iri=_graph_iri(db, graph),
    )
    if graph_created:
        _link_db_graph(store, db, graph)

    return _pretty_json_response(
        {
            "db": db,
            "graph": graph,
            "resource": resource,
            "db_created": db_created,
            "graph_created": graph_created,
            "created": True,
        },
        status_code=201,
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}
