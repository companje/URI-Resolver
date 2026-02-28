from __future__ import annotations

import gc
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Query, Request
from fastapi.responses import RedirectResponse, Response
from pyoxigraph import DefaultGraph, Literal, NamedNode, Quad, Store
from rdflib import BNode, Graph, Literal as RdfLiteral, Namespace, URIRef


BASE_PUBLIC = os.getenv("PUBLIC_BASE", "https://kvan-todb.hualab.nl").rstrip("/")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
TEST_MODE_GET_WRITE = os.getenv("TEST_MODE_GET_WRITE", "false").strip().lower() in {"1", "true", "yes", "on"}
VIEWER_BASE = os.getenv("VIEWER_BASE", f"{BASE_PUBLIC}").rstrip("/")
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
HAS_PART_PREDICATE = "https://schema.org/hasPart"
IS_PART_OF_PREDICATE = "https://schema.org/isPartOf"
DEFAULT_RELATION_PREDICATES = (HAS_PART_PREDICATE, IS_PART_OF_PREDICATE)
RDF_TYPE_PREDICATE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
DB_TYPE_IRI = f"{BASE_PUBLIC}/def/Database"
GRAPH_TYPE_IRI = f"{BASE_PUBLIC}/def/Graph"
SYSTEM_TYPE_IRI = f"{BASE_PUBLIC}/def/System"
DEF_GRAPH_IRI = f"{BASE_PUBLIC}/def"
SYSTEM_DB_NAME = "__system__"
RDFS_LABEL_PREDICATE = "http://www.w3.org/2000/01/rdf-schema#label"
RDFS_CLASS_IRI = "http://www.w3.org/2000/01/rdf-schema#Class"
RDF_PROPERTY_IRI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
_DEF_SEEDED = False
_DEF_SEEDED_LOCK = Lock()
_SYSTEM_METADATA_SEEDED = False
_SYSTEM_METADATA_SEEDED_LOCK = Lock()


def _system_iri() -> str:
    return f"{BASE_PUBLIC}/id"


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


def _evict_store(db: str) -> None:
    with _STORE_CACHE_LOCK:
        _STORE_CACHE.pop(db, None)
    gc.collect()


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


def _delete_db(db: str) -> None:
    if db == SYSTEM_DB_NAME:
        raise APIError(403, "forbidden", "Systeemdatabase kan niet verwijderd worden")
    path = _db_path(db)
    if not path.exists():
        raise APIError(404, "db_not_found", f"Database `{db}` bestaat niet")
    _evict_store(db)
    try:
        shutil.rmtree(path)
    except Exception as exc:
        raise APIError(
            500,
            "db_delete_failed",
            f"Database `{db}` kon niet worden verwijderd",
            {"db": db, "cause": str(exc)},
        ) from exc


def _db_iri(db: str) -> str:
    return f"{BASE_PUBLIC}/id/{db}"


def _graph_iri(db: str, graph: str) -> str:
    return f"{BASE_PUBLIC}/id/{db}/{graph}"


def _resource_iri(db: str, graph: str, resource: str) -> str:
    return f"{BASE_PUBLIC}/id/{db}/{graph}/{resource}"


def _def_iri(term: str) -> str:
    return f"{BASE_PUBLIC}/def/{term}"


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


def _has_explicit_rdf_negotiation(request: Request) -> bool:
    forced = getattr(request.state, "forced_format", None)
    if forced in {"turtle", "json-ld"}:
        return True
    accept = (request.headers.get("accept") or "").lower()
    if not accept:
        return False
    rdf_media_types = (
        "application/ld+json",
        "text/turtle",
        "application/x-turtle",
        "application/json",
        "application/rdf+xml",
        "application/n-triples",
        "application/n-quads",
        "application/trig",
    )
    return any(mt in accept for mt in rdf_media_types)


def _viewer_redirect_if_needed(request: Request) -> Response | None:
    if TEST_MODE_GET_WRITE and request.query_params:
        return None
    requested_suffix_format = getattr(request.state, "requested_suffix_format", None)
    if requested_suffix_format in {"turtle", "json-ld"}:
        return None
    original_path = getattr(request.state, "original_request_path", "") or ""
    if original_path.endswith(".ttl") or original_path.endswith(".json") or original_path.endswith(".jsonld"):
        return None
    if _has_explicit_rdf_negotiation(request):
        return None
    path = request.scope.get("path", "")
    if not (path.startswith("/id") or path.startswith("/def")):
        return None
    target_uri = f"{BASE_PUBLIC}{path}"
    location = f"{VIEWER_BASE}?uri={quote(target_uri, safe='')}"
    return RedirectResponse(url=location, status_code=303)


def _incoming_requested(request: Request) -> bool:
    if "incoming" not in request.query_params:
        return False
    raw = (request.query_params.get("incoming") or "").strip().lower()
    return raw in {"", "1", "true", "yes", "on"}


def _incoming_response_for_subject(*, request: Request, subject_uri: str) -> Response:
    raw_offset = (request.query_params.get("incoming_offset") or "0").strip()
    raw_limit = (request.query_params.get("incoming_limit") or "10").strip()
    raw_predicates = request.query_params.get("incoming_predicates")
    graph_filter = request.query_params.get("incoming_graph")

    try:
        offset = max(0, int(raw_offset))
    except ValueError:
        raise APIError(400, "invalid_incoming_offset", "`incoming_offset` moet integer >= 0 zijn")
    try:
        limit = int(raw_limit)
    except ValueError:
        raise APIError(400, "invalid_incoming_limit", "`incoming_limit` moet integer zijn")
    if limit < 1 or limit > 1000:
        raise APIError(400, "invalid_incoming_limit", "`incoming_limit` moet tussen 1 en 1000 liggen")

    predicate_filter = _parse_predicates_param(raw_predicates)

    items = _collect_incoming_relations(
        uri=subject_uri,
        predicate_filter=predicate_filter,
        graph_filter=graph_filter,
    )
    total = len(items)
    page = items[offset : offset + limit]
    next_offset = offset + limit if (offset + limit) < total else None
    return _pretty_json_response(
        {
            "uri": subject_uri,
            "offset": offset,
            "limit": limit,
            "total": total,
            "next_offset": next_offset,
            "items": page,
        }
    )


async def _json_or_empty(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _serialize_graph(graph: Graph, output_format: str) -> tuple[str, str]:
    graph.bind("def", URIRef(f"{BASE_PUBLIC}/def/"), override=True)
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


def _subject_graph(store: Store, subject_iri: str, graph_iri: str | None) -> Graph:
    g = Graph()
    graph_name = NamedNode(graph_iri) if graph_iri else DefaultGraph()
    subject = NamedNode(subject_iri)
    for quad in store.quads_for_pattern(subject, None, None, graph_name):
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


def _system_graph() -> Graph:
    _ensure_system_metadata_seeded()
    store = _system_store()
    g = _subject_graph(store, _system_iri(), None)
    system = URIRef(_system_iri())
    if DATA_DIR.exists():
        for child in sorted(DATA_DIR.iterdir(), key=lambda p: p.name):
            if child.is_dir() and child.name != SYSTEM_DB_NAME:
                g.add((system, URIRef(HAS_PART_PREDICATE), URIRef(_db_iri(child.name))))
    return g


def _graph_exists(store: Store, db: str, graph: str) -> bool:
    graph_node = NamedNode(_graph_iri(db, graph))
    for _ in store.quads_for_pattern(None, None, None, graph_node):
        return True
    return False


def _delete_graph(store: Store, db: str, graph: str) -> tuple[int, int]:
    graph_node = NamedNode(_graph_iri(db, graph))
    graph_quads = list(store.quads_for_pattern(None, None, None, graph_node))
    for quad in graph_quads:
        store.remove(quad)

    link_deleted = _delete_triples(
        store,
        subject=_db_iri(db),
        predicate=HAS_PART_PREDICATE,
        obj={"type": "iri", "value": _graph_iri(db, graph)},
        graph_iri=None,
    )
    return len(graph_quads), link_deleted


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


def _delete_triples(
    store: Store,
    *,
    subject: str,
    predicate: str | None,
    obj: Any | None,
    graph_iri: str | None,
) -> int:
    graph_name = NamedNode(graph_iri) if graph_iri else DefaultGraph()
    p_term = NamedNode(predicate) if predicate else None
    o_term = _parse_object(obj) if obj is not None else None
    s_term = NamedNode(subject)
    to_remove = list(store.quads_for_pattern(s_term, p_term, o_term, graph_name))
    for quad in to_remove:
        store.remove(quad)
    return len(to_remove)


def _db_names(*, include_system: bool = False) -> list[str]:
    if not DATA_DIR.exists():
        return []
    names = [p.name for p in DATA_DIR.iterdir() if p.is_dir()]
    if not include_system:
        names = [n for n in names if n != SYSTEM_DB_NAME]
    return sorted(names)


def _graph_name_to_str(graph_name: Any) -> str | None:
    cls_name = graph_name.__class__.__name__
    if cls_name == "DefaultGraph":
        return None
    if cls_name == "NamedNode":
        return getattr(graph_name, "value", str(graph_name).strip("<>"))
    return str(graph_name)


def _term_to_api_str(term: Any) -> str:
    cls_name = term.__class__.__name__
    if cls_name == "NamedNode":
        return getattr(term, "value", str(term).strip("<>"))
    if cls_name == "BlankNode":
        return f"_:{getattr(term, 'value', str(term))}"
    if cls_name == "Literal":
        return getattr(term, "value", str(term))
    return str(term)


def _parse_predicates_param(predicates: str | None) -> set[str] | None:
    if not predicates:
        return None
    values = {p.strip() for p in predicates.split(",") if p.strip()}
    return values or None


def _expand_prefixed_predicate(value: str) -> str:
    if ":" not in value:
        return value
    prefix, local = value.split(":", 1)
    base = COMMON_PREFIXES.get(prefix)
    if not base or not local:
        return value
    return f"{base}{local}"


def _parse_resolve_predicates_param(resolve: str | None) -> set[str] | None:
    if not resolve:
        return None
    try:
        raw = json.loads(resolve)
    except Exception:
        raise APIError(
            400,
            "invalid_resolve",
            "`resolve` moet JSON array zijn, bv [\"https://schema.org/text\"]",
        )
    if not isinstance(raw, list) or not all(isinstance(v, str) for v in raw):
        raise APIError(400, "invalid_resolve", "`resolve` moet array van strings zijn")
    values = {_expand_prefixed_predicate(v.strip()) for v in raw if v.strip()}
    return values or None


def _copy_graph(input_graph: Graph) -> Graph:
    out = Graph()
    for s, p, o in input_graph:
        out.add((s, p, o))
    return out


def _add_subject_description(
    *,
    store: Store,
    subject: NamedNode,
    out_graph: Graph,
    seen_triples: set[tuple[str, str, str]],
    remaining_budget: int,
) -> int:
    if remaining_budget <= 0:
        return 0
    added = 0
    for quad in store.quads_for_pattern(subject, None, None, None):
        s = _term_to_rdflib(quad.subject)
        p = _term_to_rdflib(quad.predicate)
        o = _term_to_rdflib(quad.object)
        key = (str(s), str(p), str(o))
        if key in seen_triples:
            continue
        out_graph.add((s, p, o))
        seen_triples.add(key)
        added += 1
        if added >= remaining_budget:
            break
    return added


def _resolve_graph_neighbors(
    *,
    store: Store,
    base_graph: Graph,
    predicates: set[str],
    direction: str,
    depth: int,
    limit: int,
    include_root: bool,
) -> Graph:
    out_graph = _copy_graph(base_graph) if include_root else Graph()
    seen_triples: set[tuple[str, str, str]] = {(str(s), str(p), str(o)) for s, p, o in out_graph}
    start_nodes = {URIRef(str(s)) for s in set(base_graph.subjects()) if isinstance(s, URIRef)}
    start_nodes.update({URIRef(str(o)) for o in set(base_graph.objects()) if isinstance(o, URIRef)})
    frontier = {NamedNode(str(n)) for n in start_nodes}
    visited_nodes = {str(n) for n in frontier}
    remaining_budget = max(0, limit)

    for _ in range(depth):
        if not frontier or remaining_budget <= 0:
            break
        next_frontier: set[NamedNode] = set()
        for node in frontier:
            if remaining_budget <= 0:
                break
            if direction in {"out", "both"}:
                for quad in store.quads_for_pattern(node, None, None, None):
                    p_value = _term_to_api_str(quad.predicate)
                    if p_value not in predicates:
                        continue
                    s = _term_to_rdflib(quad.subject)
                    p = _term_to_rdflib(quad.predicate)
                    o = _term_to_rdflib(quad.object)
                    key = (str(s), str(p), str(o))
                    if key not in seen_triples and remaining_budget > 0:
                        out_graph.add((s, p, o))
                        seen_triples.add(key)
                        remaining_budget -= 1
                    if quad.object.__class__.__name__ == "NamedNode":
                        reached = NamedNode(_term_to_api_str(quad.object))
                        if str(reached) not in visited_nodes:
                            next_frontier.add(reached)
                            visited_nodes.add(str(reached))
                            used = _add_subject_description(
                                store=store,
                                subject=reached,
                                out_graph=out_graph,
                                seen_triples=seen_triples,
                                remaining_budget=remaining_budget,
                            )
                            remaining_budget -= used

            if direction in {"in", "both"}:
                for quad in store.quads_for_pattern(None, None, node, None):
                    p_value = _term_to_api_str(quad.predicate)
                    if p_value not in predicates:
                        continue
                    s = _term_to_rdflib(quad.subject)
                    p = _term_to_rdflib(quad.predicate)
                    o = _term_to_rdflib(quad.object)
                    key = (str(s), str(p), str(o))
                    if key not in seen_triples and remaining_budget > 0:
                        out_graph.add((s, p, o))
                        seen_triples.add(key)
                        remaining_budget -= 1
                    if quad.subject.__class__.__name__ == "NamedNode":
                        reached = NamedNode(_term_to_api_str(quad.subject))
                        if str(reached) not in visited_nodes:
                            next_frontier.add(reached)
                            visited_nodes.add(str(reached))
                            used = _add_subject_description(
                                store=store,
                                subject=reached,
                                out_graph=out_graph,
                                seen_triples=seen_triples,
                                remaining_budget=remaining_budget,
                            )
                            remaining_budget -= used
        frontier = next_frontier
    return out_graph


def _candidate_dbs_for_graph(graph: str | None) -> list[str]:
    if not graph:
        return _db_names()
    prefix = f"{BASE_PUBLIC}/id/"
    if graph.startswith(prefix):
        rest = graph[len(prefix) :]
        db_name = rest.split("/", 1)[0] if rest else ""
        if db_name:
            return [db_name]
    return _db_names()


def _candidate_store_names_for_relations(graph: str | None) -> list[str]:
    if graph:
        prefix = f"{BASE_PUBLIC}/id/"
        if graph.startswith(prefix):
            rest = graph[len(prefix) :]
            db_name = rest.split("/", 1)[0] if rest else ""
            if db_name:
                return [db_name]
        if graph == DEF_GRAPH_IRI:
            return [SYSTEM_DB_NAME]
    names: list[str] = [SYSTEM_DB_NAME, *_db_names()]
    # preserve order while de-duplicating
    return list(dict.fromkeys(names))


def _collect_incoming_relations(
    *,
    uri: str,
    predicate_filter: set[str] | None,
    graph_filter: str | None,
) -> list[dict[str, Any]]:
    target = NamedNode(uri)
    graph_node = NamedNode(graph_filter) if graph_filter else None
    items: list[dict[str, Any]] = []
    for db_name in _candidate_store_names_for_relations(graph_filter):
        if db_name == SYSTEM_DB_NAME:
            store = _system_store()
        else:
            if not _db_exists(db_name):
                continue
            store = _open_store(db_name)
        quads = store.quads_for_pattern(None, None, target, graph_node)
        for quad in quads:
            predicate = _term_to_api_str(quad.predicate)
            if predicate_filter and predicate not in predicate_filter:
                continue
            graph_value = _graph_name_to_str(quad.graph_name)
            if graph_filter and graph_value != graph_filter:
                continue
            items.append(
                {
                    "subject": _term_to_api_str(quad.subject),
                    "predicate": predicate,
                    "object": uri,
                    "graph": graph_value,
                }
            )
    items.sort(key=lambda i: (i["subject"], i["predicate"], i["graph"] or ""))
    return items


def _collect_neighbor_relations(
    *,
    uri: str,
    direction: str,
    predicate_filter: set[str] | None,
    graph_filter: str | None,
) -> list[dict[str, Any]]:
    target = NamedNode(uri)
    graph_node = NamedNode(graph_filter) if graph_filter else None
    items: list[dict[str, Any]] = []
    for db_name in _candidate_store_names_for_relations(graph_filter):
        if db_name == SYSTEM_DB_NAME:
            store = _system_store()
        else:
            if not _db_exists(db_name):
                continue
            store = _open_store(db_name)

        if direction in {"out", "both"}:
            for quad in store.quads_for_pattern(target, None, None, graph_node):
                predicate = _term_to_api_str(quad.predicate)
                if predicate_filter and predicate not in predicate_filter:
                    continue
                graph_value = _graph_name_to_str(quad.graph_name)
                if graph_filter and graph_value != graph_filter:
                    continue
                items.append(
                    {
                        "direction": "out",
                        "subject": uri,
                        "predicate": predicate,
                        "object": _term_to_api_str(quad.object),
                        "graph": graph_value,
                    }
                )

        if direction in {"in", "both"}:
            for quad in store.quads_for_pattern(None, None, target, graph_node):
                predicate = _term_to_api_str(quad.predicate)
                if predicate_filter and predicate not in predicate_filter:
                    continue
                graph_value = _graph_name_to_str(quad.graph_name)
                if graph_filter and graph_value != graph_filter:
                    continue
                items.append(
                    {
                        "direction": "in",
                        "subject": _term_to_api_str(quad.subject),
                        "predicate": predicate,
                        "object": uri,
                        "graph": graph_value,
                    }
                )

    items.sort(
        key=lambda i: (
            0 if i["direction"] == "out" else 1,
            i["subject"],
            i["predicate"],
            i["object"],
            i["graph"] or "",
        )
    )
    return items


def _link_db_graph(store: Store, db: str, graph: str) -> None:
    _insert_triple(
        store,
        subject=_db_iri(db),
        predicate=HAS_PART_PREDICATE,
        obj={"type": "iri", "value": _graph_iri(db, graph)},
        graph_iri=None,
    )


def _link_db_system(store: Store, db: str) -> None:
    _insert_triple(
        store,
        subject=_db_iri(db),
        predicate=IS_PART_OF_PREDICATE,
        obj={"type": "iri", "value": _system_iri()},
        graph_iri=None,
    )


def _link_graph_db(store: Store, db: str, graph: str) -> None:
    _insert_triple(
        store,
        subject=_graph_iri(db, graph),
        predicate=IS_PART_OF_PREDICATE,
        obj={"type": "iri", "value": _db_iri(db)},
        graph_iri=_graph_iri(db, graph),
    )


def _add_db_type(store: Store, db: str) -> None:
    _insert_triple(
        store,
        subject=_db_iri(db),
        predicate=RDF_TYPE_PREDICATE,
        obj={"type": "iri", "value": DB_TYPE_IRI},
        graph_iri=None,
    )


def _add_graph_type(store: Store, db: str, graph: str) -> None:
    _insert_triple(
        store,
        subject=_graph_iri(db, graph),
        predicate=RDF_TYPE_PREDICATE,
        obj={"type": "iri", "value": GRAPH_TYPE_IRI},
        graph_iri=_graph_iri(db, graph),
    )


def _system_store() -> Store:
    return _ensure_store(SYSTEM_DB_NAME)


def _ensure_system_metadata_seeded() -> None:
    global _SYSTEM_METADATA_SEEDED
    with _SYSTEM_METADATA_SEEDED_LOCK:
        if _SYSTEM_METADATA_SEEDED:
            return
        store = _system_store()
        _insert_triple(
            store,
            subject=_system_iri(),
            predicate=RDF_TYPE_PREDICATE,
            obj={"type": "iri", "value": SYSTEM_TYPE_IRI},
            graph_iri=None,
        )
        _insert_triple(
            store,
            subject=_system_iri(),
            predicate=RDFS_LABEL_PREDICATE,
            obj={"type": "literal", "value": "Resolvable URI System"},
            graph_iri=None,
        )
        _SYSTEM_METADATA_SEEDED = True


def _ensure_def_seeded() -> None:
    global _DEF_SEEDED
    with _DEF_SEEDED_LOCK:
        if _DEF_SEEDED:
            return
        store = _system_store()
        seed = [
            (_def_iri("System"), RDF_TYPE_PREDICATE, {"type": "iri", "value": RDFS_CLASS_IRI}),
            (_def_iri("System"), RDFS_LABEL_PREDICATE, {"type": "literal", "value": "System"}),
            (_def_iri("Database"), RDF_TYPE_PREDICATE, {"type": "iri", "value": RDFS_CLASS_IRI}),
            (_def_iri("Database"), RDFS_LABEL_PREDICATE, {"type": "literal", "value": "Database"}),
            (_def_iri("Graph"), RDF_TYPE_PREDICATE, {"type": "iri", "value": RDFS_CLASS_IRI}),
            (_def_iri("Graph"), RDFS_LABEL_PREDICATE, {"type": "literal", "value": "Graph"}),
        ]
        for subject, predicate, obj in seed:
            _insert_triple(
                store,
                subject=subject,
                predicate=predicate,
                obj=obj,
                graph_iri=DEF_GRAPH_IRI,
            )
        _DEF_SEEDED = True


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

    db_created = not _db_exists(db)
    store = _ensure_store(db)
    if db_created:
        _add_db_type(store, db)
        _link_db_system(store, db)
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
        _add_graph_type(store, db, graph)
        _link_graph_db(store, db, graph)
        _link_db_graph(store, db, graph)


app = FastAPI(title="Resolvable URIs", version="0.1.0")


@app.middleware("http")
async def ttl_extension_override(request: Request, call_next):
    path = request.scope.get("path", "")
    request.state.original_request_path = path
    request.state.requested_suffix_format = None
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
        request.scope["path"] = path
        request.scope["raw_path"] = path.encode("utf-8")
    if path.endswith(".ttl"):
        request.scope["path"] = path[:-4]
        request.scope["raw_path"] = path[:-4].encode("utf-8")
        request.state.forced_format = "turtle"
        request.state.requested_suffix_format = "turtle"
    if path.endswith(".json"):
        request.scope["path"] = path[:-5]
        request.scope["raw_path"] = path[:-5].encode("utf-8")
        request.state.forced_format = "json-ld"
        request.state.requested_suffix_format = "json-ld"
    if path.endswith(".jsonld"):
        request.scope["path"] = path[:-7]
        request.scope["raw_path"] = path[:-7].encode("utf-8")
        request.state.forced_format = "json-ld"
        request.state.requested_suffix_format = "json-ld"
    return await call_next(request)


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    redirectable_not_found_codes = {
        "definition_not_found",
        "db_not_found",
        "db_description_not_found",
        "graph_not_found",
        "resource_not_found",
    }
    if request.method == "GET" and exc.code in redirectable_not_found_codes:
        redirect = _viewer_redirect_if_needed(request)
        if redirect is not None:
            return redirect
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
    if TEST_MODE_GET_WRITE and not _db_exists(db):
        store = _ensure_store(db)
        _add_db_type(store, db)
        _link_db_system(store, db)
        _insert_triple(
            store,
            subject=_db_iri(db),
            predicate="http://www.w3.org/2000/01/rdf-schema#label",
            obj={"type": "literal", "value": db},
            graph_iri=None,
        )
    _apply_get_test_writes(request, db=db)
    if not _db_exists(db):
        raise APIError(404, "db_not_found", f"Database `{db}` bestaat niet")

    store = _open_store(db)
    if _incoming_requested(request):
        return _incoming_response_for_subject(request=request, subject_uri=_db_iri(db))
    _add_db_type(store, db)
    g = _db_graph(store, db)
    if len(g) == 0:
        raise APIError(404, "db_description_not_found", "Geen triples gevonden voor database in default graph")

    redirect = _viewer_redirect_if_needed(request)
    if redirect is not None:
        return redirect
    body, media_type = _serialize_graph(g, _determine_format(request))
    return Response(content=body, media_type=media_type)


@app.get("/id")
def get_system(request: Request):
    if _incoming_requested(request):
        return _incoming_response_for_subject(request=request, subject_uri=_system_iri())
    redirect = _viewer_redirect_if_needed(request)
    if redirect is not None:
        return redirect
    g = _system_graph()
    body, media_type = _serialize_graph(g, _determine_format(request))
    return Response(content=body, media_type=media_type)


@app.post("/id")
async def post_system(request: Request):
    payload = await _json_or_empty(request)
    p = payload.get("p")
    o = payload.get("o")
    if not p or o is None:
        raise APIError(400, "invalid_payload", "Payload vereist minimaal `p` en `o`")
    _ensure_system_metadata_seeded()
    store = _system_store()
    _insert_triple(
        store,
        subject=_system_iri(),
        predicate=p,
        obj=o,
        graph_iri=None,
    )
    return _pretty_json_response({"system": _system_iri(), "created": True}, status_code=201)


@app.delete("/id")
async def delete_system_triples(request: Request):
    payload = await _json_or_empty(request)
    p = payload.get("p")
    if not p:
        raise APIError(400, "invalid_payload", "Payload vereist `p`")
    o = payload.get("o")
    _ensure_system_metadata_seeded()
    store = _system_store()
    deleted = _delete_triples(
        store,
        subject=_system_iri(),
        predicate=p,
        obj=o,
        graph_iri=None,
    )
    return _pretty_json_response(
        {"system": _system_iri(), "deleted_triples": deleted, "mode": "triple_delete"},
        status_code=200,
    )


@app.get("/def")
def get_definitions(
    request: Request,
    resolve: str | None = Query(None, description="JSON array met predicate IRIs"),
    resolve_depth: int = Query(1, ge=1, le=10),
    resolve_direction: str = Query("out", pattern="^(out|in|both)$"),
    resolve_limit: int = Query(1000, ge=1, le=50000),
    resolve_include_root: bool = Query(True),
):
    if _incoming_requested(request):
        return _incoming_response_for_subject(request=request, subject_uri=DEF_GRAPH_IRI)
    _ensure_def_seeded()
    store = _system_store()
    g = Graph()
    graph_node = NamedNode(DEF_GRAPH_IRI)
    for quad in store.quads_for_pattern(None, None, None, graph_node):
        g.add((_term_to_rdflib(quad.subject), _term_to_rdflib(quad.predicate), _term_to_rdflib(quad.object)))
    resolve_predicates = _parse_resolve_predicates_param(resolve)
    if resolve_predicates:
        g = _resolve_graph_neighbors(
            store=store,
            base_graph=g,
            predicates=resolve_predicates,
            direction=resolve_direction,
            depth=resolve_depth,
            limit=resolve_limit,
            include_root=resolve_include_root,
        )
    if resolve is None:
        redirect = _viewer_redirect_if_needed(request)
        if redirect is not None:
            return redirect
    body, media_type = _serialize_graph(g, _determine_format(request))
    return Response(content=body, media_type=media_type)


@app.get("/def/{term:path}")
def get_definition(
    term: str,
    request: Request,
    resolve: str | None = Query(None, description="JSON array met predicate IRIs"),
    resolve_depth: int = Query(1, ge=1, le=10),
    resolve_direction: str = Query("out", pattern="^(out|in|both)$"),
    resolve_limit: int = Query(1000, ge=1, le=50000),
    resolve_include_root: bool = Query(True),
):
    _ensure_def_seeded()
    subject_uri = _def_iri(term)
    if _incoming_requested(request):
        return _incoming_response_for_subject(request=request, subject_uri=subject_uri)
    store = _system_store()
    g = _subject_graph(store, subject_uri, DEF_GRAPH_IRI)
    if len(g) == 0:
        raise APIError(404, "definition_not_found", f"Definitie `{term}` niet gevonden")
    resolve_predicates = _parse_resolve_predicates_param(resolve)
    if resolve_predicates:
        g = _resolve_graph_neighbors(
            store=store,
            base_graph=g,
            predicates=resolve_predicates,
            direction=resolve_direction,
            depth=resolve_depth,
            limit=resolve_limit,
            include_root=resolve_include_root,
        )
    if resolve is None:
        redirect = _viewer_redirect_if_needed(request)
        if redirect is not None:
            return redirect
    body, media_type = _serialize_graph(g, _determine_format(request))
    return Response(content=body, media_type=media_type)


@app.post("/def/{term:path}")
async def post_definition(term: str, request: Request):
    _ensure_def_seeded()
    payload = await request.json()
    if not isinstance(payload, dict):
        raise APIError(400, "invalid_payload", "Payload moet JSON object zijn")
    p = payload.get("p")
    o = payload.get("o")
    if not p or o is None:
        raise APIError(400, "invalid_payload", "Payload vereist minimaal `p` en `o`")
    store = _system_store()
    _insert_triple(
        store,
        subject=_def_iri(term),
        predicate=p,
        obj=o,
        graph_iri=DEF_GRAPH_IRI,
    )
    return _pretty_json_response({"term": term, "created": True}, status_code=201)


@app.delete("/def/{term:path}")
async def delete_definition(term: str, request: Request):
    payload = await _json_or_empty(request)
    p = payload.get("p")
    if not p:
        raise APIError(400, "invalid_payload", "Payload vereist `p`")
    o = payload.get("o")
    _ensure_def_seeded()
    store = _system_store()
    deleted = _delete_triples(
        store,
        subject=_def_iri(term),
        predicate=p,
        obj=o,
        graph_iri=DEF_GRAPH_IRI,
    )
    return _pretty_json_response(
        {"term": term, "deleted_triples": deleted, "mode": "triple_delete"},
        status_code=200,
    )


@app.post("/def")
async def post_def_root(request: Request):
    payload = await _json_or_empty(request)
    p = payload.get("p")
    o = payload.get("o")
    if not p or o is None:
        raise APIError(400, "invalid_payload", "Payload vereist minimaal `p` en `o`")
    _ensure_def_seeded()
    store = _system_store()
    _insert_triple(
        store,
        subject=DEF_GRAPH_IRI,
        predicate=p,
        obj=o,
        graph_iri=DEF_GRAPH_IRI,
    )
    return _pretty_json_response({"term": "def", "created": True}, status_code=201)


@app.delete("/def")
async def delete_def_root(request: Request):
    payload = await _json_or_empty(request)
    p = payload.get("p")
    if not p:
        raise APIError(400, "invalid_payload", "Payload vereist `p`")
    o = payload.get("o")
    _ensure_def_seeded()
    store = _system_store()
    deleted = _delete_triples(
        store,
        subject=DEF_GRAPH_IRI,
        predicate=p,
        obj=o,
        graph_iri=DEF_GRAPH_IRI,
    )
    return _pretty_json_response(
        {"term": "def", "deleted_triples": deleted, "mode": "triple_delete"},
        status_code=200,
    )


@app.post("/id/{db}")
async def post_db(db: str, request: Request):
    payload = await _json_or_empty(request)

    created = not _db_exists(db)
    store = _ensure_store(db)
    _add_db_type(store, db)
    if created:
        _link_db_system(store, db)

    p = payload.get("p")
    o = payload.get("o")
    if p and o is not None:
        _insert_triple(
            store,
            subject=_db_iri(db),
            predicate=p,
            obj=o,
            graph_iri=None,
        )
        status = 201 if created else 200
        return _pretty_json_response(
            {"db": db, "created": created, "triple_created": True, "mode": "triple_create"},
            status_code=status,
        )

    label = payload.get("label", db)
    _insert_triple(
        store,
        subject=_db_iri(db),
        predicate=RDFS_LABEL_PREDICATE,
        obj={"type": "literal", "value": label},
        graph_iri=None,
    )

    status = 201 if created else 200
    return _pretty_json_response({"db": db, "created": created}, status_code=status)


@app.delete("/id/{db}")
async def delete_db(db: str, request: Request):
    payload = await _json_or_empty(request)
    action = payload.get("action")
    if "p" in payload:
        if action is not None and action != "delete_triple":
            raise APIError(400, "invalid_payload", "Gebruik geen drop action bij triple-delete")
        if not _db_exists(db):
            raise APIError(404, "db_not_found", f"Database `{db}` bestaat niet")
        p = payload.get("p")
        if not p:
            raise APIError(400, "invalid_payload", "Payload vereist `p` voor triple-delete")
        o = payload.get("o")
        store = _open_store(db)
        deleted = _delete_triples(
            store,
            subject=_db_iri(db),
            predicate=p,
            obj=o,
            graph_iri=None,
        )
        return _pretty_json_response(
            {"db": db, "deleted_triples": deleted, "mode": "triple_delete"},
            status_code=200,
        )
    if action != "drop_db":
        raise APIError(
            400,
            "explicit_action_required",
            "Voor database verwijderen is expliciet `action: \"drop_db\"` vereist",
        )
    cascade = payload.get("cascade", True)
    if not isinstance(cascade, bool):
        raise APIError(400, "invalid_payload", "`cascade` moet boolean zijn")
    if not cascade:
        raise APIError(
            400,
            "cascade_required",
            "Alleen cascade delete wordt ondersteund voor databases",
        )
    _delete_db(db)
    return _pretty_json_response({"db": db, "deleted": True, "cascade": True}, status_code=200)


@app.get("/id/{db}/{graph}")
def get_graph(
    db: str,
    graph: str,
    request: Request,
    resolve: str | None = Query(None, description="JSON array met predicate IRIs"),
    resolve_depth: int = Query(1, ge=1, le=10),
    resolve_direction: str = Query("out", pattern="^(out|in|both)$"),
    resolve_limit: int = Query(1000, ge=1, le=50000),
    resolve_include_root: bool = Query(True),
):
    _apply_get_test_writes(request, db=db, graph=graph)
    if not _db_exists(db):
        raise APIError(404, "db_not_found", f"Database `{db}` bestaat niet")

    store = _open_store(db)
    if _incoming_requested(request):
        return _incoming_response_for_subject(request=request, subject_uri=_graph_iri(db, graph))
    g = _named_graph(store, db, graph)
    if len(g) == 0:
        raise APIError(404, "graph_not_found", f"Graph `{graph}` bestaat niet of bevat geen triples")

    resolve_predicates = _parse_resolve_predicates_param(resolve)
    if resolve_predicates:
        g = _resolve_graph_neighbors(
            store=store,
            base_graph=g,
            predicates=resolve_predicates,
            direction=resolve_direction,
            depth=resolve_depth,
            limit=resolve_limit,
            include_root=resolve_include_root,
        )

    if resolve is None:
        redirect = _viewer_redirect_if_needed(request)
        if redirect is not None:
            return redirect
    response_format = _determine_format(request)
    body, media_type = _serialize_graph(g, response_format)
    return Response(content=body, media_type=media_type)


@app.post("/id/{db}/{graph}")
async def post_graph(db: str, graph: str, request: Request):
    payload = await _json_or_empty(request)
    p = payload.get("p")
    o = payload.get("o")

    if p or o is not None:
        if not p or o is None:
            raise APIError(400, "invalid_payload", "Payload vereist minimaal `p` en `o`")
        db_created = not _db_exists(db)
        store = _ensure_store(db)
        _add_db_type(store, db)
        if db_created:
            _link_db_system(store, db)
        existed = _graph_exists(store, db, graph)
        if not existed:
            _insert_triple(
                store,
                subject=_graph_iri(db, graph),
                predicate=RDFS_LABEL_PREDICATE,
                obj={"type": "literal", "value": graph},
                graph_iri=_graph_iri(db, graph),
            )
            _add_graph_type(store, db, graph)
            _link_graph_db(store, db, graph)
            _link_db_graph(store, db, graph)
        _insert_triple(
            store,
            subject=_graph_iri(db, graph),
            predicate=p,
            obj=o,
            graph_iri=_graph_iri(db, graph),
        )
        return _pretty_json_response(
            {
                "db": db,
                "graph": graph,
                "db_created": db_created,
                "graph_created": not existed,
                "triple_created": True,
            },
            status_code=201,
        )

    label = payload.get("label", graph)

    db_created = not _db_exists(db)
    store = _ensure_store(db)
    _add_db_type(store, db)
    if db_created:
        _link_db_system(store, db)
    existed = _graph_exists(store, db, graph)

    _insert_triple(
        store,
        subject=_graph_iri(db, graph),
        predicate=RDFS_LABEL_PREDICATE,
        obj={"type": "literal", "value": label},
        graph_iri=_graph_iri(db, graph),
    )
    _add_graph_type(store, db, graph)
    _link_graph_db(store, db, graph)
    if not existed:
        _link_db_graph(store, db, graph)

    status = 200 if existed else 201
    return _pretty_json_response(
        {"db": db, "graph": graph, "db_created": db_created, "created": not existed},
        status_code=status,
    )


@app.delete("/id/{db}/{graph}")
async def delete_graph(db: str, graph: str, request: Request):
    if not _db_exists(db):
        raise APIError(404, "db_not_found", f"Database `{db}` bestaat niet")
    payload = await _json_or_empty(request)
    action = payload.get("action")
    if "p" in payload:
        if action is not None and action != "delete_triple":
            raise APIError(400, "invalid_payload", "Gebruik geen drop action bij triple-delete")
        p = payload.get("p")
        if not p:
            raise APIError(400, "invalid_payload", "Payload vereist `p` voor triple-delete")
        o = payload.get("o")
        store = _open_store(db)
        deleted = _delete_triples(
            store,
            subject=_graph_iri(db, graph),
            predicate=p,
            obj=o,
            graph_iri=_graph_iri(db, graph),
        )
        return _pretty_json_response(
            {"db": db, "graph": graph, "deleted_triples": deleted, "mode": "triple_delete"},
            status_code=200,
        )
    if action != "drop_graph":
        raise APIError(
            400,
            "explicit_action_required",
            "Voor graph verwijderen is expliciet `action: \"drop_graph\"` vereist",
        )
    cascade = payload.get("cascade", True)
    if not isinstance(cascade, bool):
        raise APIError(400, "invalid_payload", "`cascade` moet boolean zijn")
    if not cascade:
        raise APIError(400, "cascade_required", "Alleen cascade delete wordt ondersteund voor graphs")

    store = _open_store(db)
    if not _graph_exists(store, db, graph):
        raise APIError(404, "graph_not_found", f"Graph `{graph}` bestaat niet")

    graph_deleted, link_deleted = _delete_graph(store, db, graph)
    return _pretty_json_response(
        {
            "db": db,
            "graph": graph,
            "deleted": True,
            "cascade": True,
            "deleted_graph_triples": graph_deleted,
            "deleted_db_links": link_deleted,
        },
        status_code=200,
    )


@app.get("/id/{db}/{graph}/{resource:path}")
def get_resource(
    db: str,
    graph: str,
    resource: str,
    request: Request,
    resolve: str | None = Query(None, description="JSON array met predicate IRIs"),
    resolve_depth: int = Query(1, ge=1, le=10),
    resolve_direction: str = Query("out", pattern="^(out|in|both)$"),
    resolve_limit: int = Query(1000, ge=1, le=50000),
    resolve_include_root: bool = Query(True),
):
    _apply_get_test_writes(request, db=db, graph=graph, resource=resource)
    if not _db_exists(db):
        raise APIError(404, "db_not_found", f"Database `{db}` bestaat niet")

    store = _open_store(db)
    if _incoming_requested(request):
        return _incoming_response_for_subject(
            request=request,
            subject_uri=_resource_iri(db, graph, resource),
        )
    g = _resource_graph(store, db, graph, resource)
    if len(g) == 0:
        raise APIError(404, "resource_not_found", f"Resource `{resource}` niet gevonden")

    resolve_predicates = _parse_resolve_predicates_param(resolve)
    if resolve_predicates:
        g = _resolve_graph_neighbors(
            store=store,
            base_graph=g,
            predicates=resolve_predicates,
            direction=resolve_direction,
            depth=resolve_depth,
            limit=resolve_limit,
            include_root=resolve_include_root,
        )

    if resolve is None:
        redirect = _viewer_redirect_if_needed(request)
        if redirect is not None:
            return redirect
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
    _add_db_type(store, db)
    if db_created:
        _link_db_system(store, db)
    graph_created = not _graph_exists(store, db, graph)

    _insert_triple(
        store,
        subject=_resource_iri(db, graph, resource),
        predicate=p,
        obj=o,
        graph_iri=_graph_iri(db, graph),
    )
    _add_graph_type(store, db, graph)
    _link_graph_db(store, db, graph)
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


@app.delete("/id/{db}/{graph}/{resource:path}")
async def delete_resource_triples(db: str, graph: str, resource: str, request: Request):
    if not _db_exists(db):
        raise APIError(404, "db_not_found", f"Database `{db}` bestaat niet")
    payload = await _json_or_empty(request)
    p = payload.get("p")
    if not p:
        raise APIError(400, "invalid_payload", "Payload vereist `p`")
    o = payload.get("o")
    store = _open_store(db)
    deleted = _delete_triples(
        store,
        subject=_resource_iri(db, graph, resource),
        predicate=p,
        obj=o,
        graph_iri=_graph_iri(db, graph),
    )
    return _pretty_json_response(
        {
            "db": db,
            "graph": graph,
            "resource": resource,
            "deleted_triples": deleted,
            "mode": "triple_delete",
        },
        status_code=200,
    )


@app.get("/api/relations/incoming")
def api_relations_incoming(
    uri: str = Query(..., description="Target IRI"),
    predicates: str | None = Query(None, description="Comma-separated predicate IRIs"),
    graph: str | None = Query(None, description="Named graph IRI filter"),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    predicate_filter = _parse_predicates_param(predicates)
    items = _collect_incoming_relations(uri=uri, predicate_filter=predicate_filter, graph_filter=graph)
    total = len(items)
    page = items[offset : offset + limit]
    return _pretty_json_response(
        {
            "uri": uri,
            "offset": offset,
            "limit": limit,
            "total": total,
            "items": page,
        }
    )


@app.get("/api/relations/neighbors")
def api_relations_neighbors(
    uri: str = Query(..., description="Target IRI"),
    direction: str = Query("both", pattern="^(out|in|both)$"),
    predicates: str | None = Query(None, description="Comma-separated predicate IRIs"),
    graph: str | None = Query(None, description="Named graph IRI filter"),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    predicate_filter = _parse_predicates_param(predicates)
    items = _collect_neighbor_relations(
        uri=uri,
        direction=direction,
        predicate_filter=predicate_filter,
        graph_filter=graph,
    )
    total = len(items)
    page = items[offset : offset + limit]
    return _pretty_json_response(
        {
            "uri": uri,
            "direction": direction,
            "offset": offset,
            "limit": limit,
            "total": total,
            "items": page,
        }
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}
