#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from pyoxigraph import Store
from rdflib import BNode, Graph, Literal as RdfLiteral, URIRef


DEFAULT_GRAPH_KEY = "__default__"
DEFAULT_GRAPH_FILENAME = "default.ttl"


def term_to_rdflib(term: Any) -> Any:
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


def graph_key_from_quad_graph_name(graph_name: Any) -> str:
    cls_name = graph_name.__class__.__name__
    if cls_name == "DefaultGraph":
        return DEFAULT_GRAPH_KEY
    if cls_name == "NamedNode":
        return getattr(graph_name, "value", str(graph_name).strip("<>"))
    return str(graph_name)


def safe_filename(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    value = value.strip("._")
    return value or "graph"


def graph_filename(graph_key: str, *, db_name: str, base_public: str) -> str:
    if graph_key == DEFAULT_GRAPH_KEY:
        return DEFAULT_GRAPH_FILENAME
    prefix = f"{base_public.rstrip('/')}/id/{db_name}/"
    if graph_key.startswith(prefix):
        logical_name = graph_key[len(prefix) :]
    else:
        logical_name = graph_key
    return f"{safe_filename(logical_name)}.ttl"


def export_db(store_path: Path, output_root: Path, base_public: str) -> tuple[int, int]:
    db_name = store_path.name
    out_dir = output_root / db_name
    out_dir.mkdir(parents=True, exist_ok=True)

    store = Store(str(store_path))
    graphs: dict[str, Graph] = {}

    for quad in store.quads_for_pattern(None, None, None, None):
        key = graph_key_from_quad_graph_name(quad.graph_name)
        bucket = graphs.get(key)
        if bucket is None:
            bucket = Graph()
            bucket.bind("def", URIRef(f"{base_public.rstrip('/')}/def/"), override=True)
            bucket.bind("sdo", URIRef("https://schema.org/"), override=True)
            graphs[key] = bucket
        bucket.add(
            (
                term_to_rdflib(quad.subject),
                term_to_rdflib(quad.predicate),
                term_to_rdflib(quad.object),
            )
        )

    files_written = 0
    triples_total = 0
    for key, graph in graphs.items():
        filename = graph_filename(key, db_name=db_name, base_public=base_public)
        path = out_dir / filename
        path.write_text(graph.serialize(format="turtle"), encoding="utf-8")
        files_written += 1
        triples_total += len(graph)
    return files_written, triples_total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export all graphs (default + named) from all databases to Turtle files."
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "data"),
        help="Path to the Oxigraph data directory (default: env DATA_DIR or ./data)",
    )
    parser.add_argument(
        "--output-dir",
        default="exports",
        help="Path for exported files (default: ./exports)",
    )
    parser.add_argument(
        "--base-public",
        default=os.getenv("PUBLIC_BASE", "https://kvan-todb.hualab.nl"),
        help="Public base URL used to derive graph file names",
    )
    parser.add_argument(
        "--include-system-db",
        action="store_true",
        help="Include internal __system__ store in export",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"data directory not found: {data_dir}")
        return 1

    db_paths = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    if not args.include_system_db:
        db_paths = [p for p in db_paths if p.name != "__system__"]

    db_count = 0
    file_count = 0
    triple_count = 0
    for db_path in db_paths:
        try:
            files_written, triples_written = export_db(db_path, output_dir, args.base_public)
        except OSError as exc:
            print(f"{db_path.name}: skipped (could not open store: {exc})")
            continue
        db_count += 1
        file_count += files_written
        triple_count += triples_written
        print(f"{db_path.name}: {files_written} file(s), {triples_written} triple(s)")

    print(f"done: {db_count} database(s), {file_count} file(s), {triple_count} triple(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
