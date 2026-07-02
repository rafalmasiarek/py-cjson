from __future__ import annotations

"""
cjson.py

Stdlib-only compact JSON helpers.

This module contains two compatible formats:

1. cjson batch format (legacy/current API)
   list[dict] -> {"$c": [...], "$r": [[...], ...]}

2. cjsonl append-only format (new optimized storage API)
   one JSON value per line:
     header object -> data rows as arrays -> optional footer object

The cjsonl format is designed to replace classic JSONL buffers and avoid the
expensive pipeline:

    JSONL buffer -> list[dict] -> cjson.dumps(...) -> gzip

Instead, write directly to .cjsonl and gzip it streamingly at rollover.
"""

import gzip
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, TextIO


# =============================================================================
# Legacy batch cjson format
# =============================================================================

C = "$c"
R = "$r"
I = "$i"

BY = "b"
SORT = "s"
MINMAX = "x"

T = "$t"
M = "$m"


class _Missing:
    pass


_MISSING = _Missing()


def _index_key(value: Any) -> str:
    if value is None:
        return "n:null"
    if value is True:
        return "b:true"
    if value is False:
        return "b:false"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"i:{value}"
    if isinstance(value, float):
        return f"f:{value!r}"
    if isinstance(value, str):
        return f"s:{value}"
    return "j:" + json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _decode_index_key(key: str) -> Any:
    prefix, _, value = key.partition(":")
    if prefix == "n":
        return None
    if prefix == "b":
        return value == "true"
    if prefix == "i":
        return int(value)
    if prefix == "f":
        return float(value)
    if prefix == "s":
        return value
    if prefix == "j":
        return json.loads(value)
    return key


def _is_reserved_key(key: Any) -> bool:
    return isinstance(key, str) and key in {C, R, I, T, M}


def _is_plain_record(obj: Any) -> bool:
    return isinstance(obj, dict) and not any(_is_reserved_key(k) for k in obj.keys())


def _is_record_list(obj: Any) -> bool:
    return isinstance(obj, list) and len(obj) > 0 and all(_is_plain_record(item) for item in obj)


def _is_packed_table(obj: Any) -> bool:
    return isinstance(obj, dict) and C in obj and R in obj and isinstance(obj[C], list) and isinstance(obj[R], list)


def _is_missing_marker(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get(T) == M


def _collect_columns(records: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    columns: list[str] = []
    for record in records:
        for raw_key in record.keys():
            key = str(raw_key)
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns


def _column_positions(columns: list[str]) -> dict[str, int]:
    return {column: i for i, column in enumerate(columns)}


def _build_indexes(
    columns: list[str],
    rows: list[list[Any]],
    *,
    index_by: Iterable[str] = (),
    index_sort: Iterable[str] = (),
    index_minmax: Iterable[str] = (),
) -> dict[str, Any]:
    positions = _column_positions(columns)
    indexes: dict[str, Any] = {}

    by_indexes: dict[str, dict[str, list[int]]] = {}
    for column in index_by:
        if column not in positions:
            continue
        pos = positions[column]
        pos_key = str(pos)
        value_to_rows: dict[str, list[int]] = {}
        for row_id, row in enumerate(rows):
            if pos >= len(row):
                continue
            value = row[pos]
            if _is_missing_marker(value):
                continue
            value_to_rows.setdefault(_index_key(value), []).append(row_id)
        by_indexes[pos_key] = value_to_rows
    if by_indexes:
        indexes[BY] = by_indexes

    sort_indexes: dict[str, list[int]] = {}
    for column in index_sort:
        if column not in positions:
            continue
        pos = positions[column]
        pos_key = str(pos)

        def sort_key(row_id: int) -> tuple[int, Any]:
            value = rows[row_id][pos]
            if _is_missing_marker(value):
                return (1, None)
            return (0, value)

        try:
            ordered = sorted(range(len(rows)), key=sort_key)
        except TypeError:
            ordered = sorted(range(len(rows)), key=lambda row_id: repr(rows[row_id][pos]))
        sort_indexes[pos_key] = ordered
    if sort_indexes:
        indexes[SORT] = sort_indexes

    minmax_indexes: dict[str, list[Any]] = {}
    for column in index_minmax:
        if column not in positions:
            continue
        pos = positions[column]
        values = [row[pos] for row in rows if pos < len(row) and not _is_missing_marker(row[pos])]
        if not values:
            continue
        try:
            minmax_indexes[str(pos)] = [min(values), max(values)]
        except TypeError:
            pass
    if minmax_indexes:
        indexes[MINMAX] = minmax_indexes

    return indexes


def pack(
    obj: Any,
    *,
    recursive: bool = True,
    index_by: Iterable[str] = (),
    index_sort: Iterable[str] = (),
    index_minmax: Iterable[str] = (),
) -> Any:
    """Pack list[dict] into legacy cjson {"$c","$r"} format."""
    if _is_record_list(obj):
        records: list[dict[str, Any]] = obj
        columns = _collect_columns(records)
        rows: list[list[Any]] = []
        for record in records:
            row: list[Any] = []
            for column in columns:
                if column in record:
                    value = record[column]
                    if recursive:
                        value = pack(
                            value,
                            recursive=recursive,
                            index_by=index_by,
                            index_sort=index_sort,
                            index_minmax=index_minmax,
                        )
                    row.append(value)
                else:
                    row.append({T: M})
            rows.append(row)

        packed: dict[str, Any] = {C: columns, R: rows}
        indexes = _build_indexes(
            columns,
            rows,
            index_by=index_by,
            index_sort=index_sort,
            index_minmax=index_minmax,
        )
        if indexes:
            packed[I] = indexes
        return packed

    if recursive and isinstance(obj, dict):
        return {
            str(k): pack(v, recursive=recursive, index_by=index_by, index_sort=index_sort, index_minmax=index_minmax)
            for k, v in obj.items()
        }
    if recursive and isinstance(obj, list):
        return [pack(v, recursive=recursive, index_by=index_by, index_sort=index_sort, index_minmax=index_minmax) for v in obj]
    return obj


def unpack(obj: Any) -> Any:
    """Unpack legacy cjson data into normal Python values."""
    if _is_missing_marker(obj):
        return _MISSING

    if _is_packed_table(obj):
        columns = obj[C]
        records: list[dict[str, Any]] = []
        for row in obj[R]:
            record: dict[str, Any] = {}
            for column, value in zip(columns, row):
                unpacked = unpack(value)
                if unpacked is not _MISSING:
                    record[column] = unpacked
            records.append(record)
        return records

    if isinstance(obj, dict):
        return {k: unpack(v) for k, v in obj.items() if k != I}
    if isinstance(obj, list):
        return [unpack(v) for v in obj]
    return obj


def dumps(
    obj: Any,
    *,
    recursive: bool = True,
    compact: bool = True,
    index_by: Iterable[str] = (),
    index_sort: Iterable[str] = (),
    index_minmax: Iterable[str] = (),
    **json_kwargs: Any,
) -> str:
    """json.dumps-like API: pack first, then serialize."""
    packed = pack(
        obj,
        recursive=recursive,
        index_by=index_by,
        index_sort=index_sort,
        index_minmax=index_minmax,
    )
    if compact:
        json_kwargs.setdefault("separators", (",", ":"))
    json_kwargs.setdefault("ensure_ascii", False)
    return json.dumps(packed, **json_kwargs)


def loads(s: str | bytes | bytearray, **json_kwargs: Any) -> Any:
    """json.loads-like API: parse first, then unpack."""
    return unpack(json.loads(s, **json_kwargs))


def dump(
    obj: Any,
    fp: TextIO,
    *,
    recursive: bool = True,
    compact: bool = True,
    index_by: Iterable[str] = (),
    index_sort: Iterable[str] = (),
    index_minmax: Iterable[str] = (),
    **json_kwargs: Any,
) -> None:
    fp.write(
        dumps(
            obj,
            recursive=recursive,
            compact=compact,
            index_by=index_by,
            index_sort=index_sort,
            index_minmax=index_minmax,
            **json_kwargs,
        )
    )


def load(fp: TextIO, **json_kwargs: Any) -> Any:
    return loads(fp.read(), **json_kwargs)


def is_packed_table(obj: Any) -> bool:
    return _is_packed_table(obj)


def column_index(packed: dict[str, Any], column: str) -> int:
    return packed[C].index(column)


def get_column(packed: dict[str, Any], column: str) -> list[Any]:
    pos = column_index(packed, column)
    return [unpack(row[pos]) for row in packed[R] if pos < len(row) and not _is_missing_marker(row[pos])]


def row_to_record(packed: dict[str, Any], row: list[Any]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for column, value in zip(packed[C], row):
        unpacked = unpack(value)
        if unpacked is not _MISSING:
            record[column] = unpacked
    return record


def iter_records(packed: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Iterate records from legacy packed table."""
    for row in packed[R]:
        yield row_to_record(packed, row)


def get_row_ids_by_value(packed: dict[str, Any], column: str, value: Any) -> list[int]:
    pos = column_index(packed, column)
    pos_key = str(pos)
    value_key = _index_key(value)
    indexed = packed.get(I, {}).get(BY, {}).get(pos_key, {}).get(value_key)
    if indexed is not None:
        return list(indexed)

    out: list[int] = []
    for row_id, row in enumerate(packed[R]):
        if pos >= len(row):
            continue
        cell = row[pos]
        if _is_missing_marker(cell):
            continue
        if unpack(cell) == value:
            out.append(row_id)
    return out


def get_rows_by_value(packed: dict[str, Any], column: str, value: Any) -> list[list[Any]]:
    rows = packed[R]
    return [rows[row_id] for row_id in get_row_ids_by_value(packed, column, value)]


def get_records_by_value(packed: dict[str, Any], column: str, value: Any) -> list[dict[str, Any]]:
    return [row_to_record(packed, row) for row in get_rows_by_value(packed, column, value)]


def maybe_range_exists(
    packed: dict[str, Any],
    column: str,
    *,
    min_value: Any | None = None,
    max_value: Any | None = None,
) -> bool:
    pos = column_index(packed, column)
    bounds = packed.get(I, {}).get(MINMAX, {}).get(str(pos))
    if not bounds:
        return True
    table_min, table_max = bounds
    if min_value is not None and table_max < min_value:
        return False
    if max_value is not None and table_min > max_value:
        return False
    return True


def get_records_in_range(
    packed: dict[str, Any],
    column: str,
    *,
    min_value: Any | None = None,
    max_value: Any | None = None,
) -> list[dict[str, Any]]:
    if not maybe_range_exists(packed, column, min_value=min_value, max_value=max_value):
        return []
    pos = column_index(packed, column)
    out: list[dict[str, Any]] = []
    for row in packed[R]:
        if pos >= len(row):
            continue
        value = row[pos]
        if _is_missing_marker(value):
            continue
        value = unpack(value)
        if min_value is not None and value < min_value:
            continue
        if max_value is not None and value > max_value:
            continue
        out.append(row_to_record(packed, row))
    return out


def get_sorted_row_ids(packed: dict[str, Any], column: str, *, reverse: bool = False) -> list[int]:
    pos = column_index(packed, column)
    indexed = packed.get(I, {}).get(SORT, {}).get(str(pos))
    if indexed is not None:
        row_ids = list(indexed)
    else:
        rows = packed[R]

        def sort_key(row_id: int) -> tuple[int, Any]:
            value = rows[row_id][pos]
            if _is_missing_marker(value):
                return (1, None)
            return (0, unpack(value))

        try:
            row_ids = sorted(range(len(rows)), key=sort_key)
        except TypeError:
            row_ids = sorted(range(len(rows)), key=lambda row_id: repr(rows[row_id][pos]))
    if reverse:
        row_ids.reverse()
    return row_ids


def iter_sorted_records(packed: dict[str, Any], column: str, *, reverse: bool = False) -> Iterator[dict[str, Any]]:
    rows = packed[R]
    for row_id in get_sorted_row_ids(packed, column, reverse=reverse):
        yield row_to_record(packed, rows[row_id])


def loads_packed(s: str | bytes | bytearray, **json_kwargs: Any) -> Any:
    return json.loads(s, **json_kwargs)


def dumps_packed(obj: Any, **json_kwargs: Any) -> str:
    json_kwargs.setdefault("separators", (",", ":"))
    json_kwargs.setdefault("ensure_ascii", False)
    return json.dumps(obj, **json_kwargs)


def to_normal(obj: Any) -> Any:
    return unpack(obj)


def dumps_normal(obj: Any, *, compact: bool = True, **json_kwargs: Any) -> str:
    normal = to_normal(obj)
    if compact:
        json_kwargs.setdefault("separators", (",", ":"))
    json_kwargs.setdefault("ensure_ascii", False)
    return json.dumps(normal, **json_kwargs)


def loads_normal(s: str | bytes | bytearray, *, compact_output: bool | None = None, **json_kwargs: Any) -> Any:
    del compact_output
    return to_normal(json.loads(s, **json_kwargs))


def dump_normal(obj: Any, fp: TextIO, *, compact: bool = True, **json_kwargs: Any) -> None:
    fp.write(dumps_normal(obj, compact=compact, **json_kwargs))


def load_normal(fp: TextIO, **json_kwargs: Any) -> Any:
    return to_normal(json.load(fp, **json_kwargs))


def convert_file_to_normal_json(
    src_path: str,
    dst_path: str,
    *,
    compact: bool = True,
    encoding: str = "utf-8",
    **json_kwargs: Any,
) -> None:
    with open(src_path, "r", encoding=encoding) as src:
        normal = load_normal(src)
    if compact:
        json_kwargs.setdefault("separators", (",", ":"))
    json_kwargs.setdefault("ensure_ascii", False)
    with open(dst_path, "w", encoding=encoding) as dst:
        json.dump(normal, dst, **json_kwargs)


def convert_file_to_pretty_normal_json(
    src_path: str,
    dst_path: str,
    *,
    indent: int = 2,
    encoding: str = "utf-8",
    **json_kwargs: Any,
) -> None:
    json_kwargs.setdefault("indent", indent)
    json_kwargs.setdefault("ensure_ascii", False)
    with open(src_path, "r", encoding=encoding) as src:
        normal = load_normal(src)
    with open(dst_path, "w", encoding=encoding) as dst:
        json.dump(normal, dst, **json_kwargs)


def is_cjson(obj: Any) -> bool:
    if _is_packed_table(obj):
        return True
    if isinstance(obj, dict):
        return any(is_cjson(v) for v in obj.values())
    if isinstance(obj, list):
        return any(is_cjson(v) for v in obj)
    return False


def normalize_json_string(s: str | bytes | bytearray, *, compact: bool = True, **json_kwargs: Any) -> str:
    normal = to_normal(json.loads(s))
    if compact:
        json_kwargs.setdefault("separators", (",", ":"))
    json_kwargs.setdefault("ensure_ascii", False)
    return json.dumps(normal, **json_kwargs)


# =============================================================================
# New cjsonl append-only format
# =============================================================================

# Official cjsonl v1 keys are intentionally short.
# v = cjsonl version
# f = optional custom codec id
# s = optional schema id
# c = columns, when no schema id is used
# b = base values by column index, for delta encoding
# d = defaults by column index, for short default markers
# e = encodings metadata when schema is not supplied
# $ = footer/seal marker
# n = row count
# x = min/max by column index, stored in encoded space

V = "v"
F = "f"
S = "s"
B = "b"
D = "d"
E = "e"
FOOTER = "$"
N = "n"
X = "x"


class CjsonlError(ValueError):
    """Base error for cjsonl parsing/writing failures."""


@dataclass(slots=True)
class Schema:
    """
    Optional schema/profile for compact cjsonl files.

    id:
        Short int/str written as "s" in the file. Prefer small ints for storage.
    columns:
        Stable column order. Data rows are arrays in exactly this order.
    bool_int:
        Columns encoded as 1/0 and decoded back to bool.
    bases:
        Base values by column name. Encoded value = raw value - base.
        Use this for timestamps, counters, monotonic ids, etc.
    defaults:
        Default values by column name. If a record is missing the column, or has
        the same value, the default marker is written instead.
    default_markers:
        Marker values by column name. If omitted, 0 is used. Pick markers that
        cannot be valid non-default data for that column.
    name:
        Human-readable name kept in Python registry, not written to compact file.
    """

    id: int | str | None
    columns: list[str]
    bool_int: set[str] = field(default_factory=set)
    bases: dict[str, Any] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    default_markers: dict[str, Any] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self) -> None:
        self.columns = [str(c) for c in self.columns]
        self.bool_int = {str(c) for c in self.bool_int}
        self.bases = {str(k): v for k, v in self.bases.items()}
        self.defaults = {str(k): v for k, v in self.defaults.items()}
        self.default_markers = {str(k): v for k, v in self.default_markers.items()}

    @classmethod
    def from_obj(cls, obj: "SchemaLike") -> "Schema":
        if isinstance(obj, Schema):
            return obj
        if isinstance(obj, (list, tuple)):
            return cls(id=None, columns=[str(c) for c in obj])
        if isinstance(obj, Mapping):
            cols = obj.get("columns") or obj.get("c")
            if cols is None:
                raise CjsonlError("schema mapping must contain 'columns' or 'c'")
            return cls(
                id=obj.get("id", obj.get("s")),
                name=obj.get("name"),
                columns=[str(c) for c in cols],
                bool_int=set(obj.get("bool_int", obj.get("bool", []))),
                bases=dict(obj.get("bases", obj.get("b", {}))),
                defaults=dict(obj.get("defaults", obj.get("d", {}))),
                default_markers=dict(obj.get("default_markers", obj.get("markers", {}))),
            )
        raise TypeError(f"unsupported schema type: {type(obj)!r}")


SchemaLike = Schema | Mapping[str, Any] | list[str] | tuple[str, ...]
SchemasLike = Mapping[int | str, SchemaLike]


@dataclass(slots=True)
class CjsonlMeta:
    version: int = 1
    codec: str | None = None
    schema_id: int | str | None = None
    columns: list[str] = field(default_factory=list)
    bases: dict[int, Any] = field(default_factory=dict)
    defaults: dict[int, Any] = field(default_factory=dict)
    default_markers: dict[int, Any] = field(default_factory=dict)
    bool_positions: set[int] = field(default_factory=set)
    sealed: bool = False
    count: int = 0
    minmax: dict[int, list[Any]] = field(default_factory=dict)
    raw_headers: list[dict[str, Any]] = field(default_factory=list)
    raw_footer: dict[str, Any] | None = None


def _json_dumps_line(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def _schema_registry_get(schemas: SchemasLike | None, schema_id: int | str | None) -> Schema | None:
    if schema_id is None:
        return None
    if schemas is None:
        return None
    if schema_id in schemas:
        return Schema.from_obj(schemas[schema_id])
    # JSON can turn numeric object keys into strings in user registries.
    alt = str(schema_id)
    if alt in schemas:  # type: ignore[operator]
        return Schema.from_obj(schemas[alt])  # type: ignore[index]
    raise CjsonlError(f"unknown cjsonl schema id: {schema_id!r}")


def _positions(columns: list[str]) -> dict[str, int]:
    return {c: i for i, c in enumerate(columns)}


def _normalize_index_dict(obj: Mapping[Any, Any] | None) -> dict[int, Any]:
    if not obj:
        return {}
    out: dict[int, Any] = {}
    for k, v in obj.items():
        out[int(k)] = v
    return out


def _schema_to_context(schema: Schema | None, header: Mapping[str, Any] | None = None) -> CjsonlMeta:
    header = header or {}
    schema_id = header.get(S) if S in header else (schema.id if schema else None)
    columns = list(schema.columns) if schema else [str(c) for c in header.get(C, [])]
    pos = _positions(columns)

    bases: dict[int, Any] = {}
    defaults: dict[int, Any] = {}
    default_markers: dict[int, Any] = {}
    bool_positions: set[int] = set()

    if schema:
        for name, base in schema.bases.items():
            if name in pos:
                bases[pos[name]] = base
        for name, default in schema.defaults.items():
            if name in pos:
                p = pos[name]
                defaults[p] = default
                default_markers[p] = schema.default_markers.get(name, 0)
        for name in schema.bool_int:
            if name in pos:
                bool_positions.add(pos[name])

    # Header can override/add compact metadata, useful for no-schema files and
    # segment-specific bases.
    for p, base in _normalize_index_dict(header.get(B)).items():
        bases[p] = base
    for p, default in _normalize_index_dict(header.get(D)).items():
        defaults[p] = default
        default_markers.setdefault(p, 0)
    enc = header.get(E, {}) or {}
    for p in [int(x) for x in enc.get("bool", [])]:
        bool_positions.add(p)
    for k, v in _normalize_index_dict(enc.get("m", {})).items():
        default_markers[k] = v

    return CjsonlMeta(
        version=int(header.get(V, 1)),
        codec=header.get(F),
        schema_id=schema_id,
        columns=columns,
        bases=bases,
        defaults=defaults,
        default_markers=default_markers,
        bool_positions=bool_positions,
        raw_headers=[dict(header)] if header else [],
    )


def _make_header(schema: Schema | None, columns: list[str], *, codec: str | None = None) -> dict[str, Any]:
    header: dict[str, Any] = {V: 1}
    if codec:
        header[F] = codec
    if schema and schema.id is not None:
        header[S] = schema.id
    else:
        header[C] = columns

    pos = _positions(columns)

    bases: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    markers: dict[str, Any] = {}
    bools: list[int] = []

    if schema:
        for name, base in schema.bases.items():
            if name in pos:
                bases[str(pos[name])] = base
        # If schema id is present, defaults/bool metadata is known via registry.
        # If no id, write the metadata so the file remains self-describing.
        if schema.id is None:
            for name, default in schema.defaults.items():
                if name in pos:
                    p = pos[name]
                    defaults[str(p)] = default
                    marker = schema.default_markers.get(name, 0)
                    if marker != 0:
                        markers[str(p)] = marker
            bools = [pos[name] for name in schema.bool_int if name in pos]
    if bases:
        header[B] = bases
    if defaults:
        header[D] = defaults
    enc: dict[str, Any] = {}
    if bools:
        enc["bool"] = sorted(bools)
    if markers:
        enc["m"] = markers
    if enc:
        header[E] = enc
    return header


def _encode_cell(value: Any, pos: int, meta: CjsonlMeta, *, missing: bool = False) -> Any:
    if missing and pos in meta.defaults:
        return meta.default_markers.get(pos, 0)
    if pos in meta.defaults and value == meta.defaults[pos]:
        return meta.default_markers.get(pos, 0)
    if pos in meta.bool_positions:
        if value is None:
            return None
        return 1 if bool(value) else 0
    if pos in meta.bases and value is not None:
        try:
            return value - meta.bases[pos]
        except TypeError:
            return value
    return value


def _decode_cell(value: Any, pos: int, meta: CjsonlMeta) -> Any:
    if pos in meta.defaults and value == meta.default_markers.get(pos, 0):
        return meta.defaults[pos]
    if pos in meta.bool_positions and value is not None:
        return bool(value)
    if pos in meta.bases and value is not None:
        try:
            return meta.bases[pos] + value
        except TypeError:
            return value
    return value


def _encode_record(record: Mapping[str, Any], meta: CjsonlMeta) -> list[Any]:
    row: list[Any] = []
    for pos, column in enumerate(meta.columns):
        if column in record:
            row.append(_encode_cell(record[column], pos, meta, missing=False))
        else:
            row.append(_encode_cell(None, pos, meta, missing=True))
    return row


def _decode_row(row: list[Any], meta: CjsonlMeta) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for pos, column in enumerate(meta.columns):
        if pos < len(row):
            record[column] = _decode_cell(row[pos], pos, meta)
        elif pos in meta.defaults:
            record[column] = meta.defaults[pos]
        else:
            record[column] = None
    return record


def _track_minmax(meta: CjsonlMeta, row: list[Any]) -> None:
    for pos, value in enumerate(row):
        if value is None:
            continue
        if pos in meta.defaults and value == meta.default_markers.get(pos, 0):
            continue
        # Skip string/dict/list minmax to keep footer small and comparable.
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        bounds = meta.minmax.get(pos)
        if bounds is None:
            meta.minmax[pos] = [value, value]
        else:
            if value < bounds[0]:
                bounds[0] = value
            if value > bounds[1]:
                bounds[1] = value


class V1Codec:
    """Official dense-only cjsonl v1 codec."""

    id: str | None = None

    def make_header(self, schema: Schema | None, columns: list[str]) -> dict[str, Any]:
        return _make_header(schema, columns, codec=self.id)

    def context_from_header(self, header: Mapping[str, Any], schemas: SchemasLike | None = None) -> CjsonlMeta:
        schema = _schema_registry_get(schemas, header.get(S))
        return _schema_to_context(schema, header)

    def encode_record(self, record: Mapping[str, Any], meta: CjsonlMeta) -> list[Any]:
        return _encode_record(record, meta)

    def decode_row(self, row: list[Any], meta: CjsonlMeta) -> dict[str, Any]:
        return _decode_row(row, meta)

    def make_footer(self, meta: CjsonlMeta) -> dict[str, Any]:
        footer: dict[str, Any] = {FOOTER: 1, N: meta.count}
        if meta.minmax:
            footer[X] = {str(k): v for k, v in sorted(meta.minmax.items())}
        return footer


class CodecRegistry:
    def __init__(self) -> None:
        self._codecs: dict[str | None, V1Codec] = {None: V1Codec()}

    def register(self, codec_id: str, codec: V1Codec) -> None:
        codec.id = codec_id
        self._codecs[codec_id] = codec

    def get(self, codec_id: str | None) -> V1Codec:
        if codec_id in self._codecs:
            return self._codecs[codec_id]
        raise CjsonlError(f"unknown cjsonl codec: {codec_id!r}")


_CODEC_REGISTRY = CodecRegistry()


def register_codec(codec_id: str, codec: V1Codec) -> None:
    """Register a custom cjsonl codec. Official v1 is used when no codec id is present."""
    _CODEC_REGISTRY.register(codec_id, codec)


class Writer:
    """
    Append-only cjsonl writer.

    The writer writes exactly one header when the target is empty, then writes
    each record as a compact JSON array. Use seal() before rotating/gzipping.
    """

    def __init__(
        self,
        fp: TextIO,
        *,
        schema: SchemaLike | None = None,
        columns: Iterable[str] | None = None,
        codec: V1Codec | None = None,
        append: bool = True,
        close_file: bool = False,
    ) -> None:
        self.fp = fp
        self._close_file = close_file
        self.codec = codec or _CODEC_REGISTRY.get(None)
        self.schema = Schema.from_obj(schema) if schema is not None else None
        if self.schema is not None:
            cols = list(self.schema.columns)
        elif columns is not None:
            cols = [str(c) for c in columns]
        else:
            cols = []
        self.meta = _schema_to_context(self.schema, {})
        self.meta.columns = cols
        self._closed = False
        self._header_written = False
        if append:
            self._header_written = self._looks_nonempty(fp)
        if not self._header_written:
            if not self.meta.columns:
                raise CjsonlError("columns or schema are required when creating a new cjsonl stream")
            self.write_header()

    @staticmethod
    def _looks_nonempty(fp: TextIO) -> bool:
        try:
            pos = fp.tell()
            fp.seek(0, os.SEEK_END)
            end = fp.tell()
            fp.seek(pos, os.SEEK_SET)
            return end > 0
        except Exception:
            return False

    def write_header(self) -> None:
        header = self.codec.make_header(self.schema, self.meta.columns)
        self.fp.write(_json_dumps_line(header))
        self._header_written = True

    def write(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise CjsonlError("cannot write to closed cjsonl writer")
        row = self.codec.encode_record(record, self.meta)
        self.fp.write(_json_dumps_line(row))
        self.meta.count += 1
        _track_minmax(self.meta, row)

    def seal(self) -> None:
        if self._closed:
            return
        footer = self.codec.make_footer(self.meta)
        self.fp.write(_json_dumps_line(footer))
        self.fp.flush()
        self._closed = True

    def close(self) -> None:
        self.fp.flush()
        if self._close_file:
            self.fp.close()

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class Reader:
    """Streaming cjsonl reader. Iterates normal dict records."""

    def __init__(self, fp: TextIO, *, schemas: SchemasLike | None = None) -> None:
        self.fp = fp
        self.schemas = schemas
        self.meta: CjsonlMeta | None = None
        self.codec: V1Codec = _CODEC_REGISTRY.get(None)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for raw_line in self.fp:
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                # JSONL/cjsonl recovery behavior: ignore a torn final line.
                continue

            if isinstance(item, dict):
                if item.get(FOOTER) == 1:
                    if self.meta is not None:
                        self.meta.sealed = True
                        self.meta.raw_footer = item
                        self.meta.count = int(item.get(N, self.meta.count))
                        self.meta.minmax = {int(k): v for k, v in item.get(X, {}).items()}
                    continue

                if V in item or C in item or S in item:
                    self.codec = _CODEC_REGISTRY.get(item.get(F))
                    self.meta = self.codec.context_from_header(item, self.schemas)
                    continue

                # Unknown object line is metadata for a custom/user layer. Skip.
                continue

            if isinstance(item, list):
                if self.meta is None:
                    raise CjsonlError("cjsonl data row encountered before header")
                yield self.codec.decode_row(item, self.meta)


class RowReader:
    """
    Streaming cjsonl row reader. Iterates decoded row arrays and exposes columns.

    Use this for fast scans where building dicts for every row is unnecessary.
    """

    def __init__(self, fp: TextIO, *, schemas: SchemasLike | None = None) -> None:
        self.fp = fp
        self.schemas = schemas
        self.meta: CjsonlMeta | None = None
        self.codec: V1Codec = _CODEC_REGISTRY.get(None)

    def __iter__(self) -> Iterator[list[Any]]:
        for raw_line in self.fp:
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                if item.get(FOOTER) == 1:
                    continue
                if V in item or C in item or S in item:
                    self.codec = _CODEC_REGISTRY.get(item.get(F))
                    self.meta = self.codec.context_from_header(item, self.schemas)
                continue
            if isinstance(item, list):
                if self.meta is None:
                    raise CjsonlError("cjsonl data row encountered before header")
                yield [_decode_cell(v, i, self.meta) for i, v in enumerate(item)]


# -----------------------------------------------------------------------------
# cjsonl file/path convenience API
# -----------------------------------------------------------------------------

def open_writer(
    path: str,
    *,
    schema: SchemaLike | None = None,
    columns: Iterable[str] | None = None,
    encoding: str = "utf-8",
) -> Writer:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fp = open(path, "a+", encoding=encoding)
    return Writer(fp, schema=schema, columns=columns, append=True, close_file=True)


def append(path: str, record: Mapping[str, Any], *, schema: SchemaLike | None = None, columns: Iterable[str] | None = None) -> None:
    """Append one record to a cjsonl file."""
    with open_writer(path, schema=schema, columns=columns) as w:
        w.write(record)


def iter_cjsonl_records(path: str, *, schemas: SchemasLike | None = None, encoding: str = "utf-8") -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding=encoding) as fp:
        yield from Reader(fp, schemas=schemas)


def iter_cjsonl_rows(path: str, *, schemas: SchemasLike | None = None, encoding: str = "utf-8") -> Iterator[list[Any]]:
    with open(path, "r", encoding=encoding) as fp:
        yield from RowReader(fp, schemas=schemas)


def load_cjsonl(path: str, *, schemas: SchemasLike | None = None, encoding: str = "utf-8") -> list[dict[str, Any]]:
    return list(iter_cjsonl_records(path, schemas=schemas, encoding=encoding))


def dump_cjsonl(
    records: Iterable[Mapping[str, Any]],
    path: str,
    *,
    schema: SchemaLike | None = None,
    columns: Iterable[str] | None = None,
    seal: bool = True,
    encoding: str = "utf-8",
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding=encoding) as fp:
        writer = Writer(fp, schema=schema, columns=columns, append=False)
        for record in records:
            writer.write(record)
        if seal:
            writer.seal()
        else:
            writer.close()


def seal(path: str, *, schemas: SchemasLike | None = None, encoding: str = "utf-8") -> CjsonlMeta:
    """
    Append a footer/seal line to an existing cjsonl file.

    This scans the file to compute count/minmax, then appends one compact footer.
    It does not rewrite the header or data rows.
    """
    meta = scan(path, schemas=schemas, encoding=encoding)
    if meta.sealed:
        return meta
    footer = {FOOTER: 1, N: meta.count}
    if meta.minmax:
        footer[X] = {str(k): v for k, v in sorted(meta.minmax.items())}
    with open(path, "a", encoding=encoding) as fp:
        fp.write(_json_dumps_line(footer))
    meta.sealed = True
    meta.raw_footer = footer
    return meta


def scan(path: str, *, schemas: SchemasLike | None = None, encoding: str = "utf-8") -> CjsonlMeta:
    """Scan cjsonl metadata/count/minmax without returning records."""
    meta: CjsonlMeta | None = None
    codec: V1Codec = _CODEC_REGISTRY.get(None)
    with open(path, "r", encoding=encoding) as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                if item.get(FOOTER) == 1:
                    if meta is None:
                        meta = CjsonlMeta()
                    meta.sealed = True
                    meta.raw_footer = item
                    meta.count = int(item.get(N, meta.count))
                    meta.minmax = {int(k): v for k, v in item.get(X, {}).items()}
                    continue
                if V in item or C in item or S in item:
                    codec = _CODEC_REGISTRY.get(item.get(F))
                    meta = codec.context_from_header(item, schemas)
                    continue
                continue
            if isinstance(item, list):
                if meta is None:
                    raise CjsonlError("cjsonl data row encountered before header")
                meta.count += 1
                _track_minmax(meta, item)
    return meta or CjsonlMeta()


def gzip_file(src_path: str, dst_path: str, *, compresslevel: int = 6, remove_src: bool = False) -> None:
    """Stream-compress any file, typically .cjsonl -> .cjsonl.gz."""
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    tmp = dst_path + ".tmp"
    with open(src_path, "rb") as src, gzip.open(tmp, "wb", compresslevel=compresslevel) as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    os.replace(tmp, dst_path)
    if remove_src:
        os.remove(src_path)


def gunzip_file(src_path: str, dst_path: str, *, remove_src: bool = False) -> None:
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    tmp = dst_path + ".tmp"
    with gzip.open(src_path, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    os.replace(tmp, dst_path)
    if remove_src:
        os.remove(src_path)


def iter_cjsonl_gzip_records(path: str, *, schemas: SchemasLike | None = None, encoding: str = "utf-8") -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding=encoding) as fp:
        yield from Reader(fp, schemas=schemas)


def iter_cjsonl_gzip_rows(path: str, *, schemas: SchemasLike | None = None, encoding: str = "utf-8") -> Iterator[list[Any]]:
    with gzip.open(path, "rt", encoding=encoding) as fp:
        yield from RowReader(fp, schemas=schemas)


def load_cjsonl_gzip(path: str, *, schemas: SchemasLike | None = None, encoding: str = "utf-8") -> list[dict[str, Any]]:
    return list(iter_cjsonl_gzip_records(path, schemas=schemas, encoding=encoding))


def iter_records_gzip_bytes(data: bytes, *, schemas: SchemasLike | None = None, encoding: str = "utf-8") -> Iterator[dict[str, Any]]:
    import io

    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as gz:
        with io.TextIOWrapper(gz, encoding=encoding) as fp:
            yield from Reader(fp, schemas=schemas)


def scan_gzip(path: str, *, schemas: SchemasLike | None = None, encoding: str = "utf-8") -> CjsonlMeta:
    meta: CjsonlMeta | None = None
    codec: V1Codec = _CODEC_REGISTRY.get(None)
    with gzip.open(path, "rt", encoding=encoding) as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                if item.get(FOOTER) == 1:
                    if meta is None:
                        meta = CjsonlMeta()
                    meta.sealed = True
                    meta.raw_footer = item
                    meta.count = int(item.get(N, meta.count))
                    meta.minmax = {int(k): v for k, v in item.get(X, {}).items()}
                    continue
                if V in item or C in item or S in item:
                    codec = _CODEC_REGISTRY.get(item.get(F))
                    meta = codec.context_from_header(item, schemas)
                    continue
                continue
            if isinstance(item, list):
                if meta is None:
                    raise CjsonlError("cjsonl data row encountered before header")
                meta.count += 1
                _track_minmax(meta, item)
    return meta or CjsonlMeta()


def convert_jsonl_to_cjsonl(
    src_path: str,
    dst_path: str,
    *,
    schema: SchemaLike | None = None,
    columns: Iterable[str] | None = None,
    seal_output: bool = True,
    encoding: str = "utf-8",
) -> None:
    """Convert classic JSONL dict records into cjsonl."""
    # If neither schema nor columns is supplied, infer columns from first line.
    inferred_columns: list[str] | None = None
    if schema is None and columns is None:
        with open(src_path, "r", encoding=encoding) as src:
            for line in src:
                if not line.strip():
                    continue
                first = json.loads(line)
                if not isinstance(first, dict):
                    raise CjsonlError("classic JSONL input must contain object records")
                inferred_columns = [str(k) for k in first.keys()]
                break
    cols = columns if columns is not None else inferred_columns
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with open(src_path, "r", encoding=encoding) as src, open(dst_path, "w", encoding=encoding) as dst:
        writer = Writer(dst, schema=schema, columns=cols, append=False)
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            writer.write(record)
        if seal_output:
            writer.seal()
        else:
            writer.close()


def convert_cjsonl_to_jsonl(
    src_path: str,
    dst_path: str,
    *,
    schemas: SchemasLike | None = None,
    encoding: str = "utf-8",
) -> None:
    """Convert cjsonl back to classic JSONL."""
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with open(dst_path, "w", encoding=encoding) as dst:
        for record in iter_cjsonl_records(src_path, schemas=schemas, encoding=encoding):
            dst.write(_json_dumps_line(record))


# Backward/ergonomic aliases for cjsonl APIs.
cjsonl_append = append
cjsonl_dump = dump_cjsonl
cjsonl_load = load_cjsonl
cjsonl_iter_records = iter_cjsonl_records
cjsonl_iter_rows = iter_cjsonl_rows
cjsonl_iter_gzip_records = iter_cjsonl_gzip_records
cjsonl_scan = scan
cjsonl_seal = seal
cjsonl_gzip_file = gzip_file


__all__ = [
    # legacy cjson
    "pack", "unpack", "dumps", "loads", "dump", "load",
    "is_packed_table", "column_index", "get_column", "row_to_record", "iter_records",
    "get_row_ids_by_value", "get_rows_by_value", "get_records_by_value",
    "maybe_range_exists", "get_records_in_range", "get_sorted_row_ids", "iter_sorted_records",
    "loads_packed", "dumps_packed", "to_normal", "dumps_normal", "loads_normal",
    "dump_normal", "load_normal", "convert_file_to_normal_json",
    "convert_file_to_pretty_normal_json", "is_cjson", "normalize_json_string",
    # cjsonl
    "CjsonlError", "Schema", "CjsonlMeta", "V1Codec", "register_codec",
    "Writer", "Reader", "RowReader", "open_writer", "append", "dump_cjsonl",
    "load_cjsonl", "iter_cjsonl_records", "iter_cjsonl_rows", "seal", "scan",
    "gzip_file", "gunzip_file", "iter_cjsonl_gzip_records", "iter_cjsonl_gzip_rows",
    "load_cjsonl_gzip", "iter_records_gzip_bytes", "scan_gzip",
    "convert_jsonl_to_cjsonl", "convert_cjsonl_to_jsonl",
    "cjsonl_append", "cjsonl_dump", "cjsonl_load", "cjsonl_iter_records",
    "cjsonl_iter_rows", "cjsonl_iter_gzip_records", "cjsonl_scan", "cjsonl_seal",
    "cjsonl_gzip_file",
]
