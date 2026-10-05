"""Safe, user-friendly dataset loading utilities."""
from __future__ import annotations

from io import BytesIO
import json
from typing import Any, Tuple

import pandas as pd


SUPPORTED_EXTENSIONS = ("csv", "xlsx", "xls", "json")


def load_uploaded_dataset(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """Load an uploaded CSV, Excel, or JSON file and return it with a description.

    CSV decoding tries common real-world encodings before giving the user a clear
    failure.  ``dtype`` is deliberately left to pandas so the profiling layer can
    inspect and improve inferred types consistently across file formats.
    """
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        if extension == "csv":
            return _read_csv_with_encoding_fallback(raw_bytes), "CSV"
        if extension in {"xlsx", "xls"}:
            return pd.read_excel(BytesIO(raw_bytes)), "Excel"
        if extension == "json":
            return _read_json_flexibly(raw_bytes), "JSON"
    except Exception as error:  # Library exceptions vary by file format.
        raise ValueError(f"Could not read this {extension.upper()} file: {error}") from error
    raise ValueError("Unsupported file type. Please upload CSV, Excel, or JSON.")


def _read_csv_with_encoding_fallback(raw_bytes: bytes) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(BytesIO(raw_bytes), encoding=encoding, on_bad_lines="warn")
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            errors.append(f"{encoding}: {error}")
    raise ValueError("CSV encoding or structure was not recognized. " + " | ".join(errors))


def _read_json_flexibly(raw_bytes: bytes) -> pd.DataFrame:
    """Read standard, nested, and lightly malformed JSON datasets.

    A trailing comma is a common export mistake (for example, the comma after the
    last item in an array). It is not valid JSON, but can be repaired safely when
    the comma is outside a quoted string. All other JSON parsing failures retain
    their normal clear error message.
    """
    try:
        frame = pd.read_json(BytesIO(raw_bytes))
    except ValueError as pandas_error:
        text = _decode_json_bytes(raw_bytes)
        repaired_text = _remove_json_trailing_commas(text)
        try:
            payload = json.loads(repaired_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON near line {error.lineno}, column {error.colno}: {error.msg}"
            ) from pandas_error
        frame = _normalize_json_payload(payload)
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    return frame


def _decode_json_bytes(raw_bytes: bytes) -> str:
    """Decode JSON's usual UTF encodings, including files exported with a BOM."""
    encodings = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")
    for encoding in encodings:
        try:
            decoded = raw_bytes.decode(encoding)
            # A successful UTF-16 decode of UTF-8 bytes often contains nulls.
            if "\x00" not in decoded:
                return decoded
        except UnicodeDecodeError:
            continue
    raise ValueError("JSON must be encoded as UTF-8 or UTF-16.")


def _remove_json_trailing_commas(text: str) -> str:
    """Remove commas directly before ``]`` or ``}``, preserving quoted content."""
    cleaned: list[str] = []
    in_string = False
    escaped = False
    length = len(text)
    for index, character in enumerate(text):
        if in_string:
            cleaned.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            cleaned.append(character)
            continue
        if character == ",":
            next_index = index + 1
            while next_index < length and text[next_index].isspace():
                next_index += 1
            if next_index < length and text[next_index] in "]}":
                continue
        cleaned.append(character)
    return "".join(cleaned)


def _normalize_json_payload(payload: Any) -> pd.DataFrame:
    """Turn common JSON record containers into a tabular dataframe."""
    if isinstance(payload, list):
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "results"):
            if isinstance(payload.get(key), list):
                return pd.json_normalize(payload[key])
        return pd.json_normalize(payload)
    raise ValueError("JSON must contain an array of records or an object with record fields.")
