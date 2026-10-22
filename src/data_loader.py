"""Dataset loading, validation and type inference.

The loader is intentionally conservative: it never drops rows or invents
values. It normalises column names, caps the number of parsed rows (reporting
the truncation), and promotes obvious numeric/date-looking text columns so the
rest of the application can treat them correctly.
"""

from __future__ import annotations

import csv
import io
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SETTINGS, Settings
from src.profile import CODE_NAME, IDENTIFIER_NAME, NUMERIC_TEXT

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
EXCEL_EXTENSIONS = {".xlsx", ".xls"}
TEXT_EXTENSIONS = {".txt", ".tsv", ".tab"}
# Everything the uploader accepts; delimited text is parsed as CSV.
ALLOWED_EXTENSIONS = SUPPORTED_EXTENSIONS | TEXT_EXTENSIONS
ALLOWED_LABEL = "CSV, TSV, TXT, XLSX, XLS"


class DatasetLoadError(ValueError):
    """Raised when a file cannot be turned into a usable dataset."""


@dataclass
class LoadedDataset:
    dataframe: pd.DataFrame
    filename: str
    truncated: bool = False
    total_rows_in_file: int | None = None
    encoding: str | None = None
    delimiter: str | None = None
    notes: list = field(default_factory=list)

    def meta(self) -> dict:
        return {
            "filename": self.filename,
            "truncated": self.truncated,
            "total_rows_in_file": self.total_rows_in_file,
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "notes": list(self.notes),
        }


@dataclass
class _ParseResult:
    dataframe: pd.DataFrame
    truncated: bool = False
    total_rows: int | None = None
    encoding: str | None = None
    delimiter: str | None = None
    notes: list = field(default_factory=list)


def normalise_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Trim names, fill blanks and de-duplicate column labels in place."""
    used: set[str] = set()
    counters: dict[str, int] = {}
    names = []
    for position, name in enumerate(dataframe.columns):
        label = str(name).replace("\n", " ").strip()
        if not label or label.lower().startswith("unnamed:"):
            label = f"column_{position + 1}"
        if label in used:
            base = label
            suffix = counters.get(base, 0)
            while label in used:
                suffix += 1
                label = f"{base}_{suffix}"
            counters[base] = suffix
        used.add(label)
        names.append(label)
    dataframe.columns = names
    return dataframe


def _sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        # Let pandas decide when the sniffer is not confident.
        return ","


def _code_dtypes(sample_text: str, delimiter: str) -> dict[str, str]:
    """Keep identifier-like columns as text so leading zeros survive parsing.

    pandas infers ``007`` as the integer 7; that silently corrupts zip codes,
    phone numbers and other identifiers. Reading a column as text is cheaper
    than repairing it later, so we detect those columns from the sample.
    """
    try:
        rows = list(csv.reader(io.StringIO(sample_text), delimiter=delimiter))[:60]
    except csv.Error:
        return {}
    if len(rows) < 2:
        return {}

    header = [str(name).strip() for name in rows[0]]
    if len(set(header)) != len(header):
        return {}

    dtypes: dict[str, str] = {}
    for index, name in enumerate(header):
        values = [
            row[index].strip()
            for row in rows[1:]
            if index < len(row) and row[index].strip()
        ]
        if not values:
            continue
        leading_zero = any(len(value) > 1 and value.startswith("0") and value[1:].isdigit() for value in values)
        if CODE_NAME.search(name) or leading_zero:
            dtypes[name] = str
    return dtypes


def _read_csv(buffer: io.BytesIO, settings: Settings) -> _ParseResult:
    head = buffer.getvalue()[:65536]
    encoding = "utf-8-sig"
    try:
        sample_text = head.decode(encoding)
    except UnicodeDecodeError:
        encoding = "latin-1"
        sample_text = head.decode(encoding)

    delimiter = _sniff_delimiter(sample_text)
    if delimiter == "," and sample_text.count(";") > sample_text.count(",") * 3:
        delimiter = ";"

    dtypes = _code_dtypes(sample_text, delimiter)

    buffer.seek(0)
    try:
        frame = pd.read_csv(
            buffer,
            sep=delimiter,
            encoding=encoding,
            nrows=settings.max_rows + 1,
            low_memory=False,
            skip_blank_lines=True,
            dtype=dtypes or None,
        )
    except pd.errors.EmptyDataError as error:
        raise DatasetLoadError("The CSV file is empty.") from error
    except pd.errors.ParserError as error:
        raise DatasetLoadError(f"Could not parse the CSV file: {error}") from error
    except UnicodeDecodeError as error:
        raise DatasetLoadError(
            "The file encoding could not be detected. Re-save it as UTF-8 CSV."
        ) from error

    truncated = len(frame) > settings.max_rows
    total = None
    if truncated:
        frame = frame.iloc[: settings.max_rows].copy()
        # Count the data lines without parsing them (single-line records).
        buffer.seek(0)
        total = _count_extra_rows(buffer, encoding)

    return _ParseResult(
        dataframe=frame,
        truncated=truncated,
        total_rows=total,
        encoding=encoding,
        delimiter=delimiter,
    )


def _count_extra_rows(buffer: io.BytesIO, encoding: str, chunk_size: int = 1 << 20) -> int:
    """Count data lines in a delimited file (header excluded).

    Raw newline counting is approximate for quoted multi-line fields, which is
    why the UI presents the figure as ``~N rows``.
    """
    lines = 0
    buffer.seek(0)
    for chunk in iter(lambda: buffer.read(chunk_size), b""):
        lines += chunk.count(b"\n")
    buffer.seek(0, io.SEEK_END)
    size = buffer.tell()
    if size:
        buffer.seek(-1, io.SEEK_END)
        if buffer.read(1) != b"\n":
            lines += 1  # final line without a trailing newline
    return max(lines - 1, 0)


def _read_excel(buffer: io.BytesIO, extension: str, settings: Settings) -> _ParseResult:
    if extension == ".xls":
        try:
            import xlrd  # noqa: F401
        except ImportError as error:
            raise DatasetLoadError(
                "Legacy .xls files need the optional 'xlrd' package. "
                "Re-save the workbook as .xlsx and upload it again."
            ) from error

    try:
        frame = pd.read_excel(buffer, engine=None if extension == ".xls" else "openpyxl")
    except DatasetLoadError:
        raise
    except Exception as error:  # openpyxl raises many exception types
        raise DatasetLoadError(f"Could not read the Excel workbook: {error}") from error

    truncated = len(frame) > settings.max_rows
    total = int(len(frame))
    if truncated:
        frame = frame.iloc[: settings.max_rows].copy()

    return _ParseResult(dataframe=frame, truncated=truncated, total_rows=total)


def _looks_like_code(values: list[str]) -> bool:
    """Leading-zero numbers such as '007' are identifiers, not quantities."""
    for value in values[:50]:
        stripped = value.strip()
        if stripped.startswith("0") and len(stripped) > 1 and "." not in stripped:
            return True
    return False


def infer_types(dataframe: pd.DataFrame, sample_size: int = 400) -> list[str]:
    """Promote numeric-looking and date-looking text columns.

    Returns human-readable notes about the conversions applied.
    """
    notes: list[str] = []
    for name in dataframe.columns:
        series = dataframe[name]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        sample = series.dropna().head(sample_size)
        if sample.empty:
            continue
        values = sample.astype(str).str.strip()
        if (values == "").all():
            continue

        if IDENTIFIER_NAME.search(str(name)) or CODE_NAME.search(str(name)):
            continue

        numeric_ratio = values.str.match(NUMERIC_TEXT).mean()
        if numeric_ratio >= 0.995 and not _looks_like_code(values.tolist()):
            converted = pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
            if int(converted.isna().sum()) <= int(series.isna().sum()):
                dataframe[name] = converted
                notes.append(f"'{name}' converted from text to numbers")
                continue

        if len(sample) >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                except (ValueError, TypeError):
                    parsed = pd.Series([], dtype="datetime64[ns]")
            if len(parsed) and parsed.notna().mean() >= 0.995:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    converted = pd.to_datetime(series, errors="coerce")
                if int(converted.isna().sum()) <= int(series.isna().sum()):
                    dataframe[name] = converted
                    notes.append(f"'{name}' recognised as a date column")
                    continue

    return notes


def prepare_dataframe(
    frame: pd.DataFrame,
    *,
    filename: str,
    settings: Settings | None = None,
    truncated: bool = False,
    total_rows: int | None = None,
    encoding: str | None = None,
    delimiter: str | None = None,
) -> LoadedDataset:
    """Normalise a freshly parsed frame and validate it."""
    settings = settings or SETTINGS

    frame = frame.copy()
    normalise_columns(frame)

    # Drop rows and columns that are entirely empty (common trailing rows in
    # exported spreadsheets); this is structural, not analytical cleaning.
    frame = frame.dropna(axis=0, how="all").reset_index(drop=True)
    frame = frame.dropna(axis=1, how="all")

    if frame.shape[1] == 0:
        raise DatasetLoadError("No columns found in the file.")
    if frame.shape[0] == 0:
        raise DatasetLoadError("No data rows found in the file.")

    notes = infer_types(frame)

    if frame.shape[1] > 500:
        raise DatasetLoadError(
            f"The file has {frame.shape[1]:,} columns. "
            "Datasets with more than 500 columns are not supported."
        )

    return LoadedDataset(
        dataframe=frame,
        filename=filename,
        truncated=truncated,
        total_rows_in_file=total_rows,
        encoding=encoding,
        delimiter=delimiter,
        notes=notes,
    )


def load_bytes(
    payload: bytes,
    filename: str,
    settings: Settings | None = None,
) -> LoadedDataset:
    """Parse an uploaded file held in memory."""
    settings = settings or SETTINGS
    filename = Path(filename or "dataset.csv").name
    extension = Path(filename).suffix.lower()

    if not payload:
        raise DatasetLoadError("The uploaded file is empty.")

    if len(payload) > settings.max_upload_bytes:
        raise DatasetLoadError(
            f"File is larger than the {settings.max_upload_mb} MB limit."
        )

    if extension in TEXT_EXTENSIONS:
        extension = ".csv"

    if extension == ".csv":
        result = _read_csv(io.BytesIO(payload), settings)
    elif extension in EXCEL_EXTENSIONS:
        result = _read_excel(io.BytesIO(payload), extension, settings)
    else:
        raise DatasetLoadError(
            f"Unsupported file type '{extension or 'unknown'}'. Supported: {ALLOWED_LABEL}."
        )

    if result.dataframe.empty:
        raise DatasetLoadError("The file contains no data rows.")

    loaded = prepare_dataframe(
        result.dataframe,
        filename=filename,
        settings=settings,
        truncated=result.truncated,
        total_rows=result.total_rows,
        encoding=result.encoding,
        delimiter=result.delimiter,
    )

    if loaded.dataframe.shape[1] == 0:
        raise DatasetLoadError("The file contains no usable columns.")

    return loaded


def load_file(
    file_path: str | Path,
    settings: Settings | None = None,
) -> LoadedDataset:
    """Load a dataset from disk (used by scripts and tests)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    extension = path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise DatasetLoadError(
            f"Unsupported file format: {extension}. Supported: {ALLOWED_LABEL}."
        )

    return load_bytes(path.read_bytes(), path.name, settings=settings)


def load_data(file_path, settings: Settings | None = None) -> pd.DataFrame:
    """Backwards-compatible helper that returns just the DataFrame."""
    return load_file(file_path, settings=settings).dataframe


def get_basic_info(dataframe: pd.DataFrame) -> dict:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": [str(name) for name in dataframe.columns],
        "missing_values": int(dataframe.isna().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }


def display_basic_info(dataframe: pd.DataFrame) -> None:
    info = get_basic_info(dataframe)
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
