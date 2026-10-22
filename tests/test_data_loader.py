"""Loader tests: formats, validation, type inference and the row cap."""

from __future__ import annotations

import io
import math

import pandas as pd
import pytest

from src.config import Settings
from src.data_loader import (
    DatasetLoadError,
    get_basic_info,
    infer_types,
    load_bytes,
    load_file,
)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_file(tmp_path / "nope.csv")


def test_directory_path(tmp_path):
    with pytest.raises(ValueError):
        load_file(tmp_path)


def test_unsupported_file(tmp_path):
    path = tmp_path / "notes.pdf"
    path.write_bytes(b"%PDF-1.4")
    with pytest.raises(DatasetLoadError, match="Unsupported"):
        load_file(path)


def test_empty_payload():
    with pytest.raises(DatasetLoadError, match="empty"):
        load_bytes(b"", "empty.csv")


def test_header_only():
    with pytest.raises(DatasetLoadError, match="no data rows"):
        load_bytes(b"a,b\n", "header.csv")


def test_binary_garbage():
    with pytest.raises(DatasetLoadError):
        load_bytes(b"\x00\x01\x02\x03\x04", "bad.csv")


def test_csv_basics(tmp_path, customers_frame):
    path = tmp_path / "customers.csv"
    customers_frame.to_csv(path, index=False)

    loaded = load_file(path)
    frame = loaded.dataframe

    assert len(frame) == len(customers_frame)
    assert list(frame.columns) == list(customers_frame.columns)
    assert pd.api.types.is_datetime64_any_dtype(frame["signup_date"])
    assert "recognised as a date column" in " ".join(loaded.notes)
    assert loaded.meta()["filename"] == "customers.csv"


def test_column_names_are_normalised():
    payload = b" first name ,first name,Unnamed: 2,value\nAlice,Alice,,1\nBob,Bob,,2\n"
    loaded = load_bytes(payload, "wide.csv")
    columns = list(loaded.dataframe.columns)

    assert columns[0] == "first name"
    assert columns[1] == "first name_1"
    assert columns[1] != columns[0]
    assert len(columns) == 3  # the unnamed all-empty column is dropped
    assert "value" in columns


def test_numeric_text_is_promoted():
    payload = b"amount,code\n1,007\n2,008\n3,009\n"
    loaded = load_bytes(payload, "numbers.csv")
    assert pd.api.types.is_numeric_dtype(loaded.dataframe["amount"])
    # leading-zero codes stay text
    assert not pd.api.types.is_numeric_dtype(loaded.dataframe["code"])


def test_semicolon_delimiter():
    payload = b"a;b;c\n1;2;3\n4;5;6\n7;8;9\n"
    loaded = load_bytes(payload, "semi.csv")
    assert list(loaded.dataframe.columns) == ["a", "b", "c"]
    assert loaded.dataframe.shape == (3, 3)


def test_delimited_text_extensions(customers_frame):
    frame = customers_frame.head(20)
    for extension in (".tsv", ".txt", ".tab"):
        payload = frame.to_csv(index=False, sep="\t").encode()
        loaded = load_bytes(payload, f"customers{extension}")
        assert loaded.dataframe.shape == frame.shape, extension
        assert list(loaded.dataframe.columns)[:2] == ["customer_id", "signup_date"]


def test_excel_roundtrip():
    frame = pd.DataFrame({"x": [1, 2, 3, 4], "y": ["a", "b", "a", "b"]})
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    buffer.seek(0)

    loaded = load_bytes(buffer.getvalue(), "book.xlsx")
    assert loaded.dataframe.shape == (4, 2)
    assert loaded.notes == []


def test_row_cap_sets_truncation_flag():
    settings = Settings(max_rows=5)
    payload = ("a\n" + "\n".join(str(value) for value in range(20))).encode()
    loaded = load_bytes(payload, "long.csv", settings=settings)

    assert len(loaded.dataframe) == 5
    assert loaded.truncated is True
    assert loaded.total_rows_in_file == 20
    assert loaded.meta()["truncated"] is True


def test_filename_path_traversal_is_stripped():
    loaded = load_bytes(b"a,b\n1,2\n3,4\n", "../../etc/passwd.csv")
    assert loaded.filename == "passwd.csv"


def test_infer_types_is_idempotent():
    frame = pd.DataFrame({"a": ["1", "2", "3"], "b": ["2024-01-01", "2024-01-02", "2024-01-03"]})
    infer_types(frame)
    assert pd.api.types.is_numeric_dtype(frame["a"])
    assert pd.api.types.is_datetime64_any_dtype(frame["b"])


def test_get_basic_info(customers_frame):
    info = get_basic_info(customers_frame)

    assert info["rows"] == len(customers_frame)
    assert info["columns"] == customers_frame.shape[1]
    assert info["column_names"][0] == "customer_id"
    assert info["missing_values"] == int(customers_frame.isna().sum().sum())
    assert info["duplicate_rows"] == 1


def test_get_basic_info_rejects_non_frames():
    with pytest.raises(TypeError):
        get_basic_info([1, 2, 3])


def test_dates_in_mixed_formats_do_not_crash():
    payload = b"when\n2024-01-01\n2024-02-01\n2024-03-01\n"
    loaded = load_bytes(payload, "dates.csv")
    assert pd.api.types.is_datetime64_any_dtype(loaded.dataframe["when"])


def test_memory_is_reported_without_nan(tmp_path, customers_frame):
    path = tmp_path / "customers.csv"
    customers_frame.to_csv(path, index=False)
    loaded = load_file(path)
    assert math.isfinite(loaded.dataframe.memory_usage(deep=True).sum())
