"""
Pyramid-Model-Core :: py/pyramid_support/dataset_reader.py
(installed as htm/pyramid/dataset_reader.py)

Batched dataset feeder for the C++ pyramid engine's PyReaderSource.

The engine pulls rows in batches so the GIL is touched once per
``batch_rows`` records (a few buffer copies), and everything else runs
GIL-free in C++.  One reader class covers the three sources the runners
actually use -- an in-memory DataFrame, a Parquet file (via pyarrow), and a
CSV (via pandas) -- with identical semantics:

  * ``columns`` selects and ORDERS the served columns (the engine's column
    order); every other column is never materialized.
  * ``start/stop/step`` apply pandas ``iloc``-style row slicing over the
    whole dataset (used by runners that carve train/test windows).
  * datetime columns are parsed IN PYTHON (``pandas.to_datetime`` -- i.e.
    the exact parser the Python pyramid used) and served as int32
    ``(n, 7)`` component arrays ``[year, month, day, hour, minute, second,
    valid]``; NaT rows carry ``valid == 0`` and encode through the NaN path.
  * numeric columns are served as float64, exactly pandas' default read.

Batch protocol (what PyReaderSource consumes):
    reader.total_rows  -> int | None   (sliced row count when cheaply known)
    reader.next_batch() -> {'n': int,
                            'num': {col: float64[n]},
                            'dt':  {col: int32[n, 7]}} | None

A fresh reader restarts the stream; the engine re-creates it through the
zero-arg factory it was given whenever it needs a reset (e.g. after the
encoder sample pass).

Design note: a native C++ Arrow/Parquet reader was evaluated and rejected
for the Windows build risk and dependency weight; this thin feeder keeps
Parquet (and anything else pandas can read) available at a per-batch GIL
cost measured well under 0.1% of a run.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


def _dt_components(series: pd.Series) -> np.ndarray:
    """Datetime series -> int32 (n, 7): y, mo, d, h, mi, s, valid."""
    s = pd.to_datetime(series, errors="coerce") if series.dtype == object \
        else series
    valid = s.notna().to_numpy()
    out = np.zeros((len(s), 7), dtype=np.int32)
    if valid.any():
        dt = s.dt
        out[:, 0] = np.where(valid, dt.year.fillna(0), 0)
        out[:, 1] = np.where(valid, dt.month.fillna(1), 1)
        out[:, 2] = np.where(valid, dt.day.fillna(1), 1)
        out[:, 3] = np.where(valid, dt.hour.fillna(0), 0)
        out[:, 4] = np.where(valid, dt.minute.fillna(0), 0)
        out[:, 5] = np.where(valid, dt.second.fillna(0), 0)
    out[:, 6] = valid.astype(np.int32)
    return out


class DatasetReader:
    """Batched, sliced, column-ordered reader over DataFrame/Parquet/CSV."""

    def __init__(self,
                 source: Union[pd.DataFrame, str, "os.PathLike[str]"],
                 columns: Sequence[str],
                 column_is_dt: Sequence[int],
                 dt_formats: Optional[Dict[str, str]] = None,
                 start: Optional[int] = None,
                 stop: Optional[int] = None,
                 step: Optional[int] = None,
                 batch_rows: int = 8192):
        self._columns: List[str] = list(columns)
        self._is_dt = list(column_is_dt)
        assert len(self._columns) == len(self._is_dt)
        self._dt_formats = dict(dt_formats or {})
        self._start = 0 if start is None else int(start)
        self._stop = stop if stop is None else int(stop)
        self._step = 1 if step is None else int(step)
        assert self._start >= 0 and self._step >= 1, \
            "DatasetReader supports non-negative start and positive step"
        self._batch_rows = int(batch_rows)

        self.total_rows: Optional[int] = None
        self._row_cursor = 0            # global row index (pre-slice)
        self._served = 0                # rows already emitted (post-slice)
        self._limit: Optional[int] = None   # sliced total when known

        if isinstance(source, pd.DataFrame):
            self._mode = "df"
            sliced = source.iloc[self._start:self._stop:self._step]
            self._df = sliced[self._columns].reset_index(drop=True)
            self.total_rows = len(self._df)
            self._limit = self.total_rows
        elif str(source).lower().endswith((".parquet", ".pq")):
            self._mode = "parquet"
            import pyarrow.parquet as pq
            self._pf = pq.ParquetFile(str(source))
            n_total = self._pf.metadata.num_rows
            self.total_rows = self._sliced_len(n_total)
            self._limit = self.total_rows
            self._batches = self._pf.iter_batches(
                batch_size=self._batch_rows, columns=self._columns)
            self._carry: Optional[pd.DataFrame] = None
        else:
            self._mode = "csv"
            # Row count would need a full scan; leave total unknown -- the
            # engine handles -1 (runs until exhausted / iteration-capped).
            self._chunks = pd.read_csv(str(source),
                                       usecols=self._columns,
                                       chunksize=self._batch_rows)

    # ------------------------------------------------------------------ #
    def _sliced_len(self, n_total: int) -> int:
        stop = n_total if self._stop is None else min(self._stop, n_total)
        span = max(0, stop - self._start)
        return (span + self._step - 1) // self._step

    def _slice_frame(self, df: pd.DataFrame, base: int) -> pd.DataFrame:
        """Apply the global (start, stop, step) to a chunk starting at
        global row index `base`."""
        n = len(df)
        idx = np.arange(base, base + n)
        keep = idx >= self._start
        if self._stop is not None:
            keep &= idx < self._stop
        if self._step != 1:
            keep &= ((idx - self._start) % self._step) == 0
        return df.iloc[np.nonzero(keep)[0]]

    def _pack(self, df: pd.DataFrame) -> dict:
        num: Dict[str, np.ndarray] = {}
        dt: Dict[str, np.ndarray] = {}
        for col, is_dt in zip(self._columns, self._is_dt):
            s = df[col]
            if is_dt:
                if s.dtype == object and col in self._dt_formats:
                    s = pd.to_datetime(s, format=self._dt_formats[col],
                                       errors="coerce")
                dt[col] = _dt_components(s)
            else:
                num[col] = s.to_numpy(dtype=np.float64, copy=True)
        return {"n": int(len(df)), "num": num, "dt": dt}

    # ------------------------------------------------------------------ #
    def next_batch(self) -> Optional[dict]:
        if self._mode == "df":
            if self._served >= len(self._df):
                return None
            part = self._df.iloc[self._served:self._served + self._batch_rows]
            self._served += len(part)
            return self._pack(part)

        if self._mode == "parquet":
            while True:
                try:
                    rb = next(self._batches)
                except StopIteration:
                    return None
                base = self._row_cursor
                self._row_cursor += rb.num_rows
                part = self._slice_frame(rb.to_pandas(), base)
                if len(part):
                    self._served += len(part)
                    return self._pack(part)
                if self._stop is not None and base >= self._stop:
                    return None

        # csv
        while True:
            try:
                chunk = next(self._chunks)
            except StopIteration:
                return None
            base = self._row_cursor
            self._row_cursor += len(chunk)
            part = self._slice_frame(chunk, base)
            if len(part):
                self._served += len(part)
                return self._pack(part)
            if self._stop is not None and base >= self._stop:
                return None
