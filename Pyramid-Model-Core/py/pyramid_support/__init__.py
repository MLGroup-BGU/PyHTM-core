"""htm.pyramid: Python support for the C++ pyramid engine
(htm.bindings.pyramid_engine).  DatasetReader feeds the engine batched
rows from a DataFrame / Parquet / CSV with iloc-style slicing."""
from .dataset_reader import DatasetReader

__all__ = ["DatasetReader"]
