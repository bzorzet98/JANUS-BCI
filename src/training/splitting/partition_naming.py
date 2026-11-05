"""Single place that builds partition folder names (R15, R16)."""
from typing import Any


def build_partition_name(unit_id: Any, fold_idx: int) -> str:
    """Partition folder name `unit_<id>_fold_<k>`, the same for every split
    schema. The unit id is used as text, without padding. Raises ValueError
    if the id contains a path separator or `_fold_`."""
    key = str(unit_id)
    if "/" in key or "\\" in key or "_fold_" in key:
        raise ValueError(
            f"Unit id {key!r} cannot contain '/', '\\' or '_fold_', because it "
            f"becomes a folder name. Change the unit ids that the data provider "
            f"returns (list_units)."
        )
    return f"unit_{key}_fold_{fold_idx}"
