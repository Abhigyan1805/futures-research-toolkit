"""Excel summary workbook (openpyxl) for the regime study.

A single ``regimes.xlsx`` is the human-facing deliverable: one sheet per table
plus a provenance sheet stating the FRED-proxy caveat. Sheet names are
sanitised to Excel's 31-character limit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MAX_SHEET = 31


def _sheet_name(name: str, used: set[str]) -> str:
    clean = "".join(c if c.isalnum() or c in "._- " else "_" for c in name)[:MAX_SHEET]
    base = clean or "sheet"
    candidate = base
    i = 1
    while candidate in used:
        suffix = f"_{i}"
        candidate = base[: MAX_SHEET - len(suffix)] + suffix
        i += 1
    used.add(candidate)
    return candidate


def write_workbook(path: Path, tables: dict, provenance: dict | None = None) -> None:
    """Write each DataFrame in ``tables`` to its own sheet, plus provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=_sheet_name(name, used))
        if provenance:
            prov = pd.DataFrame(
                {"key": list(provenance), "value": [str(v) for v in provenance.values()]}
            )
            prov.to_excel(writer, sheet_name=_sheet_name("provenance", used), index=False)
