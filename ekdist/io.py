"""SCN file I/O — thin adapter over dcio.

All binary parsing lives in ``dcio.formats.scn``.  This module re-exports
:func:`read_scn` for backward compatibility with code that imports from
``ekdist.io``, and exposes :class:`~dcio.formats.scn.SCNRecord` as the
canonical return type.
"""

from __future__ import annotations

from pathlib import Path

from dcio.formats.scn import SCNRecord, read

__all__ = ["read_scn", "SCNRecord"]


def read_scn(path: str | Path, *, verbose: bool = False) -> SCNRecord:
    """Read a SCAN binary (.SCN) file and return an :class:`SCNRecord`.

    Parameters
    ----------
    path:
        Path to the ``.scn`` file.
    verbose:
        Ignored (kept for API compatibility with the old ekscn-based reader).

    Returns
    -------
    SCNRecord
        Dataclass with ``.intervals`` (seconds), ``.amplitudes`` (pA),
        ``.flags`` (int8), and ``.header`` (SCNHeader).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file contains an unrecognised SCAN version number.
    """
    return read(path)
