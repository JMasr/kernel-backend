"""Periodic cleanup of orphaned temp files in the shared signing volume.

Background: signing jobs write a temp file to $TMPDIR (/signing) and pass its
path to the ARQ worker. In normal operation the file is consumed and left on
disk — the application never deletes it explicitly.  If a worker process is
SIGKILL-ed (the only path where the job's finally block does not run), the file
is never consumed either.  Over time these files accumulate on the signing_tmp
volume.

This cron job runs every 30 minutes and removes files older than ORPHAN_CUTOFF_MIN.
The cutoff must be safely above the maximum job duration (job_timeout = 360 s = 6 min)
to guarantee no file belonging to an active job is removed.  30 minutes gives a
24-minute margin.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

_log = logging.getLogger("kernel.cleanup")

# Files older than this are definitely not from an active job
# (job_timeout = 360 s → worst case 6 min; 30 min gives 24 min of margin)
_ORPHAN_CUTOFF_SECONDS = 30 * 60


async def cleanup_signing_tmp(ctx: dict) -> dict:
    """Delete orphaned temp files in the shared signing volume.

    Skips subdirectories and any file younger than _ORPHAN_CUTOFF_SECONDS.
    Errors on individual files are logged and skipped — one bad file must not
    abort cleanup of the rest.

    Returns a summary dict so the result is visible in ARQ's job log.
    """
    signing_dir = Path(os.environ.get("TMPDIR", "/signing"))

    if not signing_dir.is_dir():
        _log.warning("cleanup: signing dir %s does not exist — skipping", signing_dir)
        return {"skipped": True, "reason": f"{signing_dir} not found"}

    cutoff = time.time() - _ORPHAN_CUTOFF_SECONDS
    deleted: list[str] = []
    errors: list[str] = []

    for entry in signing_dir.iterdir():
        if not entry.is_file():
            # Skip subdirectories and anything that is not a regular file
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink(missing_ok=True)
                deleted.append(entry.name)
        except OSError as exc:
            # Race condition (another cleanup ran concurrently) or permission error
            _log.warning("cleanup: could not remove %s: %s", entry, exc)
            errors.append(str(exc))

    if deleted:
        _log.info(
            "cleanup: removed %d orphaned file(s) from %s: %s",
            len(deleted),
            signing_dir,
            ", ".join(deleted),
        )
    else:
        _log.debug("cleanup: no orphaned files found in %s", signing_dir)

    return {
        "dir": str(signing_dir),
        "deleted": len(deleted),
        "errors": len(errors),
        "cutoff_minutes": _ORPHAN_CUTOFF_SECONDS // 60,
    }
