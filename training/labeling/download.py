"""Stage 1 — pull images from the Zenblen Files API into ``labeling/data/images/``.

Fetches every file in a time range (default type ``image/jpg``), records metadata
in the ``files`` table, and downloads each ``file_url`` to
``data/images/<file_id>.jpg``. Resumable: existing rows/files are skipped.

Auth: the API key is read from the ``ZENBLEN_API_KEY`` environment variable
(preferred, per project convention — no hardcoded keys). ``--api-key`` overrides.

Examples
--------
  export ZENBLEN_API_KEY=...            # do NOT commit the key
  python labeling/download.py --start '2026-06-29 00:00:00' \\
                              --end   '2026-06-30 00:00:00'
  python labeling/download.py --start '2026-01-01 00:00:00' \\
                              --end   '2026-07-01 00:00:00' --category CleanDone
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

_LABELING = Path(__file__).resolve().parent
_TRAINING = _LABELING.parent
_REPO = _TRAINING.parent
sys.path.insert(0, str(_TRAINING))  # `import labeling`
sys.path.insert(0, str(_REPO / "active_pipeline"))  # `import smoothie_cv`
from labeling import db

API_URL = "https://internal-api.zenblen.net/files"


def fetch_file_list(
    api_key: str,
    start: str,
    end: str,
    category: str | None,
    file_type: str | None,
    timeout: float = 60.0,
) -> list[dict]:
    """POST to the Files API and return the array of file objects."""
    body: dict[str, object] = {"startTime": start, "endTime": end}
    if category:
        body["fileCategoryName"] = category
    if file_type:
        body["fileTypeName"] = file_type
    resp = requests.post(
        API_URL,
        headers={"Content-Type": "application/json", "x-api-key": api_key},
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array, got {type(data).__name__}: {data!r:.200}")
    return data


def _record_files(conn, files: list[dict]) -> None:
    """Upsert metadata rows (does not download yet)."""
    conn.executemany(
        """
        INSERT INTO files (file_id, order_id, file_name, file_url,
                           category_name, file_type, created_at)
        VALUES (:file_id, :order_id, :file_name, :file_url,
                :category_name, :file_type, :created_at)
        ON CONFLICT(file_id) DO UPDATE SET
            order_id=excluded.order_id, file_name=excluded.file_name,
            file_url=excluded.file_url, category_name=excluded.category_name,
            file_type=excluded.file_type, created_at=excluded.created_at
        """,
        [
            {
                "file_id": f["file_id"],
                "order_id": f.get("order_id"),
                "file_name": f.get("file_name", ""),
                "file_url": f.get("file_url", ""),
                "category_name": f.get("category_name", ""),
                "file_type": f.get("file_type", ""),
                "created_at": f.get("created_at", ""),
            }
            for f in files
        ],
    )
    conn.commit()


def _download_pending(conn, session: requests.Session, timeout: float,
                      category: str | None = None) -> tuple[int, int, int]:
    """Download not-yet-downloaded files with a non-empty url.

    ``category`` scopes the pass to one ``files.category_name``. Without it this
    drains the WHOLE pending backlog across every category — which is the right
    behaviour for resuming an interrupted pull, but a nasty surprise when you
    asked for one category: ``--category XYTable`` would spend its time fetching
    thousands of pending CleanDone images and never reach the 162 you wanted.
    main() therefore passes the requested category through by default.

    Returns (downloaded, skipped_existing, skipped_no_url).
    """
    sql = "SELECT file_id, file_url FROM files WHERE downloaded = 0"
    params: tuple = ()
    if category:
        sql += " AND category_name = ?"
        params = (category,)
    rows = conn.execute(sql, params).fetchall()
    downloaded = skipped_existing = skipped_no_url = 0
    total = len(rows)
    for i, row in enumerate(rows, 1):
        fid, url = row["file_id"], row["file_url"]
        dest = db.IMAGES_DIR / f"{fid}.jpg"
        if dest.exists():
            conn.execute("UPDATE files SET downloaded = 1 WHERE file_id = ?", (fid,))
            skipped_existing += 1
            continue
        if not url:
            skipped_no_url += 1
            print(f"[{i}/{total}] {fid}  SKIP (empty file_url)")
            continue
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            dest.write_bytes(r.content)
            conn.execute("UPDATE files SET downloaded = 1 WHERE file_id = ?", (fid,))
            downloaded += 1
            print(f"[{i}/{total}] {fid}  ok ({len(r.content) // 1024} KB)")
        except requests.RequestException as e:
            print(f"[{i}/{total}] {fid}  ERROR {e}")
        if i % 50 == 0:
            conn.commit()
    conn.commit()
    return downloaded, skipped_existing, skipped_no_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Download images from the Zenblen Files API")
    parser.add_argument("--start", required=True,
                        help="Start of range, 'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--end", required=True,
                        help="End of range, 'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--category", default=None,
                        help="Optional fileCategoryName filter (e.g. CleanDone)")
    parser.add_argument("--type", dest="file_type", default="image/jpg",
                        help="fileTypeName filter (default: image/jpg; '' for any)")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: ZENBLEN_API_KEY env var)")
    parser.add_argument("--list-only", action="store_true",
                        help="Fetch + record metadata but do not download images")
    parser.add_argument("--backfill-all", action="store_true",
                        help="Download EVERY pending image in the DB, not just "
                        "--category's. Use to resume an interrupted pull; "
                        "without --category it is already the behaviour.")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Per-request timeout in seconds (default: 60)")
    args = parser.parse_args()

    db.load_dotenv()  # pull ZENBLEN_API_KEY from labeling/.env or repo .env if present
    api_key = args.api_key or os.environ.get("ZENBLEN_API_KEY")
    if not api_key:
        parser.error("no API key: set ZENBLEN_API_KEY (env or .env file) or pass --api-key")

    db.ensure_dirs()
    conn = db.connect()

    print(f"fetching file list {args.start} .. {args.end} "
          f"(category={args.category or 'any'}, type={args.file_type or 'any'})")
    files = fetch_file_list(api_key, args.start, args.end,
                            args.category, args.file_type or None, args.timeout)
    print(f"  {len(files)} files returned")
    _record_files(conn, files)

    if args.list_only:
        print("--list-only: metadata recorded, skipping downloads")
        return

    # Scope the download pass to --category unless --backfill-all is asked for,
    # so a single-category pull doesn't drain an unrelated backlog first.
    scope = None if args.backfill_all else args.category
    if scope:
        print(f"downloading pending {scope} images only (--backfill-all for every category)")
    with requests.Session() as session:
        dl, skip_ex, skip_no_url = _download_pending(conn, session, args.timeout, scope)
    print(f"done: {dl} downloaded, {skip_ex} already on disk, "
          f"{skip_no_url} skipped (empty url)")


if __name__ == "__main__":
    main()
