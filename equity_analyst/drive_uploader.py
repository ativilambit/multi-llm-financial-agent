from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from google.auth.exceptions import MalformedError, TransportError
from google.oauth2 import service_account
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
from googleapiclient.http import MediaFileUpload  # type: ignore[import-untyped]

from equity_analyst.config import RunConfig

logger = logging.getLogger(__name__)

_DRIVE_SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/drive",)


def _is_malformed_service_account_key_error(exc: BaseException) -> bool:
    """Detect malformed service-account JSON errors.

    ``google.auth.exceptions.MalformedError`` subclasses ``ValueError``. In rare setups the
    exception can come from a second loaded copy of ``google.auth``, so ``isinstance`` against
    the module's imported ``MalformedError`` may fail; fall back to duck-typing by class location.
    """
    if isinstance(exc, MalformedError):
        return True
    t = type(exc)
    return t.__module__ == "google.auth.exceptions" and t.__name__ == "MalformedError"


def _log_invalid_drive_service_account_key(*, cred_path: Path, detail: str) -> None:
    logger.warning(
        "Drive upload skipped: credentials file at %s is not a valid service account key (%s). "
        "Replace it with a key from Google Cloud Console → IAM & Admin → Service accounts → "
        "(your SA) → Keys → Add key → JSON. Run continues.",
        cred_path,
        detail,
    )

_RESUMABLE_THRESHOLD_BYTES = 5 * 1024 * 1024


def _expand_credentials_path(raw: str | None) -> Path | None:
    if raw is None or not str(raw).strip():
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(raw).strip()))).resolve()


def _service_account_key_file_issue(path: Path) -> str | None:
    """Return a short message if ``path`` is not a usable Drive service-account JSON key, else None.

    Uses ``from_service_account_file`` so validation matches the upload path in
    :meth:`DriveUploader._ensure_service`.
    """
    try:
        service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
            str(path),
            scopes=_DRIVE_SCOPES,
        )
    except OSError as exc:
        return f"cannot read file ({exc})"
    except (MalformedError, ValueError) as exc:
        return str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        return str(exc)
    return None


def log_drive_upload_plan(
    *,
    drive_upload_enabled: bool,
    drive_credentials_path: str | None,
    drive_root_folder_id: str | None,
) -> None:
    """Log whether Drive upload will run and why; call once per run near startup."""
    if not drive_upload_enabled:
        logger.info(
            "Drive upload: DISABLED (reason=config: drive_upload_enabled is false)",
        )
        return

    cred_path = _expand_credentials_path(drive_credentials_path)
    root_id = (drive_root_folder_id or "").strip()

    if cred_path is None:
        logger.warning(
            "Drive upload: DISABLED (reason=no credentials path; set drive_credentials_path in YAML "
            "or DRIVE_CREDENTIALS_PATH to a Google Cloud service account JSON key file)",
        )
        return
    if not cred_path.is_file():
        logger.warning(
            "Drive upload: DISABLED (reason=credentials file not found at %s)",
            cred_path,
        )
        return
    if not root_id:
        logger.warning(
            "Drive upload: DISABLED (reason=no drive_root_folder_id; set drive_root_folder_id in YAML "
            "or DRIVE_ROOT_FOLDER_ID to the shared Drive folder id)",
        )
        return

    issue = _service_account_key_file_issue(cred_path)
    if issue:
        _log_invalid_drive_service_account_key(cred_path=cred_path, detail=issue)
        return

    logger.info(
        "Drive upload: ENABLED (folder_id=%s, credentials=%s)",
        root_id,
        cred_path,
    )


def log_drive_upload_plan_from_config(cfg: RunConfig) -> None:
    log_drive_upload_plan(
        drive_upload_enabled=cfg.drive_upload_enabled,
        drive_credentials_path=cfg.drive_credentials_path,
        drive_root_folder_id=cfg.drive_root_folder_id,
    )


def _guess_mimetype(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".md":
        return "text/markdown"
    if ext == ".json":
        return "application/json"
    if ext == ".log":
        return "text/plain"
    if ext == ".sqlite":
        return "application/x-sqlite3"
    return "application/octet-stream"


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, HttpError):
        status = getattr(getattr(exc, "resp", None), "status", None)
        if status is None:
            return True
        try:
            code = int(status)
        except (TypeError, ValueError):
            return True
        return code in (429, 500, 502, 503, 504)
    return isinstance(exc, (TimeoutError, TransportError, ConnectionError, OSError))


def _drive_query_escape_name(name: str) -> str:
    return name.replace("\\", "\\\\").replace("'", "\\'")


class DriveUploader:
    """Upload a local directory tree to Google Drive using a service account."""

    def __init__(self, credentials_path: str | Path, root_folder_id: str) -> None:
        self._credentials_path = Path(credentials_path)
        self._root_folder_id = root_folder_id.strip()
        self._service: Any = None

    def _ensure_service(self) -> Any:
        if self._service is not None:
            return self._service
        creds = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
            str(self._credentials_path),
            scopes=_DRIVE_SCOPES,
        )
        # Lazy import keeps mypy happy with googleapiclient.discovery.build return type
        from googleapiclient import discovery  # type: ignore[import-untyped]

        self._service = discovery.build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def _list_child_folder_id(self, *, parent_id: str, name: str) -> str | None:
        svc = self._ensure_service()
        q = (
            f"'{parent_id}' in parents and "
            f"mimeType = 'application/vnd.google-apps.folder' and "
            f"name = '{_drive_query_escape_name(name)}' and trashed = false"
        )
        res = (
            svc.files()
            .list(
                q=q,
                spaces="drive",
                fields="files(id, name)",
                pageSize=10,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files = res.get("files") or []
        if not files:
            return None
        return str(files[0]["id"])

    def _create_folder(self, *, parent_id: str, name: str) -> str:
        svc = self._ensure_service()
        body: dict[str, Any] = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        created = (
            svc.files()
            .create(body=body, fields="id", supportsAllDrives=True)
            .execute()
        )
        return str(created["id"])

    def _get_or_create_folder(self, *, parent_id: str, name: str) -> str:
        existing = self._list_child_folder_id(parent_id=parent_id, name=name)
        if existing is not None:
            return existing
        return self._create_folder(parent_id=parent_id, name=name)

    def _upload_file_with_retries(
        self,
        *,
        local_path: Path,
        rel_path: str,
        parent_folder_id: str,
    ) -> tuple[str, int]:
        size_bytes = local_path.stat().st_size
        mime = _guess_mimetype(local_path)
        resumable = size_bytes > _RESUMABLE_THRESHOLD_BYTES
        body: dict[str, Any] = {"name": local_path.name, "parents": [parent_folder_id]}
        last_exc: BaseException | None = None
        for attempt in range(3):
            try:
                media = MediaFileUpload(str(local_path), mimetype=mime, resumable=resumable)
                svc = self._ensure_service()
                created = (
                    svc.files()
                    .create(
                        body=body,
                        media_body=media,
                        fields="id",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
                file_id = str(created["id"])
                logger.info(
                    "Uploaded path=%s drive_file_id=%s size_bytes=%s",
                    rel_path,
                    file_id,
                    size_bytes,
                )
                return file_id, size_bytes
            except BaseException as exc:
                last_exc = exc
                if attempt >= 2 or not _is_retryable_error(exc):
                    raise
                delay_s = 2.0**attempt
                logger.warning(
                    "Drive upload retry path=%s attempt=%s/%s delay_s=%s error=%s: %s",
                    rel_path,
                    attempt + 2,
                    3,
                    delay_s,
                    type(exc).__name__,
                    exc,
                )
                time.sleep(delay_s)
        assert last_exc is not None
        raise last_exc

    def _ensure_parent_chain(self, *, run_folder_id: str, relative_dir: Path) -> str:
        """Return Drive folder id for `relative_dir` under `run_folder_id`."""
        parent_id = run_folder_id
        if relative_dir == Path("."):
            return parent_id
        for part in relative_dir.parts:
            if part in (".", ""):
                continue
            parent_id = self._get_or_create_folder(parent_id=parent_id, name=part)
        return parent_id

    def upload_directory(self, local_dir: Path, *, run_id: str) -> str:
        local_dir = local_dir.resolve()
        if not local_dir.is_dir():
            raise NotADirectoryError(str(local_dir))

        run_folder_id = self._get_or_create_folder(parent_id=self._root_folder_id, name=run_id)

        files_uploaded = 0
        total_bytes = 0

        for root, dirnames, filenames in os.walk(local_dir, topdown=True):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            rel_root = Path(root).resolve().relative_to(local_dir)
            parent_drive_id = self._ensure_parent_chain(
                run_folder_id=run_folder_id,
                relative_dir=rel_root,
            )

            for fname in filenames:
                if fname.startswith("."):
                    continue
                fp = Path(root) / fname
                if not fp.is_file():
                    continue
                rel_file = fp.relative_to(local_dir)
                _, sz = self._upload_file_with_retries(
                    local_path=fp,
                    rel_path=str(rel_file).replace(os.sep, "/"),
                    parent_folder_id=parent_drive_id,
                )
                files_uploaded += 1
                total_bytes += int(sz)

        folder_url = f"https://drive.google.com/drive/folders/{run_folder_id}"
        logger.info(
            "Drive upload complete folder_url=%s files_uploaded=%s bytes=%s",
            folder_url,
            files_uploaded,
            total_bytes,
        )
        return folder_url


def _append_drive_url_to_run_json(out_dir: Path, folder_url: str) -> None:
    run_json = out_dir / "run.json"
    if not run_json.is_file():
        return
    meta = json.loads(run_json.read_text(encoding="utf-8"))
    meta["drive_folder_url"] = folder_url
    run_json.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_synthesis_footer(out_dir: Path, folder_url: str) -> None:
    syn = out_dir / "synthesis.md"
    if not syn.is_file():
        return
    text = syn.read_text(encoding="utf-8")
    footer = f"\n\n---\n\n**Google Drive:** {folder_url}\n"
    if "**Google Drive:**" in text:
        return
    syn.write_text(text.rstrip() + footer, encoding="utf-8")


async def maybe_upload_run_to_drive_raw(
    *,
    drive_upload_enabled: bool,
    drive_credentials_path: str | None,
    drive_root_folder_id: str | None,
    out_dir: Path,
    run_id: str | None = None,
    append_synthesis_footer: bool = False,
) -> str | None:
    """Optionally upload `out_dir` to Drive; never raises. Returns folder URL or None."""
    if not drive_upload_enabled:
        return None

    cred_path = _expand_credentials_path(drive_credentials_path)
    root_id = (drive_root_folder_id or "").strip()

    if cred_path is None or not cred_path.is_file():
        logger.warning(
            "Drive upload skipped: drive_upload_enabled=true but drive_credentials_path is missing "
            "or not a readable file (resolved=%s). Set DRIVE_CREDENTIALS_PATH or drive_credentials_path.",
            str(cred_path) if cred_path is not None else None,
        )
        return None
    if not root_id:
        logger.warning(
            "Drive upload skipped: drive_upload_enabled=true but drive_root_folder_id is missing. "
            "Set DRIVE_ROOT_FOLDER_ID or drive_root_folder_id.",
        )
        return None

    issue = _service_account_key_file_issue(cred_path)
    if issue is not None:
        _log_invalid_drive_service_account_key(cred_path=cred_path, detail=issue)
        return None

    rid = run_id if run_id is not None else out_dir.name

    def _sync_upload() -> str:
        uploader = DriveUploader(credentials_path=cred_path, root_folder_id=root_id)
        return uploader.upload_directory(out_dir.resolve(), run_id=rid)

    try:
        folder_url = await asyncio.to_thread(_sync_upload)
    except Exception as exc:
        if _is_malformed_service_account_key_error(exc):
            _log_invalid_drive_service_account_key(cred_path=cred_path, detail=str(exc))
            return None
        logger.error("Drive upload failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        return None

    logger.info("Run uploaded to Drive: %s", folder_url)

    def _sync_record() -> None:
        _append_drive_url_to_run_json(out_dir, folder_url)
        if append_synthesis_footer:
            _append_synthesis_footer(out_dir, folder_url)

    try:
        await asyncio.to_thread(_sync_record)
    except Exception as exc:
        logger.error(
            "Drive upload succeeded but failed to update run artifacts: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )

    return folder_url


async def maybe_upload_run_to_drive(
    cfg: RunConfig,
    out_dir: Path,
    *,
    run_id: str | None = None,
    append_synthesis_footer: bool = False,
) -> str | None:
    """Optionally upload `out_dir` using settings from ``RunConfig``."""
    return await maybe_upload_run_to_drive_raw(
        drive_upload_enabled=cfg.drive_upload_enabled,
        drive_credentials_path=cfg.drive_credentials_path,
        drive_root_folder_id=cfg.drive_root_folder_id,
        out_dir=out_dir,
        run_id=run_id,
        append_synthesis_footer=append_synthesis_footer,
    )
