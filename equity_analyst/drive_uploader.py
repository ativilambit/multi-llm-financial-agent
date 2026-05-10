from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, cast

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


def _build_drive_service(cred_path: Path) -> Any:
    creds = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
        str(cred_path),
        scopes=_DRIVE_SCOPES,
    )
    from googleapiclient import discovery  # type: ignore[import-untyped]

    return discovery.build("drive", "v3", credentials=creds, cache_discovery=False)


def _drive_root_preflight_probe(
    cred_path: Path,
    root_id: str,
    *,
    emit_capability_warning: bool,
) -> dict[str, Any] | None:
    """Fetch root folder metadata; return metadata dict on success, else None.

    The personal-drive WARNING is emitted at most once per process per ``(cred_path, root_id)``.
    """
    key = (str(cred_path.resolve()), root_id)
    meta: dict[str, Any]
    try:
        svc = _build_drive_service(cred_path)
        raw_meta = (
            svc.files()
            .get(
                fileId=root_id,
                fields="id,name,mimeType,driveId,capabilities",
                supportsAllDrives=True,
            )
            .execute()
        )
        if not isinstance(raw_meta, dict):
            logger.warning(
                "Drive upload: unexpected root folder response type for id=%s. "
                "Disabling Drive upload for this run.",
                root_id,
            )
            return None
        meta = cast(dict[str, Any], raw_meta)
    except HttpError as exc:
        if _is_sa_my_drive_storage_quota_error(exc):
            _log_drive_storage_quota_warning_once(already_emitted=[False])
        else:
            _log_drive_http_client_error_no_traceback(exc)
        return None
    except (TransportError, OSError, ValueError, TypeError) as exc:
        logger.warning(
            "Drive upload: preflight could not reach Google Drive for folder id=%s (%s: %s). "
            "Disabling Drive upload for this run.",
            root_id,
            type(exc).__name__,
            exc,
        )
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Drive upload: unexpected preflight error for folder id=%s (%s: %s). "
            "Disabling Drive upload for this run.",
            root_id,
            type(exc).__name__,
            exc,
        )
        return None

    mime = meta.get("mimeType")
    if mime != _FOLDER_MIMETYPE:
        logger.warning(
            "Drive upload: drive_root_folder_id %s points to mimeType=%r (expected a folder). "
            "Disabling Drive upload for this run.",
            root_id,
            mime,
        )
        return None

    if not meta.get("driveId"):
        if key not in _PERSONAL_DRIVE_WARNED_KEYS:
            _PERSONAL_DRIVE_WARNED_KEYS.add(key)
            logger.warning(
                "Drive folder %s is not inside a Shared Drive. Service-account uploads will fail with "
                "storageQuotaExceeded. Move the folder into a Shared Drive (Workspace) or switch to "
                "OAuth delegation. Disabling Drive upload for this run.",
                root_id,
            )
        return None

    raw_caps = meta.get("capabilities")
    caps: dict[str, Any] = cast(dict[str, Any], raw_caps) if isinstance(raw_caps, dict) else {}
    can_add = caps.get("canAddChildren")
    if emit_capability_warning and can_add is not True:
        logger.warning(
            "Drive upload: service account may lack Content Manager access on folder %r (%s): "
            "capabilities.canAddChildren is not true. Grant the service account Content Manager on this "
            "folder (or the Shared Drive) so uploads can create subfolders and files.",
            meta.get("name") or "?",
            root_id,
        )

    return meta


def log_drive_upload_plan(
    *,
    drive_upload_enabled: bool,
    drive_credentials_path: str | None,
    drive_root_folder_id: str | None,
) -> bool:
    """Log whether Drive upload will run and why; call once per run near startup.

    Returns True if uploads should proceed for this run (Shared Drive preflight passed).
    """
    if not drive_upload_enabled:
        logger.info(
            "Drive upload: DISABLED (reason=config: drive_upload_enabled is false)",
        )
        return False

    cred_path = _expand_credentials_path(drive_credentials_path)
    root_id = (drive_root_folder_id or "").strip()

    if cred_path is None:
        logger.warning(
            "Drive upload: DISABLED (reason=no credentials path; set drive_credentials_path in YAML "
            "or DRIVE_CREDENTIALS_PATH to a Google Cloud service account JSON key file)",
        )
        return False
    if not cred_path.is_file():
        logger.warning(
            "Drive upload: DISABLED (reason=credentials file not found at %s)",
            cred_path,
        )
        return False
    if not root_id:
        logger.warning(
            "Drive upload: DISABLED (reason=no drive_root_folder_id; set drive_root_folder_id in YAML "
            "or DRIVE_ROOT_FOLDER_ID to the shared Drive folder id)",
        )
        return False

    issue = _service_account_key_file_issue(cred_path)
    if issue:
        _log_invalid_drive_service_account_key(cred_path=cred_path, detail=issue)
        return False

    meta = _drive_root_preflight_probe(
        cred_path,
        root_id,
        emit_capability_warning=True,
    )
    if meta is None:
        return False

    folder_name = str(meta.get("name") or "?")
    drive_id = str(meta.get("driveId") or "?")
    logger.info(
        "Drive upload: ENABLED (folder=%s, shared_drive=%s)",
        folder_name,
        drive_id,
    )
    return True


def log_drive_upload_plan_from_config(cfg: RunConfig) -> RunConfig:
    """Log Drive plan and return config with ``drive_upload_enabled`` cleared if preflight forbids upload."""
    ok = log_drive_upload_plan(
        drive_upload_enabled=cfg.drive_upload_enabled,
        drive_credentials_path=cfg.drive_credentials_path,
        drive_root_folder_id=cfg.drive_root_folder_id,
    )
    if ok or not cfg.drive_upload_enabled:
        return cfg
    return cfg.model_copy(update={"drive_upload_enabled": False})


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


_FOLDER_MIMETYPE = "application/vnd.google-apps.folder"
_SHARED_DRIVE_DOC = "https://developers.google.com/workspace/drive/api/guides/about-shareddrives"

# Dedupe personal-drive preflight warnings when both startup logging and upload probe run.
_PERSONAL_DRIVE_WARNED_KEYS: set[tuple[str, str]] = set()


def _read_service_account_email(cred_path: Path) -> str | None:
    try:
        data = json.loads(cred_path.read_text(encoding="utf-8"))
        raw = data.get("client_email")
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        return None
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip()


def _is_sa_my_drive_storage_quota_error(exc: BaseException) -> bool:
    """403 from Drive when a service account uploads bytes into non-Shared-Drive storage."""
    if not isinstance(exc, HttpError):
        return False
    status_raw = getattr(getattr(exc, "resp", None), "status", None)
    try:
        code = int(status_raw) if status_raw is not None else 0
    except (TypeError, ValueError):
        return False
    if code != 403:
        return False
    text = str(exc).lower()
    details = getattr(exc, "error_details", "")
    try:
        details_blob = json.dumps(details).lower() if details else ""
    except (TypeError, ValueError):
        details_blob = str(details).lower()
    content = getattr(exc, "content", b"") or b""
    if isinstance(content, bytes):
        text += content.decode("utf-8", errors="replace").lower()
    text += details_blob
    return "storagequotaexceeded" in text or "do not have storage quota" in text


def _http_error_status_reason_message(exc: HttpError) -> tuple[int, str | None, str]:
    """Parse status, first errors[].reason, and message from a Drive ``HttpError``."""
    resp = getattr(exc, "resp", None)
    status_raw = getattr(resp, "status", None) if resp is not None else None
    try:
        status = int(status_raw) if status_raw is not None else 0
    except (TypeError, ValueError):
        status = 0
    reason: str | None = None
    message = str(exc)
    content = getattr(exc, "content", b"") or b""
    if isinstance(content, bytes):
        try:
            payload = json.loads(content.decode("utf-8"))
            err = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(err, dict):
                message = str(err.get("message") or message)
                errs = err.get("errors")
                if isinstance(errs, list) and errs and isinstance(errs[0], dict):
                    r0 = errs[0].get("reason")
                    if r0 is not None and str(r0).strip():
                        reason = str(r0).strip()
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            pass
    return status, reason, message


def _log_drive_http_client_error_no_traceback(exc: HttpError) -> None:
    status, reason, message = _http_error_status_reason_message(exc)
    if status == 403:
        logger.warning(
            "Drive API HTTP 403 (reason=%r, message=%r). Run continues without Drive upload for this path.",
            reason,
            message,
        )
        return
    logger.warning(
        "Drive API HTTP %s (reason=%r, message=%r). Run continues without Drive upload for this path.",
        status,
        reason,
        message,
    )


def _log_drive_storage_quota_warning_once(*, already_emitted: list[bool]) -> None:
    """Emit a single remediation WARNING per upload attempt (no traceback)."""
    if already_emitted[0]:
        return
    already_emitted[0] = True
    logger.warning(
        "Drive upload failed: service account has no storage quota. Either (1) put drive_root_folder_id "
        "inside a Shared Drive and share it with the service account email as Content Manager, or (2) "
        "configure OAuth delegation. See %s. Run continues.",
        _SHARED_DRIVE_DOC,
    )


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
        from googleapiclient import discovery

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

    def upload_directory(self, local_dir: Path, *, run_id: str) -> str | None:
        local_dir = local_dir.resolve()
        if not local_dir.is_dir():
            raise NotADirectoryError(str(local_dir))

        quota_emit = [False]
        try:
            run_folder_id = self._get_or_create_folder(parent_id=self._root_folder_id, name=run_id)
        except HttpError as exc:
            if _is_sa_my_drive_storage_quota_error(exc):
                _log_drive_storage_quota_warning_once(already_emitted=quota_emit)
            else:
                st, _, _ = _http_error_status_reason_message(exc)
                if st == 403:
                    _log_drive_http_client_error_no_traceback(exc)
                else:
                    logger.warning(
                        "Drive API error creating run folder (%s: %s). Run continues.",
                        type(exc).__name__,
                        exc,
                    )
            return None

        files_uploaded = 0
        total_bytes = 0
        files_failed = 0
        last_file_exc: BaseException | None = None
        abort_quota = False

        try:
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
                    rel_key = str(rel_file).replace(os.sep, "/")
                    try:
                        _, sz = self._upload_file_with_retries(
                            local_path=fp,
                            rel_path=rel_key,
                            parent_folder_id=parent_drive_id,
                        )
                    except BaseException as exc:
                        last_file_exc = exc
                        if _is_sa_my_drive_storage_quota_error(exc):
                            _log_drive_storage_quota_warning_once(already_emitted=quota_emit)
                            abort_quota = True
                            break
                        if isinstance(exc, HttpError):
                            _log_drive_http_client_error_no_traceback(exc)
                            files_failed += 1
                            continue
                        files_failed += 1
                        logger.warning(
                            "Drive upload skipped path=%s after retries: %s: %s. "
                            "Other files will still be attempted; see final summary.",
                            rel_key,
                            type(exc).__name__,
                            exc,
                        )
                        continue
                    files_uploaded += 1
                    total_bytes += int(sz)
                if abort_quota:
                    break
        except HttpError as exc:
            if _is_sa_my_drive_storage_quota_error(exc):
                _log_drive_storage_quota_warning_once(already_emitted=quota_emit)
            else:
                st, _, _ = _http_error_status_reason_message(exc)
                if st == 403:
                    _log_drive_http_client_error_no_traceback(exc)
                else:
                    logger.warning(
                        "Drive API error during upload (%s: %s). Run continues.",
                        type(exc).__name__,
                        exc,
                    )
            return None

        if abort_quota:
            return None

        folder_url = f"https://drive.google.com/drive/folders/{run_folder_id}"
        if files_failed:
            logger.warning(
                "Drive upload incomplete folder_url=%s files_uploaded=%s files_failed=%s bytes=%s",
                folder_url,
                files_uploaded,
                files_failed,
                total_bytes,
            )
        logger.info(
            "Drive upload complete folder_url=%s files_uploaded=%s files_failed=%s bytes=%s",
            folder_url,
            files_uploaded,
            files_failed,
            total_bytes,
        )
        if files_uploaded == 0 and files_failed > 0:
            assert last_file_exc is not None
            if isinstance(last_file_exc, HttpError):
                if _is_sa_my_drive_storage_quota_error(last_file_exc):
                    _log_drive_storage_quota_warning_once(already_emitted=quota_emit)
                    return None
                st, _, _ = _http_error_status_reason_message(last_file_exc)
                if st == 403:
                    _log_drive_http_client_error_no_traceback(last_file_exc)
                    return None
            raise last_file_exc
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

    if _drive_root_preflight_probe(cred_path, root_id, emit_capability_warning=False) is None:
        return None

    rid = run_id if run_id is not None else out_dir.name

    def _sync_upload() -> str | None:
        uploader = DriveUploader(credentials_path=cred_path, root_folder_id=root_id)
        return uploader.upload_directory(out_dir.resolve(), run_id=rid)

    try:
        folder_url = await asyncio.to_thread(_sync_upload)
    except Exception as exc:
        if _is_malformed_service_account_key_error(exc):
            _log_invalid_drive_service_account_key(cred_path=cred_path, detail=str(exc))
            return None
        if isinstance(exc, HttpError):
            if _is_sa_my_drive_storage_quota_error(exc):
                _log_drive_storage_quota_warning_once(already_emitted=[False])
            else:
                _log_drive_http_client_error_no_traceback(exc)
            return None
        logger.warning(
            "Drive upload failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        return None

    if folder_url is None:
        return None

    logger.info("Run uploaded to Drive: %s", folder_url)

    def _sync_record() -> None:
        _append_drive_url_to_run_json(out_dir, folder_url)
        if append_synthesis_footer:
            _append_synthesis_footer(out_dir, folder_url)

    try:
        await asyncio.to_thread(_sync_record)
    except Exception as exc:
        logger.warning(
            "Drive upload succeeded but failed to update run artifacts: %s: %s",
            type(exc).__name__,
            exc,
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
