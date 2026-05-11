from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal, cast

from google.auth.exceptions import MalformedError, RefreshError, TransportError
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
from googleapiclient.http import MediaFileUpload  # type: ignore[import-untyped]

from equity_analyst.config import (
    RunConfig,
    RunEnvironment,
    resolve_drive_oauth_token_path_from_optional,
)

logger = logging.getLogger(__name__)

_DRIVE_SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/drive",)
_OAUTH_DRIVE_SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/drive",)

DriveAuthMode = Literal["service_account", "oauth_user"]


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


def _build_drive_service_from_oauth_creds(creds: Credentials) -> Any:
    from googleapiclient import discovery

    return discovery.build("drive", "v3", credentials=creds, cache_discovery=False)


def _oauth_granted_scope_set(*, token_path: Path, creds: Credentials) -> frozenset[str]:
    """Scopes granted when the user consented (from the token file / loaded credentials)."""
    raw = creds.scopes
    if raw:
        return frozenset(str(s) for s in raw)
    try:
        data = json.loads(token_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return frozenset()
    sc = data.get("scopes")
    if isinstance(sc, str):
        return frozenset(s for s in sc.split() if s)
    if isinstance(sc, list):
        return frozenset(str(s) for s in sc if str(s).strip())
    return frozenset()


def _oauth_scopes_satisfy_app_requirement(granted: frozenset[str]) -> bool:
    required = frozenset(_OAUTH_DRIVE_SCOPES)
    if not required:
        return True
    if not granted:
        return True
    return required <= granted


def _load_oauth_user_credentials(token_path: Path) -> tuple[Credentials | None, str | None]:
    """Return ``(credentials, None)`` or ``(None, error_tag)``.

    ``error_tag`` is ``load_failed``, ``no_refresh``, ``refresh_failed``, or ``scope_mismatch``.

    Loads without overriding on-disk ``scopes`` so we can detect tokens authorized under a
    narrower scope than :data:`_OAUTH_DRIVE_SCOPES` (e.g. after widening the app's requirement).
    """
    try:
        creds = Credentials.from_authorized_user_file(str(token_path))  # type: ignore[no-untyped-call]
    except (ValueError, OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
        logger.debug("OAuth token load failed: %s: %s", type(exc).__name__, exc)
        return None, "load_failed"
    if creds.refresh_token is None and not creds.valid:
        return None, "no_refresh"
    if not creds.valid:
        try:
            creds.refresh(Request())  # type: ignore[no-untyped-call]
        except RefreshError:
            return None, "refresh_failed"
        except (TransportError, OSError, ValueError, TypeError) as exc:
            logger.debug("OAuth token refresh error: %s: %s", type(exc).__name__, exc)
            return None, "refresh_failed"
    if not creds.valid:
        return None, "no_refresh"
    granted = _oauth_granted_scope_set(token_path=token_path, creds=creds)
    if granted and not _oauth_scopes_satisfy_app_requirement(granted):
        req = sorted(_OAUTH_DRIVE_SCOPES)
        got = sorted(granted)
        logger.warning(
            "OAuth token at %s was authorized with scopes %s but app now requires %s. "
            "Re-run `python -m equity_analyst.drive_oauth_setup` to re-consent. "
            "Disabling Drive upload for this run.",
            token_path,
            got,
            req,
        )
        return None, "scope_mismatch"
    return creds, None


def _oauth_user_email_from_service(svc: Any) -> str | None:
    try:
        raw = svc.about().get(fields="user(emailAddress)").execute()
        if not isinstance(raw, dict):
            return None
        user = raw.get("user")
        if isinstance(user, dict):
            em = user.get("emailAddress")
            if em is not None and str(em).strip():
                return str(em).strip()
    except (HttpError, TransportError, TypeError, AttributeError, ValueError):
        return None
    return None


def _log_oauth_drive_permission_hint(*, root_id: str) -> None:
    logger.warning(
        "Drive upload (OAuth): access denied or folder not found for drive_root_folder_id=%s. "
        "Confirm the folder exists in the signed-in Google account and that the id matches your Drive URL. "
        "Run continues without Drive upload for this path.",
        root_id,
        exc_info=False,
    )


def _drive_root_folder_metadata_or_log(
    svc: Any,
    root_id: str,
    *,
    auth_mode: DriveAuthMode,
) -> dict[str, Any] | None:
    """Return folder metadata dict or None; logs transport/API errors without traceback."""
    meta: dict[str, Any]
    try:
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
                exc_info=False,
            )
            return None
        meta = cast(dict[str, Any], raw_meta)
    except HttpError as exc:
        if auth_mode == "oauth_user":
            st, _, _ = _http_error_status_reason_message(exc)
            if st == 403:
                _log_oauth_drive_permission_hint(root_id=root_id)
            else:
                _log_drive_http_client_error_no_traceback(exc)
        elif _is_sa_my_drive_storage_quota_error(exc):
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
            exc_info=False,
        )
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Drive upload: unexpected preflight error for folder id=%s (%s: %s). "
            "Disabling Drive upload for this run.",
            root_id,
            type(exc).__name__,
            exc,
            exc_info=False,
        )
        return None

    mime = meta.get("mimeType")
    if mime != _FOLDER_MIMETYPE:
        logger.warning(
            "Drive upload: drive_root_folder_id %s points to mimeType=%r (expected a folder). "
            "Disabling Drive upload for this run.",
            root_id,
            mime,
            exc_info=False,
        )
        return None

    return meta


def _emit_folder_capability_warning_if_needed(
    *,
    meta: dict[str, Any],
    root_id: str,
    emit_capability_warning: bool,
    auth_mode: DriveAuthMode,
) -> None:
    raw_caps = meta.get("capabilities")
    caps: dict[str, Any] = cast(dict[str, Any], raw_caps) if isinstance(raw_caps, dict) else {}
    can_add = caps.get("canAddChildren")
    if not emit_capability_warning or can_add is True:
        return
    if auth_mode == "service_account":
        logger.warning(
            "Drive upload: service account may lack Content Manager access on folder %r (%s): "
            "capabilities.canAddChildren is not true. Grant the service account Content Manager on this "
            "folder (or the Shared Drive) so uploads can create subfolders and files.",
            meta.get("name") or "?",
            root_id,
            exc_info=False,
        )
    else:
        logger.warning(
            "Drive upload (OAuth): folder %r (%s) reports capabilities.canAddChildren is not true; "
            "uploads may fail if this account cannot add files here.",
            meta.get("name") or "?",
            root_id,
            exc_info=False,
        )


def _drive_root_preflight_probe(
    cred_path: Path,
    root_id: str,
    *,
    emit_capability_warning: bool,
) -> dict[str, Any] | None:
    """Fetch root folder metadata for **service account** auth; return metadata dict on success, else None.

    The personal-drive WARNING is emitted at most once per process per ``(cred_path, root_id)``.
    """
    key = (str(cred_path.resolve()), root_id)
    svc = _build_drive_service(cred_path)
    meta = _drive_root_folder_metadata_or_log(svc, root_id, auth_mode="service_account")
    if meta is None:
        return None

    if not meta.get("driveId"):
        if key not in _PERSONAL_DRIVE_WARNED_KEYS:
            _PERSONAL_DRIVE_WARNED_KEYS.add(key)
            logger.warning(
                "Drive folder %s is not inside a Shared Drive. Service-account uploads will fail with "
                "storageQuotaExceeded. Move the folder into a Shared Drive (Workspace) or switch to "
                "OAuth user mode (drive_auth_mode: oauth_user). Disabling Drive upload for this run.",
                root_id,
                exc_info=False,
            )
        return None

    _emit_folder_capability_warning_if_needed(
        meta=meta,
        root_id=root_id,
        emit_capability_warning=emit_capability_warning,
        auth_mode="service_account",
    )

    return meta


def _drive_root_preflight_probe_oauth(
    creds: Credentials,
    root_id: str,
    *,
    emit_capability_warning: bool,
) -> dict[str, Any] | None:
    """Preflight for OAuth user credentials (no Shared Drive requirement)."""
    svc = _build_drive_service_from_oauth_creds(creds)
    meta = _drive_root_folder_metadata_or_log(svc, root_id, auth_mode="oauth_user")
    if meta is None:
        return None
    _emit_folder_capability_warning_if_needed(
        meta=meta,
        root_id=root_id,
        emit_capability_warning=emit_capability_warning,
        auth_mode="oauth_user",
    )
    return meta


def log_drive_upload_plan(
    *,
    drive_upload_enabled: bool,
    drive_credentials_path: str | None,
    drive_root_folder_id: str | None,
    drive_auth_mode: DriveAuthMode = "service_account",
    drive_oauth_token_path: str | None = None,
) -> bool:
    """Log whether Drive upload will run and why; call once per run near startup.

    Returns True if uploads should proceed for this run (preflight passed).
    """
    if not drive_upload_enabled:
        logger.info(
            "Drive upload: DISABLED (reason=config: drive_upload_enabled is false)",
        )
        return False

    root_id = (drive_root_folder_id or "").strip()
    if not root_id:
        logger.warning(
            "Drive upload: DISABLED (reason=no drive_root_folder_id; set drive_root_folder_id in YAML "
            "or DRIVE_ROOT_FOLDER_ID to the Drive folder id)",
        )
        return False

    if drive_auth_mode == "oauth_user":
        token_path = resolve_drive_oauth_token_path_from_optional(drive_oauth_token_path)
        if not token_path.is_file():
            logger.warning(
                "Drive upload: DISABLED (reason=oauth token file missing at %s; run python -m "
                "equity_analyst.drive_oauth_setup to authorize)",
                token_path,
            )
            return False
        loaded, err = _load_oauth_user_credentials(token_path)
        if loaded is None:
            if err == "scope_mismatch":
                pass  # already logged in _load_oauth_user_credentials
            elif err in ("refresh_failed", "no_refresh"):
                logger.warning(
                    "Drive upload: DISABLED (reason=oauth token expired/revoked; re-run python -m "
                    "equity_analyst.drive_oauth_setup)",
                )
            else:
                logger.warning(
                    "Drive upload: DISABLED (reason=oauth token invalid or unreadable at %s; run python -m "
                    "equity_analyst.drive_oauth_setup to authorize)",
                    token_path,
                )
            return False

        meta = _drive_root_preflight_probe_oauth(
            loaded,
            root_id,
            emit_capability_warning=True,
        )
        if meta is None:
            return False

        svc = _build_drive_service_from_oauth_creds(loaded)
        email = _oauth_user_email_from_service(svc) or "unknown"
        logger.info(
            "Drive upload: ENABLED (auth=oauth_user, folder=%s, account=%s)",
            root_id,
            email,
        )
        return True

    cred_path = _expand_credentials_path(drive_credentials_path)
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
        "Drive upload: ENABLED (auth=service_account, folder=%s, shared_drive=%s)",
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
        drive_auth_mode=cfg.drive_auth_mode,
        drive_oauth_token_path=cfg.drive_oauth_token_path,
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

_SKIP_FILENAME_PATTERNS: tuple[str, ...] = (
    "checkpoint.sqlite",
    "checkpoint.sqlite-wal",
    "checkpoint.sqlite-shm",
    "checkpoint.sqlite-journal",
)
_SKIP_FILENAME_SET: frozenset[str] = frozenset(_SKIP_FILENAME_PATTERNS)

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


def drive_upload_child_folder_name(run_environment: RunEnvironment) -> str:
    """Lowercase Drive folder segment: ``prod`` for production runs, ``test`` for test runs."""
    return "prod" if run_environment == "production" else "test"


def resolve_drive_upload_parent_folder_id(service: Any, root_folder_id: str, environment: str) -> str:
    """Return the Drive folder id for ``prod`` or ``test`` under ``root_folder_id``, creating it if absent.

    ``environment`` must be ``\"production\"`` or ``\"test\"`` (maps to child folder names ``prod`` / ``test``).

    If ``files().create`` fails (for example two concurrent creators), the folder is listed again and an
    existing match is returned when found.
    """
    if environment not in ("production", "test"):
        raise ValueError(f"environment must be 'production' or 'test', not {environment!r}")
    folder_name = drive_upload_child_folder_name(cast(RunEnvironment, environment))
    root = root_folder_id.strip()
    q = (
        f"'{root}' in parents and "
        f"mimeType = 'application/vnd.google-apps.folder' and "
        f"name = '{_drive_query_escape_name(folder_name)}' and trashed = false"
    )

    def _list_match() -> str | None:
        res = (
            service.files()
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

    found = _list_match()
    if found is not None:
        logger.info(
            "Drive upload: resolved environment=%s upload_parent=%s (folder name %s under root %s)",
            environment,
            found,
            folder_name,
            root,
        )
        return found

    body: dict[str, Any] = {
        "name": folder_name,
        "mimeType": _FOLDER_MIMETYPE,
        "parents": [root],
    }
    try:
        created = (
            service.files()
            .create(body=body, fields="id", supportsAllDrives=True)
            .execute()
        )
        folder_id = str(created["id"])
    except HttpError:
        found2 = _list_match()
        if found2 is not None:
            logger.info(
                "Drive upload: resolved environment=%s upload_parent=%s (folder name %s under root %s)",
                environment,
                found2,
                folder_name,
                root,
            )
            return found2
        raise

    logger.info(
        "Drive upload: resolved environment=%s upload_parent=%s (folder name %s under root %s)",
        environment,
        folder_id,
        folder_name,
        root,
    )
    return folder_id


class DriveUploader:
    """Upload a local directory tree to Google Drive (service account or OAuth user)."""

    def __init__(
        self,
        credentials_path: str | Path | None,
        root_folder_id: str,
        *,
        auth_mode: DriveAuthMode = "service_account",
        oauth_token_path: str | Path | None = None,
        run_environment: RunEnvironment = "production",
    ) -> None:
        self._auth_mode: DriveAuthMode = auth_mode
        self._credentials_path = Path(credentials_path) if credentials_path is not None else None
        self._oauth_token_path = (
            Path(oauth_token_path) if oauth_token_path is not None else None
        )
        self._root_folder_id = root_folder_id.strip()
        self._run_environment: RunEnvironment = run_environment
        self._service: Any = None
        self._cached_upload_parent_folder_id: str | None = None
        self.last_resolved_upload_parent_folder_id: str | None = None
        self.last_resolved_upload_parent_folder_name: str | None = None

    def _ensure_service(self) -> Any:
        if self._service is not None:
            return self._service
        from googleapiclient import discovery

        if self._auth_mode == "service_account":
            if self._credentials_path is None:
                raise TypeError("service_account mode requires credentials_path")
            creds = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
                str(self._credentials_path),
                scopes=_DRIVE_SCOPES,
            )
        else:
            if self._oauth_token_path is None:
                raise TypeError("oauth_user mode requires oauth_token_path")
            loaded, _err = _load_oauth_user_credentials(self._oauth_token_path)
            if loaded is None:
                raise RuntimeError("OAuth credentials are missing or invalid")
            creds = loaded

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
        if self._cached_upload_parent_folder_id is None:
            self._cached_upload_parent_folder_id = resolve_drive_upload_parent_folder_id(
                self._ensure_service(),
                self._root_folder_id,
                self._run_environment,
            )
        upload_parent_id = self._cached_upload_parent_folder_id
        self.last_resolved_upload_parent_folder_id = upload_parent_id
        self.last_resolved_upload_parent_folder_name = drive_upload_child_folder_name(self._run_environment)
        try:
            run_folder_id = self._get_or_create_folder(parent_id=upload_parent_id, name=run_id)
        except HttpError as exc:
            if self._auth_mode == "oauth_user":
                st, _, _ = _http_error_status_reason_message(exc)
                if st == 403:
                    _log_oauth_drive_permission_hint(root_id=self._root_folder_id)
                else:
                    _log_drive_http_client_error_no_traceback(exc)
            elif _is_sa_my_drive_storage_quota_error(exc):
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
                        exc_info=False,
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
                    if fname in _SKIP_FILENAME_SET:
                        fp_skip = Path(root) / fname
                        logger.info(
                            "Drive upload: skipping checkpoint file %s",
                            str(fp_skip.resolve()),
                        )
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
                            if self._auth_mode == "oauth_user":
                                st, _, _ = _http_error_status_reason_message(exc)
                                if st == 403:
                                    _log_oauth_drive_permission_hint(root_id=self._root_folder_id)
                                else:
                                    _log_drive_http_client_error_no_traceback(exc)
                            else:
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
                            exc_info=False,
                        )
                        continue
                    files_uploaded += 1
                    total_bytes += int(sz)
                if abort_quota:
                    break
        except HttpError as exc:
            if self._auth_mode == "oauth_user":
                st, _, _ = _http_error_status_reason_message(exc)
                if st == 403:
                    _log_oauth_drive_permission_hint(root_id=self._root_folder_id)
                else:
                    _log_drive_http_client_error_no_traceback(exc)
            elif _is_sa_my_drive_storage_quota_error(exc):
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
                        exc_info=False,
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
                exc_info=False,
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


def _append_drive_url_to_run_json(
    out_dir: Path,
    folder_url: str,
    *,
    run_environment: str,
    drive_upload_parent_folder_id: str,
    drive_upload_parent_folder_name: str,
) -> None:
    run_json = out_dir / "run.json"
    if not run_json.is_file():
        return
    meta = json.loads(run_json.read_text(encoding="utf-8"))
    meta["drive_folder_url"] = folder_url
    meta["run_environment"] = run_environment
    meta["drive_upload_parent_folder_id"] = drive_upload_parent_folder_id
    meta["drive_upload_parent_folder_name"] = drive_upload_parent_folder_name
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
    drive_auth_mode: DriveAuthMode = "service_account",
    drive_oauth_token_path: str | None = None,
    run_environment: RunEnvironment = "production",
) -> str | None:
    """Optionally upload `out_dir` to Drive; never raises. Returns folder URL or None."""
    if not drive_upload_enabled:
        return None

    root_id = (drive_root_folder_id or "").strip()
    if not root_id:
        logger.warning(
            "Drive upload skipped: drive_upload_enabled=true but drive_root_folder_id is missing. "
            "Set DRIVE_ROOT_FOLDER_ID or drive_root_folder_id.",
        )
        return None

    rid = run_id if run_id is not None else out_dir.name

    if drive_auth_mode == "oauth_user":
        token_path = resolve_drive_oauth_token_path_from_optional(drive_oauth_token_path)
        if not token_path.is_file():
            logger.warning(
                "Drive upload skipped: oauth token file missing at %s. Run python -m "
                "equity_analyst.drive_oauth_setup to authorize.",
                token_path,
            )
            return None
        loaded, err = _load_oauth_user_credentials(token_path)
        if loaded is None:
            if err == "scope_mismatch":
                pass  # already logged in _load_oauth_user_credentials
            elif err in ("refresh_failed", "no_refresh"):
                logger.warning(
                    "Drive upload skipped: OAuth token expired or revoked; re-run python -m "
                    "equity_analyst.drive_oauth_setup.",
                )
            else:
                logger.warning(
                    "Drive upload skipped: OAuth token invalid or unreadable at %s. Run python -m "
                    "equity_analyst.drive_oauth_setup to authorize.",
                    token_path,
                )
            return None
        if _drive_root_preflight_probe_oauth(loaded, root_id, emit_capability_warning=False) is None:
            return None

        last_uploader: list[DriveUploader | None] = [None]

        def _sync_upload_oauth() -> str | None:
            uploader = DriveUploader(
                None,
                root_id,
                auth_mode="oauth_user",
                oauth_token_path=token_path,
                run_environment=run_environment,
            )
            last_uploader[0] = uploader
            return uploader.upload_directory(out_dir.resolve(), run_id=rid)

        try:
            folder_url = await asyncio.to_thread(_sync_upload_oauth)
        except Exception as exc:
            if isinstance(exc, HttpError):
                st, _, _ = _http_error_status_reason_message(exc)
                if st == 403:
                    _log_oauth_drive_permission_hint(root_id=root_id)
                else:
                    _log_drive_http_client_error_no_traceback(exc)
                return None
            logger.warning(
                "Drive upload failed: %s: %s",
                type(exc).__name__,
                exc,
                exc_info=False,
            )
            return None

        if folder_url is None:
            return None

        final_folder_url: str = folder_url
        logger.info("Run uploaded to Drive: %s", final_folder_url)

        up = last_uploader[0]
        parent_folder_id = (
            (up.last_resolved_upload_parent_folder_id or "") if up is not None else ""
        )
        parent_folder_name = (
            (up.last_resolved_upload_parent_folder_name or drive_upload_child_folder_name(run_environment))
            if up is not None
            else drive_upload_child_folder_name(run_environment)
        )

        def _sync_record_oauth() -> None:
            _append_drive_url_to_run_json(
                out_dir,
                final_folder_url,
                run_environment=run_environment,
                drive_upload_parent_folder_id=parent_folder_id,
                drive_upload_parent_folder_name=parent_folder_name,
            )
            if append_synthesis_footer:
                _append_synthesis_footer(out_dir, final_folder_url)

        try:
            await asyncio.to_thread(_sync_record_oauth)
        except Exception as exc:
            logger.warning(
                "Drive upload succeeded but failed to update run artifacts: %s: %s",
                type(exc).__name__,
                exc,
                exc_info=False,
            )

        return final_folder_url

    cred_path = _expand_credentials_path(drive_credentials_path)
    if cred_path is None or not cred_path.is_file():
        logger.warning(
            "Drive upload skipped: drive_upload_enabled=true but drive_credentials_path is missing "
            "or not a readable file (resolved=%s). Set DRIVE_CREDENTIALS_PATH or drive_credentials_path.",
            str(cred_path) if cred_path is not None else None,
        )
        return None

    issue = _service_account_key_file_issue(cred_path)
    if issue is not None:
        _log_invalid_drive_service_account_key(cred_path=cred_path, detail=issue)
        return None

    if _drive_root_preflight_probe(cred_path, root_id, emit_capability_warning=False) is None:
        return None

    last_uploader_sa: list[DriveUploader | None] = [None]

    def _sync_upload() -> str | None:
        uploader = DriveUploader(
            cred_path,
            root_id,
            auth_mode="service_account",
            run_environment=run_environment,
        )
        last_uploader_sa[0] = uploader
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
            exc_info=False,
        )
        return None

    if folder_url is None:
        return None

    logger.info("Run uploaded to Drive: %s", folder_url)

    up_sa = last_uploader_sa[0]
    parent_folder_id_sa = (
        (up_sa.last_resolved_upload_parent_folder_id or "") if up_sa is not None else ""
    )
    parent_folder_name_sa = (
        (up_sa.last_resolved_upload_parent_folder_name or drive_upload_child_folder_name(run_environment))
        if up_sa is not None
        else drive_upload_child_folder_name(run_environment)
    )

    def _sync_record() -> None:
        _append_drive_url_to_run_json(
            out_dir,
            folder_url,
            run_environment=run_environment,
            drive_upload_parent_folder_id=parent_folder_id_sa,
            drive_upload_parent_folder_name=parent_folder_name_sa,
        )
        if append_synthesis_footer:
            _append_synthesis_footer(out_dir, folder_url)

    try:
        await asyncio.to_thread(_sync_record)
    except Exception as exc:
        logger.warning(
            "Drive upload succeeded but failed to update run artifacts: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=False,
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
        drive_auth_mode=cfg.drive_auth_mode,
        drive_oauth_token_path=cfg.drive_oauth_token_path,
        run_environment=cfg.run_environment,
    )
