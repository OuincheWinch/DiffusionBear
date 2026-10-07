import os
import re
import tempfile
import time
from pathlib import Path
from typing import Callable
from urllib.parse import quote, unquote, urlsplit

from huggingface_hub import HfApi

import civitai_service
import app_settings


DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 DiffusionBear/0.1.2"
_HF_HOSTS = {"huggingface.co", "hf.co"}
_REPO_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")


class HFAuthError(RuntimeError):
    pass


def _valid_token(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token or len(token) > 8192 or any(ord(char) < 32 or ord(char) == 127 for char in token):
        return None
    return token


def get_hf_token(token: str | None = None) -> str | None:
    if token is not None and str(token).strip() and _valid_token(token) is None:
        raise ValueError("invalid Hugging Face token")
    resolved = _valid_token(token)
    if resolved:
        return resolved
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    resolved = _valid_token(env_token)
    if resolved:
        return resolved
    # Must be the same file the token route writes to. This resolved
    # Path(__file__).parent / "data", which inside the standalone app is the signed
    # bundle, while POST /api/tokens wrote DATA_DIR -- so the token was saved to
    # Application Support and read from a path nothing had ever written. The UI then
    # reported it unset after every relaunch. civitai_service already resolves this
    # via app_settings; this now matches it.
    try:
        token_file = app_settings.DATA_DIR / "hf_token.txt"
    except Exception:
        token_file = Path(__file__).resolve().parent / "data" / "hf_token.txt"
    if token_file.is_file():
        try:
            resolved = _valid_token(token_file.read_text("utf-8"))
            if resolved:
                return resolved
        except (OSError, UnicodeError):
            pass
    hf_cache_token = Path.home() / ".cache" / "huggingface" / "token"
    if hf_cache_token.is_file():
        try:
            return _valid_token(hf_cache_token.read_text("utf-8"))
        except (OSError, UnicodeError):
            pass
    return None


def _valid_repo_id(repo_id: object) -> str | None:
    if not isinstance(repo_id, str) or len(repo_id) > 193:
        return None
    parts = repo_id.split("/")
    if len(parts) != 2 or any(not _REPO_COMPONENT_RE.fullmatch(part) or part in (".", "..") for part in parts):
        return None
    return repo_id


def _valid_revision(revision: object) -> str | None:
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision) or ".." in revision:
        return None
    return revision


def _valid_filename(filename: object) -> str | None:
    if not isinstance(filename, str) or not filename or len(filename) > 1024:
        return None
    value = unquote(filename).replace("\\", "/").strip("/")
    parts = value.split("/")
    if not value or not value.lower().endswith(".safetensors") or any(part in ("", ".", "..") for part in parts):
        return None
    if any("\x00" in part or any(ord(char) < 32 or ord(char) == 127 for char in part) for part in parts):
        return None
    return value


def parse_hf_input(input_str: str) -> dict | None:
    cleaned = str(input_str or "").strip()
    if not cleaned or len(cleaned) > 4096 or "\x00" in cleaned:
        return None
    if cleaned.lower().startswith(("http://", "https://")):
        try:
            parsed = urlsplit(cleaned)
            port = parsed.port
        except ValueError:
            return None
        host = (parsed.hostname or "").lower()
        if parsed.scheme.lower() != "https" or host not in _HF_HOSTS or port not in (None, 443):
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        path_parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
        if len(path_parts) < 2 or any(part in (".", "..") or "\x00" in part for part in path_parts):
            return None
        repo_id = _valid_repo_id(f"{path_parts[0]}/{path_parts[1]}")
        if repo_id is None:
            return None
        revision = "main"
        filename = None
        if len(path_parts) >= 4 and path_parts[2] in ("resolve", "blob", "raw"):
            revision = _valid_revision(path_parts[3])
            if revision is None:
                return None
            if len(path_parts) >= 5:
                filename = _valid_filename("/".join(path_parts[4:]))
                if filename is None:
                    return None
        elif len(path_parts) >= 4 and path_parts[2] == "tree":
            revision = _valid_revision(path_parts[3])
            if revision is None:
                return None
            if len(path_parts) >= 5:
                filename = _valid_filename("/".join(path_parts[4:]))
                if filename is None:
                    return None
        elif len(path_parts) == 3:
            filename = _valid_filename(path_parts[2])
            if filename is None:
                return None
        elif len(path_parts) > 3:
            return None
        return {"repo_id": repo_id, "filename": filename, "revision": revision, "raw_input": cleaned[:4096]}
    if cleaned.isdigit() or cleaned.startswith(("/", "~", "\\")):
        return None
    if ":" in cleaned:
        repo_part, file_part = cleaned.split(":", 1)
        repo_id = _valid_repo_id(repo_part.strip("/"))
        filename = _valid_filename(file_part)
        if repo_id and filename:
            return {"repo_id": repo_id, "filename": filename, "revision": "main", "raw_input": cleaned[:4096]}
    parts = [part for part in cleaned.strip("/").split("/") if part]
    repo_id = _valid_repo_id(f"{parts[0]}/{parts[1]}") if len(parts) >= 2 else None
    if repo_id is None:
        return None
    filename = None
    if len(parts) == 2:
        pass
    elif len(parts) >= 3:
        filename = _valid_filename("/".join(parts[2:]))
        if filename is None:
            return None
    else:
        return None
    return {"repo_id": repo_id, "filename": filename, "revision": "main", "raw_input": cleaned[:4096]}


def _sibling_hash(sibling) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict):
        value = lfs.get("sha256") or lfs.get("sha256sum")
    else:
        value = getattr(lfs, "sha256", None)
    normalized = civitai_service._normalize_sha256(value)
    if normalized:
        return normalized
    value = getattr(sibling, "lfs_sha256", None) or getattr(sibling, "blob_id", None)
    return civitai_service._normalize_sha256(value)


def fetch_hf_metadata(
    repo_id: str,
    filename: str | None = None,
    revision: str = "main",
    token: str | None = None,
) -> dict:
    repo_id = _valid_repo_id(repo_id)
    revision = _valid_revision(revision) or "main"
    if repo_id is None:
        raise ValueError("Invalid Hugging Face repository id")
    requested_filename = _valid_filename(filename) if filename else None
    if filename and requested_filename is None:
        raise ValueError("Invalid Hugging Face filename")
    resolved_token = get_hf_token(token)
    api = HfApi(token=resolved_token)
    try:
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
    except Exception as e:
        err_msg = str(e).lower()
        if "401" in err_msg or "403" in err_msg or "unauthorized" in err_msg or "gated" in err_msg:
            raise HFAuthError(
                f"Hugging Face repository '{repo_id}' requires authentication or acceptance of license terms."
            ) from e
        if "404" in err_msg or "not found" in err_msg:
            raise ValueError(f"Hugging Face repository '{repo_id}' (revision: {revision}) was not found.") from e
        raise RuntimeError(f"Failed querying Hugging Face repository '{repo_id}'") from e
    siblings = [sibling for sibling in (info.siblings or []) if getattr(sibling, "rfilename", None)]
    safetensors = [sibling for sibling in siblings if str(sibling.rfilename).lower().endswith(".safetensors")]
    if requested_filename:
        exact = [sibling for sibling in safetensors if sibling.rfilename == requested_filename]
        suffix = [sibling for sibling in safetensors if sibling.rfilename.endswith(f"/{requested_filename}")]
        matches = exact or suffix
        if len(matches) != 1:
            raise ValueError(f"Filename '{requested_filename}' was not found uniquely in '{repo_id}'")
        selected = matches[0]
    else:
        if not safetensors:
            raise ValueError(f"No .safetensors files found in Hugging Face repository '{repo_id}'.")
        if len(safetensors) == 1:
            selected = safetensors[0]
        else:
            candidates = [sibling for sibling in safetensors if "lora" in sibling.rfilename.lower()]
            if len(candidates) != 1:
                raise ValueError(f"Multiple .safetensors files found in '{repo_id}'; specify one filename.")
            selected = candidates[0]
    resolved_filename = _valid_filename(selected.rfilename)
    if resolved_filename is None:
        raise ValueError("Hugging Face returned an invalid safetensors filename")
    expected_bytes = int(getattr(selected, "size", 0) or 0)
    if expected_bytes < 0 or expected_bytes > civitai_service.MAX_LORA_DOWNLOAD_BYTES:
        raise ValueError("Hugging Face LoRA exceeds the 8 GB download limit")
    tags = [str(tag).lower() for tag in (info.tags or [])]
    base_model_candidate = next((tag.split(":", 1)[1] for tag in tags if tag.startswith("base_model:")), None)
    if not base_model_candidate:
        if any("sdxl" in tag for tag in tags) or "sdxl" in repo_id.lower() or "sdxl" in resolved_filename.lower():
            base_model_candidate = "sdxl"
        elif any("flux" in tag for tag in tags) or "flux" in repo_id.lower():
            base_model_candidate = "flux2"
        elif "krea" in repo_id.lower():
            base_model_candidate = "krea2"
        elif "z-image" in repo_id.lower() or "zit" in repo_id.lower():
            base_model_candidate = "z-image"
    base_model = civitai_service.normalize_base_model(base_model_candidate)
    triggers: list[str] = []
    card_data = getattr(info, "card_data", None)
    if card_data:
        instance_prompt = getattr(card_data, "instance_prompt", None)
        if isinstance(instance_prompt, list):
            triggers.extend(str(prompt).strip() for prompt in instance_prompt[:20] if str(prompt).strip())
        elif isinstance(instance_prompt, str):
            triggers.extend(part.strip() for part in instance_prompt.split(",")[:50] if part.strip())
        widget = getattr(card_data, "widget", None)
        if isinstance(widget, list):
            for item in widget[:50]:
                if isinstance(item, dict) and "text" in item:
                    triggers.extend(part.strip() for part in str(item["text"]).split(",") if part.strip())
    triggers = list(dict.fromkeys(trigger[:200] for trigger in triggers if trigger))[:5]
    commit_sha = str(getattr(info, "sha", "") or "").strip()
    if not re.fullmatch(r"[A-Fa-f0-9]{40,64}", commit_sha):
        commit_sha = ""
    resolved_revision = commit_sha or revision
    raw_download_url = (
        f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/"
        f"{quote(resolved_revision, safe='/')}/{quote(resolved_filename, safe='/')}"
    )
    return {
        "source": "huggingface",
        "repo_id": repo_id,
        "filename": resolved_filename,
        "revision": revision,
        "resolved_revision": resolved_revision,
        "commit_sha": commit_sha,
        "model_name": repo_id.split("/")[-1],
        "base_model": base_model,
        "triggers": triggers,
        "download_url": raw_download_url,
        "author": repo_id.split("/")[0],
        "expected_sha256": _sibling_hash(selected),
        "expected_bytes": expected_bytes or None,
    }


def _size_matches(actual: int, expected: int | None) -> bool:
    return not expected or abs(actual - expected) <= 4096


def _cached_matches(path: Path, expected_sha256: str | None, expected_bytes: int | None) -> bool:
    if not civitai_service.is_valid_safetensors(path):
        return False
    try:
        if expected_bytes and not _size_matches(path.stat().st_size, expected_bytes):
            return False
        if expected_sha256:
            return civitai_service.compute_file_sha256(path) == expected_sha256
    except OSError:
        return False
    return True


def download_hf_lora(
    repo_id: str,
    filename: str,
    dest_dir: Path,
    revision: str = "main",
    custom_name: str | None = None,
    token: str | None = None,
    progress_cb: Callable | None = None,
    cancel_event: object | None = None,
) -> tuple[Path, dict]:
    dest_dir = Path(dest_dir).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(dest_dir, 0o700)
    except OSError:
        pass
    meta = fetch_hf_metadata(repo_id, filename=filename, revision=revision, token=token)
    resolved_token = get_hf_token(token)
    identity = (meta.get("commit_sha") or meta.get("resolved_revision") or revision)[:12]
    safe_identity = civitai_service._safe_filename(identity, "revision.safetensors").removesuffix(".safetensors")[:20]
    safe_display = civitai_service._safe_filename(custom_name or meta.get("model_name") or repo_id.replace("/", "__"), "lora.safetensors").removesuffix(".safetensors")[:60]
    safe_fname = civitai_service._safe_filename(meta["filename"].rsplit("/", 1)[-1], "lora.safetensors")[:100]
    final_filename = f"{safe_display}__{safe_identity}__{safe_fname}"
    dest_root = dest_dir.resolve()
    dest_path = (dest_root / final_filename).resolve()
    if not dest_path.is_relative_to(dest_root):
        raise ValueError("invalid Hugging Face destination")
    expected_sha256 = meta.get("expected_sha256")
    expected_bytes = meta.get("expected_bytes")
    if _cached_matches(dest_path, expected_sha256, expected_bytes):
        file_size = dest_path.stat().st_size
        civitai_service._emit_progress(progress_cb, "LoRA already present on disk", 1.0, file_size, file_size, 0.0)
        meta["sha256"] = civitai_service.compute_file_sha256(dest_path, cancel_event=cancel_event)
        meta["path"] = str(dest_path)
        return dest_path, meta
    try:
        dest_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        raise RuntimeError("Could not replace the cached Hugging Face LoRA") from e
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/octet-stream"}
    if resolved_token:
        headers["Authorization"] = f"Bearer {resolved_token}"
    civitai_service._emit_progress(progress_cb, f"Connecting to Hugging Face for {meta['model_name']}...", 0.05, 0, 0, 0.0)
    deadline = time.monotonic() + civitai_service.download_deadline_seconds()
    fd = None
    tmp_file = None
    downloaded = 0
    total_bytes = 0
    try:
        response, final_url = civitai_service.open_public_https_stream(
            meta["download_url"],
            headers=headers,
            auth_hosts=_HF_HOSTS,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        with response:
            content_type = str(response.headers.get("Content-Type", "")).lower()
            final_path = urlsplit(final_url).path.lower().rstrip("/")
            if response.status_code in (401, 403) or final_path.endswith("/login") or final_path == "/login":
                raise HFAuthError(f"Hugging Face denied access to '{repo_id}' (HTTP {response.status_code}).")
            if response.status_code == 404:
                raise ValueError(f"File '{filename}' not found in Hugging Face repository '{repo_id}'.")
            if response.status_code != 200:
                raise RuntimeError(f"Hugging Face download failed with HTTP {response.status_code}")
            if content_type.startswith("text/html"):
                raise HFAuthError("Hugging Face requires authentication for this file.")
            if content_type.startswith("application/json"):
                raise ValueError("Hugging Face returned a non-binary response")
            total_bytes = civitai_service._content_length(response.headers)
            if total_bytes > civitai_service.MAX_LORA_DOWNLOAD_BYTES:
                raise ValueError("Hugging Face LoRA exceeds the 8 GB download limit")
            if expected_bytes and total_bytes and not _size_matches(total_bytes, expected_bytes):
                raise ValueError("Hugging Face download size does not match model metadata")
            fd, tmp_file = tempfile.mkstemp(dir=str(dest_dir), prefix=".hf-", suffix=".download")
            with os.fdopen(fd, "wb") as out_f:
                fd = None
                start_time = time.monotonic()
                last_speed_time = start_time
                last_speed_bytes = 0
                current_speed_mb = 0.0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if cancel_event is not None and cancel_event.is_set():
                        raise civitai_service.DownloadCancelled("Download cancelled by user")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Hugging Face download deadline exceeded")
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > civitai_service.MAX_LORA_DOWNLOAD_BYTES:
                        raise ValueError("Hugging Face LoRA exceeds the 8 GB download limit")
                    if expected_bytes and not _size_matches(downloaded, expected_bytes):
                        raise ValueError("Hugging Face download size does not match model metadata")
                    out_f.write(chunk)
                    now = time.monotonic()
                    dt = now - last_speed_time
                    if dt >= 0.5:
                        current_speed_mb = ((downloaded - last_speed_bytes) / (1 << 20)) / dt
                        last_speed_time = now
                        last_speed_bytes = downloaded
                    frac = min(0.95, downloaded / total_bytes) if total_bytes > 0 else 0.5
                    status_text = (
                        f"Downloading {meta['model_name']}: {downloaded >> 20}MB / {total_bytes >> 20}MB "
                        f"({frac * 100:.0f}%) - {current_speed_mb:.1f} MB/s"
                    ) if total_bytes > 0 else f"Downloading: {downloaded >> 20}MB - {current_speed_mb:.1f} MB/s"
                    civitai_service._emit_progress(progress_cb, status_text, frac, downloaded, total_bytes, current_speed_mb)
                out_f.flush()
                os.fsync(out_f.fileno())
        if total_bytes and downloaded != total_bytes:
            raise ValueError("Hugging Face download ended before the expected byte count")
        if expected_bytes and not _size_matches(downloaded, expected_bytes):
            raise ValueError("Hugging Face download size does not match model metadata")
        if not civitai_service.is_valid_safetensors(tmp_file):
            raise ValueError("Downloaded Hugging Face payload is not a valid .safetensors archive")
        civitai_service._emit_progress(progress_cb, "Computing SHA-256 digest...", 0.98, downloaded, total_bytes, 0.0)
        sha = civitai_service.compute_file_sha256(tmp_file, cancel_event=cancel_event)
        if expected_sha256 and sha != expected_sha256:
            raise ValueError("Hugging Face LoRA SHA-256 does not match model metadata")
        if cancel_event is not None and cancel_event.is_set():
            raise civitai_service.DownloadCancelled("Download cancelled by user")
        os.replace(tmp_file, dest_path)
        tmp_file = None
        try:
            os.chmod(dest_path, 0o600)
        except OSError:
            pass
    except (HFAuthError, civitai_service.DownloadCancelled):
        raise
    except Exception as e:
        raise RuntimeError("Failed downloading LoRA from Hugging Face") from e
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_file:
            try:
                os.unlink(tmp_file)
            except OSError:
                pass
    meta["sha256"] = sha
    meta["path"] = str(dest_path)
    civitai_service._emit_progress(progress_cb, "LoRA ready!", 1.0, downloaded, total_bytes, 0.0)
    return dest_path, meta
