import hashlib
import ipaddress
import json
import math
import os
import re
import socket
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import requests


CIVITAI_API_BASE = "https://civitai.com/api/v1"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 DiffusionBear/0.1.2"
MAX_LORA_DOWNLOAD_BYTES = 8 * (1 << 30)
MAX_API_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_DOWNLOAD_REDIRECTS = 5
DOWNLOAD_CONNECT_TIMEOUT_SECONDS = 10
DOWNLOAD_READ_TIMEOUT_SECONDS = 30
DEFAULT_DOWNLOAD_DEADLINE_SECONDS = 3600
_CIVITAI_AUTH_HOSTS = {"civitai.com", "www.civitai.com", "civitai.red", "www.civitai.red"}
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SHA256_RE = re.compile(r"^[A-Fa-f0-9]{64}$")


class CivitaiAuthError(RuntimeError):
    pass


class DownloadCancelled(RuntimeError):
    pass


def download_deadline_seconds() -> int:
    raw = os.environ.get(
        "DIFFUSIONBEAR_DOWNLOAD_TIMEOUT_S",
        os.environ.get("DIFFUSIONBEAR_DOWNLOAD_TIMEOUT_S", str(DEFAULT_DOWNLOAD_DEADLINE_SECONDS)),
    )
    try:
        return min(86400, max(60, int(raw)))
    except (TypeError, ValueError):
        return DEFAULT_DOWNLOAD_DEADLINE_SECONDS


def _valid_token(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token or len(token) > 8192 or any(ord(char) < 32 or ord(char) == 127 for char in token):
        return None
    return token


def _safe_filename(value: object, fallback: str = "download.safetensors") -> str:
    raw = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = _SAFE_FILENAME_RE.sub("_", raw).strip("._")[:160]
    if not cleaned:
        cleaned = _SAFE_FILENAME_RE.sub("_", fallback).strip("._")[:160] or "download"
    if not cleaned.lower().endswith(".safetensors"):
        cleaned = f"{cleaned[:150]}.safetensors"
    return cleaned[:180]


def _is_public_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return address.is_global and not address.is_multicast
    except ValueError:
        return False


def validate_public_https_url(url: str) -> str:
    if not isinstance(url, str) or not url or len(url) > 8192:
        raise ValueError("invalid download URL")
    if "\\" in url or any(ord(char) < 32 or ord(char) == 127 or char.isspace() for char in url):
        raise ValueError("invalid download URL")
    try:
        parsed = urlsplit(url.strip())
        port = parsed.port
    except ValueError as e:
        raise ValueError("invalid download URL") from e
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("download URL must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("download URL credentials are not allowed")
    try:
        host = parsed.hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as e:
        raise ValueError("invalid download hostname") from e
    if not host or len(host) > 253 or host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("download host is not public")
    port = port or 443
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise ValueError("download hostname could not be resolved") from e
    if not addresses or any(not _is_public_ip(str(item[4][0])) for item in addresses):
        raise ValueError("download host resolves to a non-public address")
    display_host = f"[{host}]" if ":" in host else host
    netloc = display_host if port == 443 else f"{display_host}:{port}"
    return urlunsplit(("https", netloc, parsed.path or "/", parsed.query, ""))


def open_public_https_stream(
    url: str,
    headers: dict[str, str] | None = None,
    auth_hosts: set[str] | None = None,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
):
    current_url = url
    base_headers = dict(headers or {})
    base_headers.setdefault("Accept-Encoding", "identity")
    session = requests.Session()
    session.trust_env = False
    try:
        for _ in range(MAX_DOWNLOAD_REDIRECTS + 1):
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelled("Download cancelled by user")
            current_url = validate_public_https_url(current_url)
            remaining = max(1.0, deadline - time.monotonic()) if deadline is not None else float(DOWNLOAD_READ_TIMEOUT_SECONDS)
            request_headers = dict(base_headers)
            hostname = urlsplit(current_url).hostname or ""
            if auth_hosts is not None and hostname.lower() not in auth_hosts:
                request_headers.pop("Authorization", None)
                request_headers.pop("Cookie", None)
                request_headers.pop("Proxy-Authorization", None)
            response = session.get(
                current_url,
                headers=request_headers,
                stream=True,
                allow_redirects=False,
                timeout=(min(DOWNLOAD_CONNECT_TIMEOUT_SECONDS, remaining), min(DOWNLOAD_READ_TIMEOUT_SECONDS, remaining)),
            )
            peer_is_public = _response_peer_is_public(response)
            redirect_status = response.status_code in (301, 302, 303, 307, 308)
            if peer_is_public is False or (peer_is_public is None and not redirect_status):
                response.close()
                raise ValueError("download host connected to an unverified address")
            if not redirect_status:
                return response, current_url
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise ValueError("download redirect did not include a destination")
            current_url = urljoin(current_url, location)
        raise ValueError("too many download redirects")
    finally:
        session.close()


def _response_peer_is_public(response) -> bool | None:
    try:
        connection = getattr(getattr(response, "raw", None), "_connection", None)
        sock = getattr(connection, "sock", None)
        peer = sock.getpeername()[0] if sock is not None else None
        if peer is None:
            return None
        return _is_public_ip(str(peer))
    except Exception:
        return None


def _content_length(headers) -> int:
    value = headers.get("Content-Length")
    if value is None:
        return 0
    try:
        length = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, length)


def _size_matches(actual: int, expected: int | None) -> bool:
    return not expected or abs(actual - expected) <= 4096


def _normalize_sha256(value: object) -> str | None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.strip()):
        return None
    return value.strip().upper()


def is_valid_safetensors(path: Path | str) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    try:
        size = p.stat().st_size
        if size < 16:
            return False
        with open(p, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return False
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            if header_len <= 0 or header_len >= size - 8 or header_len > 100 * 1024 * 1024:
                return False
            header_bytes = f.read(header_len)
            if header_bytes.lstrip().lower().startswith((b"<!doctype", b"<html")):
                return False
            header = json.loads(header_bytes.decode("utf-8"))
            if not isinstance(header, dict):
                return False
            data_size = size - 8 - header_len
            tensor_count = 0
            for name, tensor in header.items():
                if name == "__metadata__":
                    continue
                if not isinstance(tensor, dict):
                    return False
                offsets = tensor.get("data_offsets")
                if not isinstance(offsets, list) or len(offsets) != 2:
                    return False
                start, end = offsets
                if (
                    not isinstance(start, int)
                    or isinstance(start, bool)
                    or not isinstance(end, int)
                    or isinstance(end, bool)
                    or start < 0
                    or end < start
                    or end > data_size
                ):
                    return False
                tensor_count += 1
            if tensor_count == 0:
                return False
        return True
    except Exception:
        return False


def get_civitai_api_key(api_key: str | None = None) -> str | None:
    if api_key is not None and str(api_key).strip() and _valid_token(api_key) is None:
        raise ValueError("invalid Civitai API key")
    resolved = _valid_token(api_key)
    if resolved:
        return resolved
    env_key = os.environ.get("CIVITAI_API_KEY") or os.environ.get("CIVITAI_TOKEN")
    resolved = _valid_token(env_key)
    if resolved:
        return resolved
    # Read the token from DATA_DIR, which is where routers/tokens.py WRITES it (the
    # Parameters -> Secrets UI). Reading it from __file__/data only worked in a source
    # checkout, where the two happen to be the same directory. In the standalone app they
    # are not -- the bundle's data/ is excluded by build_app.sh -- so the token the user
    # saved through the UI was never found here, and every gated Civitai download would
    # have failed with 401. Verified: auth=False from the installed app before this change.
    try:
        import app_settings

        token_file = app_settings.DATA_DIR / "civitai_token.txt"
    except Exception:
        token_file = Path(__file__).resolve().parent / "data" / "civitai_token.txt"
    if token_file.is_file():
        try:
            return _valid_token(token_file.read_text("utf-8"))
        except (OSError, UnicodeError):
            pass
    return None


def compute_file_sha256(
    path: Path | str,
    chunk_size: int = 4 * 1024 * 1024,
    cancel_event: threading.Event | None = None,
) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hash calculation: {p.name}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelled("Download cancelled by user")
            h.update(chunk)
    return h.hexdigest().upper()


def _request_json(url: str, timeout: int = 8, api_key: str | None = None) -> dict | None:
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/json",
    }
    resolved_key = get_civitai_api_key(api_key)
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"
    try:
        response, _ = open_public_https_stream(
            url,
            headers=headers,
            auth_hosts=_CIVITAI_AUTH_HOSTS,
            deadline=time.monotonic() + max(1, timeout),
        )
        with response:
            if response.status_code == 404:
                return None
            if response.status_code != 200:
                print(f"[civitai] HTTP {response.status_code} querying Civitai API", flush=True)
                return None
            if _content_length(response.headers) > MAX_API_RESPONSE_BYTES:
                print("[civitai] Civitai API response exceeded size limit", flush=True)
                return None
            raw = response.raw.read(MAX_API_RESPONSE_BYTES + 1)
            if len(raw) > MAX_API_RESPONSE_BYTES:
                return None
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, requests.RequestException, UnicodeError, json.JSONDecodeError) as e:
        print(f"[civitai] Error querying Civitai API: {type(e).__name__}", flush=True)
        return None


def _positive_id(value: int | str) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if 0 < result <= 2**63 - 1 else None


def fetch_by_hash(sha256: str, api_key: str | None = None) -> dict | None:
    normalized = _normalize_sha256(sha256)
    if normalized is None:
        return None
    return _request_json(f"{CIVITAI_API_BASE}/model-versions/by-hash/{normalized}", api_key=api_key)


def fetch_by_version_id(version_id: int | str, api_key: str | None = None) -> dict | None:
    version = _positive_id(version_id)
    if version is None:
        return None
    return _request_json(f"{CIVITAI_API_BASE}/model-versions/{version}", api_key=api_key)


def fetch_by_model_id(model_id: int | str, api_key: str | None = None) -> dict | None:
    model = _positive_id(model_id)
    if model is None:
        return None
    return _request_json(f"{CIVITAI_API_BASE}/models/{model}", api_key=api_key)


def parse_civitai_input(input_str: str, api_key: str | None = None) -> int | None:
    cleaned = str(input_str or "").strip()
    if not cleaned or len(cleaned) > 2048:
        return None
    if cleaned.lower().startswith(("http://", "https://")):
        try:
            parsed = urlsplit(cleaned)
            host = (parsed.hostname or "").lower()
        except ValueError:
            return None
        if host not in {"civitai.com", "www.civitai.com", "civitai.red", "www.civitai.red"}:
            return None
    match = re.search(r"(?:modelVersionId=|/api/download/models/)(\d+)", cleaned)
    if match:
        return _positive_id(match.group(1))
    match = re.search(r"/model-versions/(\d+)", cleaned)
    if match:
        return _positive_id(match.group(1))
    match = re.search(r"/models/(\d+)", cleaned)
    if match:
        model_id = _positive_id(match.group(1))
        if model_id is None:
            return None
        model_data = fetch_by_model_id(model_id, api_key=api_key)
        versions = model_data.get("modelVersions") if model_data else None
        if isinstance(versions, list) and versions and isinstance(versions[0], dict):
            return _positive_id(versions[0].get("id"))
        return None
    if cleaned.isdigit():
        num_id = _positive_id(cleaned)
        if num_id is None:
            return None
        if fetch_by_version_id(num_id, api_key=api_key):
            return num_id
        model_data = fetch_by_model_id(num_id, api_key=api_key)
        versions = model_data.get("modelVersions") if model_data else None
        if isinstance(versions, list) and versions and isinstance(versions[0], dict):
            return _positive_id(versions[0].get("id"))
        return num_id
    return None


def normalize_base_model(civitai_base: str | None) -> str:
    if not civitai_base:
        return "sdxl"
    b = str(civitai_base).lower().strip()
    if any(k in b for k in ("sd 1.5", "sd 1.4", "sd 2.0", "sd 2.1", "sd15", "sd14", "sd21")):
        raise ValueError(
            f"Civitai model architecture '{civitai_base}' is SD 1.5/2.x, which is not supported by DiffusionBear. "
            f"Supported architectures: SDXL (Lightning/Pony/Illustrious), FLUX.2-klein, Krea 2, Z-Image Turbo."
        )
    if any(k in b for k in ("flux.1", "flux1", "flux dev", "flux schnell")):
        raise ValueError(
            f"Civitai model architecture '{civitai_base}' is FLUX.1 (12B), which is incompatible with FLUX.2-klein (4B)."
        )
    if any(k in b for k in ("cascade", "pixart", "auraflow", "hunyuan")):
        raise ValueError(f"Civitai model architecture '{civitai_base}' is not supported by DiffusionBear.")
    if any(k in b for k in ("sdxl", "pony", "illustrious", "sd xl")):
        return "sdxl"
    if any(k in b for k in ("flux.2", "flux2", "klein")):
        return "flux2"
    if "krea" in b:
        return "krea2"
    if any(k in b for k in ("z-image", "zit", "zimage")):
        return "z-image"
    return "sdxl"


def extract_civitai_metadata(version_info: dict) -> dict:
    if not isinstance(version_info, dict):
        raise ValueError("Invalid Civitai model metadata")
    model_obj = version_info.get("model") if isinstance(version_info.get("model"), dict) else {}
    model_type = str(model_obj.get("type") or version_info.get("type") or "LORA").upper()
    if model_type not in ("LORA", "LOCON", "DORA"):
        raise ValueError(f"The Civitai resource is a '{model_type}', not a LoRA.")
    files = [item for item in version_info.get("files") or [] if isinstance(item, dict)]
    safetensor_file = next((item for item in files if str(item.get("name", "")).lower().endswith(".safetensors")), None)
    if safetensor_file is None:
        raise ValueError("The selected Civitai version has no .safetensors file")
    trained_words = version_info.get("trainedWords") or []
    if not isinstance(trained_words, list):
        trained_words = []
    hashes = safetensor_file.get("hashes") if isinstance(safetensor_file.get("hashes"), dict) else {}
    expected_sha256 = _normalize_sha256(
        hashes.get("SHA256") or hashes.get("sha256") or hashes.get("SHA-256") or hashes.get("sha_256")
    )
    try:
        size_kb = float(safetensor_file.get("sizeKB"))
    except (TypeError, ValueError):
        size_kb = 0.0
    expected_bytes = int(size_kb * 1024) if math.isfinite(size_kb) and 0 < size_kb <= MAX_LORA_DOWNLOAD_BYTES / 1024 else None
    if expected_bytes and expected_bytes > MAX_LORA_DOWNLOAD_BYTES:
        expected_bytes = None
    triggers = [str(word).strip()[:200] for word in trained_words[:100] if isinstance(word, str) and word.strip()]
    model_name = str(model_obj.get("name") or version_info.get("name") or "Civitai-LoRA")
    version_name = str(version_info.get("name") or "v1.0")
    model_name = "".join("_" if ord(char) < 32 or ord(char) == 127 else char for char in model_name)[:200]
    version_name = "".join("_" if ord(char) < 32 or ord(char) == 127 else char for char in version_name)[:200]
    return {
        "civitai_version_id": _positive_id(version_info.get("id")),
        "civitai_model_id": _positive_id(version_info.get("modelId")),
        "civitai_model_name": model_name,
        "civitai_version_name": version_name,
        "base_model": normalize_base_model(version_info.get("baseModel")),
        "triggers": triggers[:5],
        "download_url": safetensor_file.get("downloadUrl"),
        "filename": _safe_filename(safetensor_file.get("name"), f"civitai_lora_{_positive_id(version_info.get('id')) or 0}.safetensors"),
        "size_kb": safetensor_file.get("sizeKB"),
        "expected_sha256": expected_sha256,
        "expected_bytes": expected_bytes,
    }


def _emit_progress(progress_cb: Callable | None, status_text: str, frac: float, downloaded: int, total: int, speed: float):
    if not progress_cb:
        return
    payload = {
        "status": status_text,
        "progress": max(0.0, min(1.0, frac)),
        "downloaded_bytes": downloaded,
        "total_bytes": total,
        "speed_mb_s": round(speed, 2),
    }
    try:
        progress_cb(payload)
    except TypeError:
        try:
            progress_cb(status_text, frac)
        except Exception:
            pass


def _cached_lora_matches(path: Path, expected_sha256: str | None, expected_bytes: int | None) -> bool:
    if not is_valid_safetensors(path):
        return False
    try:
        actual_size = path.stat().st_size
    except OSError:
        return False
    if not _size_matches(actual_size, expected_bytes):
        return False
    if expected_sha256:
        try:
            return compute_file_sha256(path) == expected_sha256
        except OSError:
            return False
    return True


def download_civitai_lora(
    version_id: int,
    dest_dir: Path,
    custom_name: str | None = None,
    progress_cb: Callable | None = None,
    api_key: str | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[Path, dict]:
    version = _positive_id(version_id)
    if version is None:
        raise ValueError("Invalid Civitai model version id")
    dest_dir = Path(dest_dir).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(dest_dir, 0o700)
    except OSError:
        pass
    version_info = fetch_by_version_id(version, api_key=api_key)
    if not version_info:
        raise ValueError(f"Could not find model version {version} on Civitai")
    meta = extract_civitai_metadata(version_info)
    resolved_key = get_civitai_api_key(api_key)
    download_url = meta.get("download_url") or f"https://civitai.com/api/download/models/{version}"
    download_url = validate_public_https_url(str(download_url))
    if resolved_key:
        parsed = urlsplit(download_url)
        if (parsed.hostname or "").lower().rstrip(".") in _CIVITAI_AUTH_HOSTS:
            query = dict(parse_qsl(parsed.query, keep_blank_values=True))
            query["token"] = resolved_key
            download_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), ""))
    display_name = custom_name or meta.get("civitai_model_name") or f"lora_{version}"
    safe_display = _safe_filename(display_name, "lora.safetensors").removesuffix(".safetensors")[:60]
    safe_fname = _safe_filename(meta.get("filename"), f"civitai_lora_{version}.safetensors")
    final_filename = f"{safe_display}__{safe_fname}" if safe_display.lower() not in safe_fname.lower() else safe_fname
    dest_root = dest_dir.resolve()
    dest_path = (dest_root / final_filename).resolve()
    if not dest_path.is_relative_to(dest_root):
        raise ValueError("invalid Civitai destination")
    expected_sha256 = meta.get("expected_sha256")
    expected_bytes = meta.get("expected_bytes")
    if _cached_lora_matches(dest_path, expected_sha256, expected_bytes):
        file_size = dest_path.stat().st_size
        _emit_progress(progress_cb, "LoRA already present on disk", 1.0, file_size, file_size, 0.0)
        meta["sha256"] = compute_file_sha256(dest_path, cancel_event=cancel_event)
        meta["path"] = str(dest_path)
        return dest_path, meta
    try:
        dest_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        raise RuntimeError("Could not replace the cached Civitai LoRA") from e
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/octet-stream"}
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"
    _emit_progress(progress_cb, f"Connecting to Civitai for {meta['civitai_model_name']}...", 0.05, 0, 0, 0.0)
    deadline = time.monotonic() + download_deadline_seconds()
    fd = None
    tmp_file = None
    downloaded = 0
    total_bytes = 0
    try:
        response, final_url = open_public_https_stream(
            download_url,
            headers=headers,
            auth_hosts=_CIVITAI_AUTH_HOSTS,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        with response:
            final_host = (urlsplit(final_url).hostname or "").lower()
            content_type = str(response.headers.get("Content-Type", "")).lower()
            if "auth.civitai.com" in final_host or "reason=download-auth" in final_url.lower() or content_type.startswith("text/html"):
                raise CivitaiAuthError(
                    f"Civitai model '{meta['civitai_model_name']}' requires authentication. "
                    "Configure a Civitai API key or download the .safetensors file manually."
                )
            if response.status_code in (401, 403):
                raise CivitaiAuthError("Civitai rejected the configured credentials or requires authentication.")
            if response.status_code != 200:
                raise RuntimeError(f"Civitai download failed with HTTP {response.status_code}")
            total_bytes = _content_length(response.headers)
            if total_bytes > MAX_LORA_DOWNLOAD_BYTES:
                raise ValueError("Civitai LoRA exceeds the 8 GB download limit")
            if expected_bytes and total_bytes and not _size_matches(total_bytes, expected_bytes):
                raise ValueError("Civitai download size does not match model metadata")
            fd, tmp_file = tempfile.mkstemp(dir=str(dest_dir), prefix=".civitai-", suffix=".download")
            with os.fdopen(fd, "wb") as out_f:
                fd = None
                start_time = time.monotonic()
                last_speed_time = start_time
                last_speed_bytes = 0
                current_speed_mb = 0.0
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise DownloadCancelled("Download cancelled by user")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Civitai download deadline exceeded")
                    chunk = response.raw.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > MAX_LORA_DOWNLOAD_BYTES:
                        raise ValueError("Civitai LoRA exceeds the 8 GB download limit")
                    if expected_bytes and downloaded > expected_bytes + 4096:
                        raise ValueError("Civitai download size does not match model metadata")
                    out_f.write(chunk)
                    now = time.monotonic()
                    dt = now - last_speed_time
                    if dt >= 0.5:
                        current_speed_mb = ((downloaded - last_speed_bytes) / (1 << 20)) / dt
                        last_speed_time = now
                        last_speed_bytes = downloaded
                    frac = min(0.95, downloaded / total_bytes) if total_bytes > 0 else 0.5
                    status_text = (
                        f"Downloading {meta['civitai_model_name']}: {downloaded >> 20}MB / {total_bytes >> 20}MB "
                        f"({frac * 100:.0f}%) - {current_speed_mb:.1f} MB/s"
                    ) if total_bytes > 0 else f"Downloading: {downloaded >> 20}MB - {current_speed_mb:.1f} MB/s"
                    _emit_progress(progress_cb, status_text, frac, downloaded, total_bytes, current_speed_mb)
                out_f.flush()
                os.fsync(out_f.fileno())
        if total_bytes and downloaded != total_bytes:
            raise ValueError("Civitai download ended before the expected byte count")
        if expected_bytes and not _size_matches(downloaded, expected_bytes):
            raise ValueError("Civitai download size does not match model metadata")
        if not is_valid_safetensors(tmp_file):
            raise ValueError("Downloaded Civitai payload is not a valid .safetensors archive")
        _emit_progress(progress_cb, "Computing SHA-256 digest...", 0.98, downloaded, total_bytes, 0.0)
        sha = compute_file_sha256(tmp_file, cancel_event=cancel_event)
        if expected_sha256 and sha != expected_sha256:
            raise ValueError("Civitai LoRA SHA-256 does not match model metadata")
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelled("Download cancelled by user")
        os.replace(tmp_file, dest_path)
        tmp_file = None
        try:
            os.chmod(dest_path, 0o600)
        except OSError:
            pass
    except (CivitaiAuthError, DownloadCancelled):
        raise
    except Exception as e:
        raise RuntimeError("Failed downloading LoRA from Civitai") from e
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
    meta["civitai_version_id"] = version
    _emit_progress(progress_cb, "LoRA ready!", 1.0, downloaded, total_bytes, 0.0)
    return dest_path, meta
