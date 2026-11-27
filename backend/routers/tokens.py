import threading

from fastapi import APIRouter, HTTPException

import civitai_service
import hf_service
from state import DATA_DIR, TokenRequest, _atomic_write_text

router = APIRouter(tags=["tokens"])

CIVITAI_TOKEN_FILE = DATA_DIR / "civitai_token.txt"
HF_TOKEN_FILE = DATA_DIR / "hf_token.txt"
_token_lock = threading.Lock()


def _mask(value: str) -> str:
    return value[:4] + "..." + value[-4:] if len(value) > 8 else "***"


def _clean_token(value: str) -> str:
    token = value.strip()
    if len(token) > 4096 or any(ord(char) < 32 or ord(char) == 127 for char in token):
        raise HTTPException(400, "invalid token")
    return token


def _clear_token(path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as e:
        raise HTTPException(500, "failed to clear stored token") from e


@router.get("/api/civitai/token")
def get_civitai_token():
    key = civitai_service.get_civitai_api_key()
    return {"configured": bool(key), "masked": _mask(key) if key else None}


@router.post("/api/civitai/token")
def set_civitai_token(req: TokenRequest):
    token = _clean_token(req.token)
    with _token_lock:
        if not token:
            _clear_token(CIVITAI_TOKEN_FILE)
            return {"configured": False, "masked": None}
        _atomic_write_text(CIVITAI_TOKEN_FILE, token)
    return {"configured": True, "masked": _mask(token)}


@router.get("/api/hf/token")
def get_hf_token():
    key = hf_service.get_hf_token()
    return {"configured": bool(key), "masked": _mask(key) if key else None}


@router.post("/api/hf/token")
def set_hf_token(req: TokenRequest):
    token = _clean_token(req.token)
    with _token_lock:
        if not token:
            _clear_token(HF_TOKEN_FILE)
            return {"configured": False, "masked": None}
        _atomic_write_text(HF_TOKEN_FILE, token)
    return {"configured": True, "masked": _mask(token)}
