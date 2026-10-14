"""
core/rate_limit.py
Limitador de ventana deslizante en memoria (por proceso). Protege endpoints costosos
(LLM, TTS, GPU) contra abuso sin afectar el uso normal de un usuario.
"""

import threading
import time
from collections import deque
from typing import Deque, Dict

from fastapi import HTTPException

_buckets: Dict[str, Deque[float]] = {}
_lock = threading.Lock()
_last_prune = time.time()


def _prune(now: float):
    global _last_prune
    if now - _last_prune < 300:
        return
    _last_prune = now
    for key in [k for k, q in _buckets.items() if not q or now - q[-1] > 3600]:
        _buckets.pop(key, None)


def check_rate_limit(key: str, limit: int, window_seconds: float, message: str = None):
    """Lanza HTTP 429 si `key` superó `limit` peticiones en `window_seconds`."""
    if limit <= 0:
        return
    now = time.time()
    with _lock:
        _prune(now)
        q = _buckets.setdefault(key, deque())
        while q and now - q[0] > window_seconds:
            q.popleft()
        if len(q) >= limit:
            retry_after = max(1, int(window_seconds - (now - q[0])))
            raise HTTPException(
                status_code=429,
                detail=message or f"Demasiadas solicitudes. Intenta de nuevo en {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )
        q.append(now)
