"""
core/auth.py
Verificación de identidad por petición usando el JWT de sesión de Supabase.

- El frontend envía `Authorization: Bearer <access_token>` (o `?access_token=` para
  descargas abiertas con window.open / <a href>).
- Se valida contra `GET {SUPABASE_URL}/auth/v1/user` y se cachea 60s para no añadir latencia.
- El `user_id` usado por el backend sale SIEMPRE del token verificado, nunca del body.
- `AUTH_ENFORCE=false` (env) actúa como interruptor de emergencia: permite peticiones sin token
  y usa el `user_id` legacy enviado por el cliente.
"""

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from fastapi import Header, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qfxxspgxsvlbsbeosqgu.supabase.co").rstrip("/")
# La clave publicable (anon) es pública por diseño; se usa solo como `apikey` del gateway de Supabase.
SUPABASE_ANON_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_MHwQqLk6xwTwRKPT0BFUKg_u67pLrCL")
# Clave de servicio (solo backend) para leer tablas protegidas por RLS (p.ej. sub_api_keys).
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

AUTH_ENFORCE = os.getenv("AUTH_ENFORCE", "true").strip().lower() in ("1", "true", "yes", "on")
# Mismo criterio que el frontend (getCurrentUser): solo cuentas Google acceden a la plataforma.
ALLOWED_PROVIDERS = {p.strip() for p in os.getenv("AUTH_ALLOWED_PROVIDERS", "google").split(",") if p.strip()}

_CACHE_TTL_SECONDS = 60
_CACHE_MAX_ENTRIES = 5000
_cache: Dict[str, Tuple[float, Optional["AuthUser"]]] = {}
_cache_lock = threading.Lock()


@dataclass(frozen=True)
class AuthUser:
    id: Optional[str]
    email: Optional[str] = None
    token: Optional[str] = None

    @property
    def is_authenticated(self) -> bool:
        return bool(self.id)


ANONYMOUS = AuthUser(id=None)


def _token_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _cache_get(key: str):
    with _cache_lock:
        entry = _cache.get(key)
        if not entry:
            return None
        expires_at, user = entry
        if expires_at < time.time():
            _cache.pop(key, None)
            return None
        return entry


def _cache_set(key: str, user: Optional[AuthUser], ttl: float = _CACHE_TTL_SECONDS):
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX_ENTRIES:
            now = time.time()
            for k in [k for k, (exp, _) in _cache.items() if exp < now]:
                _cache.pop(k, None)
            if len(_cache) >= _CACHE_MAX_ENTRIES:
                _cache.clear()
        _cache[key] = (time.time() + ttl, user)


def _fetch_supabase_user(token: str) -> Optional[AuthUser]:
    """Consulta síncrona a Supabase Auth. Devuelve None si el token no es válido."""
    res = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}"},
        timeout=5,
    )
    if res.status_code != 200:
        return None
    data = res.json() or {}
    user_id = data.get("id")
    if not user_id:
        return None
    provider = (data.get("app_metadata") or {}).get("provider")
    if ALLOWED_PROVIDERS and provider not in ALLOWED_PROVIDERS:
        return None
    email = data.get("email") or (data.get("user_metadata") or {}).get("email")
    return AuthUser(id=user_id, email=email, token=token)


async def verify_token(token: Optional[str]) -> Optional[AuthUser]:
    """Valida un access_token de Supabase con caché. Lanza 503 si Supabase no responde."""
    if not token:
        return None
    key = _token_key(token)
    cached = _cache_get(key)
    if cached is not None:
        return cached[1]
    try:
        user = await run_in_threadpool(_fetch_supabase_user, token)
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="Servicio de autenticación no disponible. Intenta de nuevo.")
    # Los tokens inválidos se cachean menos tiempo para no penalizar un refresh de sesión.
    _cache_set(key, user, ttl=_CACHE_TTL_SECONDS if user else 10)
    return user


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        return token or None
    return None


async def get_current_user(
    authorization: Optional[str] = Header(None),
    access_token: Optional[str] = Query(None, include_in_schema=False),
) -> AuthUser:
    """Dependencia FastAPI: exige una sesión válida de la plataforma (salvo AUTH_ENFORCE=false)."""
    token = _extract_bearer(authorization) or access_token
    user = await verify_token(token)
    if user:
        return user
    if not AUTH_ENFORCE:
        return ANONYMOUS
    raise HTTPException(status_code=401, detail="Sesión no válida o expirada. Inicia sesión nuevamente.")


def resolve_user_id(user: AuthUser, legacy_user_id: Optional[str] = None) -> str:
    """ID efectivo: el del token; solo en modo no estricto se acepta el legacy del cliente."""
    if user.is_authenticated:
        return user.id  # type: ignore[return-value]
    return legacy_user_id or "default_user"
