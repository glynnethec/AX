"""
core — Capa de seguridad y orquestación compartida de GLYNNE CORE.
- auth: verificación de identidad (JWT de Supabase) por petición.
- rate_limit: limitador en memoria por usuario / clave.
- jobs: gestor de entrenamientos QLoRA aislados por usuario.
"""
