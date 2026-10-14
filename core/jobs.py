"""
core/jobs.py
Gestor de entrenamientos QLoRA (Modal) AISLADOS POR USUARIO.

Cada usuario tiene su propio estado, su propio subproceso `modal run` y su propia ruta de
modelo en el volumen (`/models/users/<uid>/ax_model`). Ningún usuario puede ver, cancelar
ni sobrescribir el entrenamiento de otro.

Ciclo de vida ligado a la página:
- El frontend consulta `/api/train/status` cada 2s mientras está en Create_you_GLYNNE_model;
  cada consulta actúa como *heartbeat*.
- Si no llega ningún heartbeat durante HEARTBEAT_TIMEOUT (pestaña cerrada, crash del navegador,
  navegación fuera del Studio), el watchdog cancela el job de ese usuario automáticamente.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
MODAL_VOLUME = "ax-qlora-models"
MODAL_APP_NAME = "ax-glynne-trainer"

HEARTBEAT_TIMEOUT = int(os.getenv("TRAIN_HEARTBEAT_TIMEOUT", "120"))
MAX_LOG_LINES = 1000
IDLE_JOB_TTL = 3600  # liberar memoria de jobs inactivos tras 1h


def safe_user_id(user_id: str) -> str:
    """Normaliza el ID para usarlo de forma segura en rutas de volumen/disco."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "", user_id or "")[:64]
    return cleaned or "default_user"


def user_model_dir(user_id: str) -> str:
    """Ruta relativa dentro del volumen de Modal para el modelo del usuario."""
    return f"users/{safe_user_id(user_id)}/ax_model"


def user_downloads_dir(user_id: str) -> str:
    return os.path.join(DOWNLOADS_DIR, safe_user_id(user_id))


def get_modal_cmd() -> str:
    modal_bin = shutil.which("modal")
    if modal_bin:
        return modal_bin
    venv_dir = os.path.dirname(sys.executable)
    potential_modal = os.path.join(venv_dir, "modal")
    if os.path.exists(potential_modal):
        return potential_modal
    relative_modal = os.path.join(BASE_DIR, "venv", "bin", "modal")
    if os.path.exists(relative_modal):
        return relative_modal
    return "modal"


def _idle_state() -> dict:
    return {
        "status": "idle",  # idle, running, completed, error
        "model_name": "",
        "progress": 0,
        "logs": [],
        "error_message": None,
        "download_command": None,
    }


class _TrainingJob:
    __slots__ = ("state", "process", "last_seen", "generation", "dataset_file")

    def __init__(self):
        self.state = _idle_state()
        self.process: Optional[subprocess.Popen] = None
        self.last_seen = time.time()
        self.generation = 0  # se incrementa en cada reset para invalidar hilos antiguos
        self.dataset_file: Optional[str] = None


class TrainingJobManager:
    def __init__(self):
        self._jobs: Dict[str, _TrainingJob] = {}
        self._lock = threading.RLock()
        self._watchdog_started = False

    # ── helpers ──────────────────────────────────────────────────────────
    def _job(self, user_id: str) -> _TrainingJob:
        with self._lock:
            job = self._jobs.get(user_id)
            if job is None:
                job = _TrainingJob()
                self._jobs[user_id] = job
            return job

    def _log(self, job: _TrainingJob, line: str):
        logs = job.state["logs"]
        logs.append(line)
        if len(logs) > MAX_LOG_LINES:
            del logs[: len(logs) - MAX_LOG_LINES]

    # ── API pública ──────────────────────────────────────────────────────
    def get_status(self, user_id: str) -> dict:
        """Devuelve el estado del usuario y registra el heartbeat."""
        job = self._job(user_id)
        with self._lock:
            job.last_seen = time.time()
            state = dict(job.state)
            state["logs"] = list(job.state["logs"])
            return state

    def is_running(self, user_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(user_id)
            return bool(job and job.state["status"] == "running")

    def start(self, user_id: str, model_name: str, dataset: List[dict]) -> dict:
        self._ensure_watchdog()
        job = self._job(user_id)
        with self._lock:
            # Si estaba en error o completado previamente, reiniciar limpiamente
            if job.state["status"] in ("completed", "error"):
                job.state = _idle_state()
            if job.state["status"] == "running":
                return {"status": "error", "message": "Ya hay un entrenamiento en curso. Puedes pulsar Resetear si deseas comenzar de nuevo."}

            job.generation += 1
            generation = job.generation
            job.last_seen = time.time()
            job.state = _idle_state()
            job.state.update({
                "status": "running",
                "model_name": model_name,
                "progress": 10,
                "logs": [f"[GLYNNE CORE] Conectando con la nube de Modal para {model_name}..."],
            })

        # Un nuevo entrenamiento invalida cualquier .gguf descargado previamente por este usuario
        shutil.rmtree(user_downloads_dir(user_id), ignore_errors=True)

        threading.Thread(
            target=self._run, args=(user_id, generation, model_name, dataset),
            name=f"train-{safe_user_id(user_id)[:8]}", daemon=True,
        ).start()
        return {"status": "started", "message": f"Entrenamiento de {model_name} iniciado en la nube de Modal."}

    def reset(self, user_id: str) -> dict:
        """Cancela SOLO el proceso de este usuario y limpia su estado."""
        with self._lock:
            job = self._jobs.get(user_id)
            if job is None:
                return {"status": "reset", "message": "Proceso cancelado y estado reseteado correctamente."}
            job.generation += 1
            process = job.process
            job.process = None
            job.state = _idle_state()
        if process is not None:
            threading.Thread(target=self._terminate, args=(process,), daemon=True).start()
        return {"status": "reset", "message": "Proceso cancelado y estado reseteado correctamente."}

    # ── ejecución ────────────────────────────────────────────────────────
    def _is_current(self, user_id: str, generation: int) -> bool:
        with self._lock:
            job = self._jobs.get(user_id)
            return bool(job and job.generation == generation)

    def _update(self, user_id: str, generation: int, **fields):
        with self._lock:
            job = self._jobs.get(user_id)
            if job and job.generation == generation:
                job.state.update(fields)

    def _append(self, user_id: str, generation: int, line: str, progress: Optional[int] = None):
        with self._lock:
            job = self._jobs.get(user_id)
            if job and job.generation == generation:
                self._log(job, line)
                if progress is not None:
                    job.state["progress"] = progress

    def _run(self, user_id: str, generation: int, model_name: str, dataset_raw: List[dict]):
        dataset_path = None
        try:
            # El dataset viaja en un archivo temporal (evita el límite de longitud de argv)
            fd, dataset_path = tempfile.mkstemp(prefix="ax_dataset_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(dataset_raw, f, ensure_ascii=False)

            cmd = [
                get_modal_cmd(), "run", "AX_Trainer/AX_QLoRA_Trainer.py",
                "--model-name", model_name,
                "--dataset-file", dataset_path,
                "--user-id", safe_user_id(user_id),
            ]
            self._append(user_id, generation, "[MODAL] Solicitando GPU remota en la nube...", progress=20)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=BASE_DIR,
                start_new_session=True,  # permite señalizar todo el grupo al cancelar
            )
            with self._lock:
                job = self._jobs.get(user_id)
                if not job or job.generation != generation:
                    # Fue reseteado mientras arrancaba
                    threading.Thread(target=self._terminate, args=(process,), daemon=True).start()
                    return
                job.process = process

            for line in iter(process.stdout.readline, ""):
                if not self._is_current(user_id, generation):
                    break
                if not line:
                    continue
                cleaned = line.strip()
                progress = None
                if "[1/4]" in cleaned or "Cargando modelo" in cleaned:
                    progress = 35
                elif "[2/4]" in cleaned or "LoRA" in cleaned:
                    progress = 55
                elif "[3/4]" in cleaned or "datos" in cleaned:
                    progress = 70
                elif "[4/4]" in cleaned or "SFTTrainer" in cleaned:
                    progress = 85
                elif "Guardando modelo" in cleaned or "GGUF" in cleaned:
                    progress = 95
                self._append(user_id, generation, cleaned, progress)

            try:
                process.stdout.close()
            except Exception:
                pass
            return_code = process.wait()

            if not self._is_current(user_id, generation):
                # Si ya fue reseteado voluntariamente por el usuario
                return

            if return_code == 0:
                model_dir = user_model_dir(user_id)
                self._update(
                    user_id, generation,
                    progress=100,
                    status="completed",
                    download_command=f"modal volume get {MODAL_VOLUME} {model_dir}/unsloth.Q4_K_M.gguf ./",
                )
                self._append(user_id, generation, "[SISTEMA] ✅ Proceso de entrenamiento finalizado con éxito. Modelo .GGUF listo para descarga.")
                # Pre-calentar el modelo del usuario en Modal para que la primera consulta de chat sea inmediata
                threading.Thread(target=self._warmup, args=(user_id,), daemon=True).start()
            else:
                self._update(user_id, generation, status="error", error_message="El comando de Modal finalizó con error o fue cancelado.")
                self._append(user_id, generation, "[ERROR] El proceso fue abortado o devolvió un código de error.")
        except Exception as e:
            if self._is_current(user_id, generation):
                self._update(user_id, generation, status="error", error_message=str(e))
                self._append(user_id, generation, f"[EXCEPTION] {str(e)}")
        finally:
            with self._lock:
                job = self._jobs.get(user_id)
                if job and job.generation == generation:
                    job.process = None
            if dataset_path:
                try:
                    os.remove(dataset_path)
                except OSError:
                    pass

    @staticmethod
    def _warmup(user_id: str):
        try:
            import modal as m_mod
            f_w = m_mod.Function.from_name(MODAL_APP_NAME, "inferir_modelo_remote")
            try:
                f_w.remote("Hola", "Responder de forma amable y concisa.", safe_user_id(user_id))
            except Exception:
                f_w.remote("Hola", "Responder de forma amable y concisa.")
        except Exception:
            pass

    @staticmethod
    def _terminate(process: subprocess.Popen):
        """SIGINT → `modal run` detiene su app efímera en la nube; luego SIGTERM/SIGKILL si no responde."""
        if process.poll() is not None:
            return
        try:
            pgid = os.getpgid(process.pid)
        except Exception:
            pgid = None
        for sig, wait_s in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 0)):
            try:
                if pgid is not None:
                    os.killpg(pgid, sig)
                else:
                    process.send_signal(sig)
            except ProcessLookupError:
                return
            except Exception as e:
                print(f"[TRAIN] Error enviando señal {sig} al proceso de entrenamiento: {e}")
            if wait_s:
                try:
                    process.wait(timeout=wait_s)
                    return
                except subprocess.TimeoutExpired:
                    continue

    # ── watchdog ─────────────────────────────────────────────────────────
    def _ensure_watchdog(self):
        with self._lock:
            if self._watchdog_started:
                return
            self._watchdog_started = True
        threading.Thread(target=self._watchdog_loop, name="train-watchdog", daemon=True).start()

    def _watchdog_loop(self):
        while True:
            time.sleep(15)
            now = time.time()
            to_cancel, to_drop = [], []
            with self._lock:
                for uid, job in self._jobs.items():
                    idle_for = now - job.last_seen
                    if job.state["status"] == "running" and idle_for > HEARTBEAT_TIMEOUT:
                        to_cancel.append(uid)
                    elif job.state["status"] != "running" and job.process is None and idle_for > IDLE_JOB_TTL:
                        to_drop.append(uid)
                for uid in to_drop:
                    self._jobs.pop(uid, None)
            for uid in to_cancel:
                print(f"[TRAIN WATCHDOG] Sin heartbeat de {safe_user_id(uid)[:8]}… durante {HEARTBEAT_TIMEOUT}s. Cancelando entrenamiento huérfano.")
                self.reset(uid)

    def shutdown(self):
        """Detiene todos los procesos al apagar el servidor (evita GPUs huérfanas en Modal)."""
        with self._lock:
            processes = [j.process for j in self._jobs.values() if j.process is not None]
            for j in self._jobs.values():
                j.generation += 1
                j.process = None
        for p in processes:
            self._terminate(p)


training_jobs = TrainingJobManager()
