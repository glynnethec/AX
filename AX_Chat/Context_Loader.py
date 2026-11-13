# -*- coding: utf-8 -*-
"""
Context_Loader.py
Motor de Inyección de Contexto Dinámico (Zero-Waste).
Garantiza que el Core (Identidad, Reglas, Quiénes Somos, Herramientas del Panel y Navegación)
esté siempre presente, e inyecta módulos profundos bajo demanda del Router.
"""

import os
from typing import List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))

def load_context(selected_files: Optional[List[str]] = None) -> str:
    """Lee el CORE base + Contexto Dinámico Seleccionado por el Router."""
    prompts_dir = os.path.join(current_dir, 'prompts')

    # Core obligatorio en cada turno: el asesor NUNCA pierde noción de quién es GLYNNE ni de las tools del Panel
    core_files = [
        'persona.md',
        'rules.md',
        'knowledge.md',
        'platform_tools.md',
        'navigation.md',
    ]

    files_to_load = core_files + (selected_files if selected_files else [])

    # Limpiar duplicados manteniendo el orden de prioridad
    seen = set()
    final_files = [x for x in files_to_load if not (x in seen or seen.add(x))]

    context = []
    for f in final_files:
        path = os.path.join(prompts_dir, f)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                context.append(file.read().strip())

    return "\n\n".join(context)
