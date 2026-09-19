# -*- coding: utf-8 -*-
"""
Context_Loader.py
Motor de Inyección de Contexto Dinámico (Zero-Waste).
Lee del disco únicamente los archivos solicitados por el Router.
"""

import os
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))

def load_context(selected_files: List[str] = None) -> str:
    """Lee el CORE + Contexto Dinámico Seleccionado."""
    prompts_dir = os.path.join(current_dir, 'prompts')
    
    # Core (Obligatorio siempre)
    core_files = ['persona.md', 'rules.md', 'navigation.md']
    
    files_to_load = core_files + (selected_files if selected_files else [])
    
    # Limpiar duplicados manteniendo orden
    seen = set()
    final_files = [x for x in files_to_load if not (x in seen or seen.add(x))]
    
    context = []
    for f in final_files:
        path = os.path.join(prompts_dir, f)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                context.append(file.read().strip())
    
    return "\n\n".join(context)
