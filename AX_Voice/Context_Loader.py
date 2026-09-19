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
    
    final_context = "\n\n".join(context)
    voice_constraint = "\n\n[VOICE INTERFACE CONSTRAINT]:\nEstás hablando por voz. TUS RESPUESTAS DEBEN SER EXTREMADAMENTE CONCISAS, CORTAS Y NATURALES. NO uses formatos de markdown complejos (ni listas largas, ni negritas) porque el usuario te escuchará, no te leerá. Responde máximo en 1 a 3 oraciones breves a menos que el usuario pida una explicación extensa."
    
    return final_context + voice_constraint
