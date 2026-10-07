import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_file_bytes(filename: str, content_bytes: bytes) -> str:
    """
    Extrae texto plano de archivos .pdf, .md, .txt o .json.
    """
    ext = os.path.splitext(filename.lower())[1]
    
    if ext == ".pdf":
        try:
            import io
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error leyendo PDF {filename}: {e}")
            return ""
    elif ext in [".txt", ".md", ".json", ".csv"]:
        try:
            return content_bytes.decode("utf-8", errors="ignore").strip()
        except Exception as e:
            print(f"Error leyendo archivo de texto {filename}: {e}")
            return ""
    else:
        try:
            return content_bytes.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""


def parse_json_robustly(raw_json: str) -> list:
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        # Intentar reparar array JSON truncado encontrando el último objeto completo '}'
        last_brace = raw_json.rfind('}')
        if last_brace != -1:
            repaired = raw_json[:last_brace+1] + '\n]'
            try:
                return json.loads(repaired)
            except Exception:
                pass
        # Fallback con regex para extraer todos los objetos estructurados
        matches = re.findall(r'\{\s*"instruction".*?"output":\s*".*?"\s*\}', raw_json, re.DOTALL)
        items = []
        for m in matches:
            try:
                items.append(json.loads(m))
            except Exception:
                pass
        if items:
            return items
        raise


def generate_dataset_with_groq(
    personality_info: str,
    business_info: str,
    num_examples: int = 25
) -> list:
    """
    Genera un dataset sintético estructurado para Fine-Tuning QLoRA utilizando agentes de Groq.
    Recibe la personalidad/perfil del agente y la información completa del negocio.
    """
    api_key = (
        os.getenv("GROQ_TRAIN_MODELS")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("GROQ_RESPONDER_API_KEY")
        or os.getenv("GROQ_ROUTER_API_KEY")
    )
    if not api_key:
        raise ValueError("No se encontró GROQ_TRAIN_MODELS o GROQ_API_KEY en el archivo .env.")

    client = Groq(api_key=api_key)

    system_prompt = """Eres un Ingeniero de Datos experto en Fine-Tuning de LLMs (SFT / QLoRA) de nivel mundial.
Tu tarea es generar un dataset sintético con ALTÍSIMA DIVERSIDAD SINTÁCTICA Y CONVERSACIONAL para evitar el sobreajuste (overfitting) y evitar que el modelo memorice muletillas o frases idénticas.

DEBES RETORNAR ÚNICAMENTE UN ARRAY JSON VÁLIDO CON EL SIGUIENTE FORMATO EXACTO:
[
  {
    "instruction": "Instrucción de sistema del agente que define su rol y personalidad.",
    "input": "Pregunta o consulta del cliente en lenguaje natural realista.",
    "output": "Respuesta ideal del agente respetando la información del negocio con estilo natural y variado."
  }
]

REGLAS CRÍTICAS DE DIVERSIDAD Y ANTI-SOBREAJUSTE:
1. VARIABILIDAD EN SALUDOS Y PRESENTACIÓN:
   - PROHIBIDO empezar todas las respuestas con la misma frase (NO uses siempre "¡Hola! Soy Valentina...").
   - Alterna entre:
     a) Respuestas directas al grano sin saludo formal ("Para un comedor de 6 puestos, lo ideal es...").
     b) Saludos breves u entusiastas ("¡Claro que sí! Te cuento que...", "¡Hola! Qué gusto saludarte...").
     c) Explicaciones técnicas iniciales según la pregunta.
2. VARIABILIDAD EN EL CIERRE (CALL TO ACTION):
   - PROHIBIDO terminar todas las respuestas pidiendo datos de cotización exactamente igual.
   - Alterna cierres: preguntas sobre preferencias de diseño, consejos de uso/espacio, respuestas conclusivas amables sin preguntas o sugerencias de catálogos.
3. VARIABILIDAD EN 'instruction':
   - Adapta sutilmente la instrucción del sistema para reflejar diferentes enfoques del rol según la pregunta (ej. "Eres Valentina, asesora en diseño de muebles...", "Eres Valentina, experta en materiales y acabados...").
4. RETORNA ÚNICAMENTE EL ARRAY JSON PURO SIN TEXTO ADICIONAL."""

    user_prompt = f"""
=== SECCIÓN 1: PERSONALIDAD Y ROL DEL AGENTE ===
{personality_info.strip() or 'Agente de atención al cliente amable, conciso y profesional.'}

=== SECCIÓN 2: INFORMACIÓN Y BASE DE CONOCIMIENTO DEL NEGOCIO ===
{business_info.strip() or 'Información general de atención al cliente.'}

Genera exactamente {num_examples} pares estructurados de entrenamiento en formato JSON puro con máxima variabilidad.
"""

    try:
        models_to_try = [
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "llama-3.1-8b-instant"
        ]

        completion = None
        last_err = None
        for model_name in models_to_try:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.85,
                    max_tokens=8192,
                )
                if completion:
                    break
            except Exception as e:
                last_err = e
                continue

        if not completion:
            raise RuntimeError(f"Fallo en la generación sintética con Groq: {str(last_err)}")

        raw_content = completion.choices[0].message.content.strip()

        # Extraer bloque JSON si viene envuelto en markdown ```json ... ```
        json_match = re.search(r'\[\s*\{.*\}\s*\]', raw_content, re.DOTALL)
        if json_match:
            raw_json = json_match.group(0)
        else:
            raw_json = raw_content

        dataset = parse_json_robustly(raw_json)
        
        # Validar y limpiar el dataset
        cleaned_dataset = []
        for item in dataset:
            if isinstance(item, dict) and "input" in item and "output" in item:
                cleaned_dataset.append({
                    "instruction": item.get("instruction") or "Eres un asistente virtual amable y profesional.",
                    "input": item.get("input", "").strip(),
                    "output": item.get("output", "").strip()
                })

        return cleaned_dataset
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "invalid_api_key" in error_msg.lower():
            raise RuntimeError("La API Key de Groq configurada en .env no es válida (401 Invalid API Key). Por favor actualiza GROQ_TRAIN_MODELS en tu archivo .env.")
        print(f"Error generando dataset sintético con Groq: {e}")
        raise RuntimeError(f"Fallo en la generación sintética con Groq: {error_msg}")
