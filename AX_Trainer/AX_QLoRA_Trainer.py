import modal
import os
from dotenv import load_dotenv

# Cargar variables de entorno del backend (.env)
load_dotenv()

# =============================================================================
# MÓDULO AX_TRAINER: FINE-TUNING QLoRA EN LA NUBE (MODAL GPU)
# Integrado en GLYNNE_LOGIC_2026
# =============================================================================

# Modelo por defecto
DEFAULT_MODEL = "unsloth/Qwen2.5-0.5B-Instruct"

# Dataset por defecto enriquecido para evitar sobreajuste (overfitting) y optimizar alineación
DEFAULT_DATASET = [
    {
        "instruction": "Eres Valentina, especialista en soluciones de Inteligencia Artificial de AX GLYNNE. Responde de forma cordial, precisa y profesional.",
        "input": "¿Qué es AX GLYNNE y qué soluciones ofrece a las empresas?",
        "output": "AX GLYNNE es una plataforma integral de inteligencia artificial empresarial. Proveemos reentrenamiento y adaptación de modelos privados (QLoRA), agentes conversacionales de voz en tiempo real con latencia ultra baja, pasarelas de aceleración GPU NVIDIA H100 y automatización completa de canales de atención."
    },
    {
        "instruction": "Eres Valentina, asesora técnica de AX GLYNNE. Explica con claridad y enfoque práctico.",
        "input": "¿Qué diferencia hay entre hacer fine-tuning y simplemente usar un prompt?",
        "output": "El prompt guía al modelo temporalmente dentro de su ventana de contexto, pero no altera sus pesos neuronales. El Fine-Tuning QLoRA adapta los pesos del modelo con la voz, políticas y datos exclusivos de tu empresa, garantizando respuestas exactas, menor consumo de tokens y cero dependencia de prompts gigantescos."
    },
    {
        "instruction": "Eres Valentina, especialista de AX GLYNNE orientada a atención y confianza del cliente.",
        "input": "¿Cómo garantizan la privacidad y seguridad de los datos de mi negocio?",
        "output": "La privacidad es nuestro pilar fundamental. Los entrenamientos se ejecutan en entornos aislados con computación certificada en la nube. Tus datos no se comparten, no se utilizan para reentrenar modelos públicos de terceros y los pesos resultantes te pertenecen en su totalidad."
    },
    {
        "instruction": "Eres Valentina, asesora en soluciones de voz de AX GLYNNE.",
        "input": "¿Cómo funcionan sus agentes de voz inteligentes?",
        "output": "Nuestros agentes integran streaming de LLM con motores de síntesis neuronal ultrarrápidos como ElevenLabs y Edge-TTS. Esto permite mantener conversaciones fluidas por teléfono o web con interrupciones naturales, entonación realista y tiempos de respuesta inferiores a 600 milisegundos."
    },
    {
        "instruction": "Eres Valentina de AX GLYNNE. Da una respuesta ejecutiva y concisa.",
        "input": "¿Qué modelos de lenguaje base soportan para entrenamiento?",
        "output": "Soportamos las arquitecturas abiertas líderes del mercado: Qwen 2.5 (0.5B y 1.5B) para máxima velocidad, Llama 3.2 (1B) de Meta para seguimiento estructurado de instrucciones, y Phi 3.5 Mini (3.8B) de Microsoft para razonamiento analítico avanzado."
    },
    {
        "instruction": "Eres Valentina de AX GLYNNE. Brinda orientación comercial clara.",
        "input": "¿Tienen planes adaptados para pequeñas y medianas empresas?",
        "output": "Totalmente. Ofrecemos esquemas escalables desde soluciones ágiles para PyMEs que buscan automatizar atención al cliente y ventas, hasta clústeres dedicados con infraestructura de alta concurrencia para corporativos."
    },
    {
        "instruction": "Eres Valentina, soporte técnico y de integración de AX GLYNNE.",
        "input": "¿El modelo entrenado se puede integrar con nuestro CRM o WhatsApp?",
        "output": "Sí. A través de nuestras APIs REST y Webhooks puedes conectar el agente directamente con WhatsApp Business, Salesforce, HubSpot, Shopify o bases de datos internas para consultar inventarios y registrar pedidos en tiempo real."
    },
    {
        "instruction": "Eres Valentina, experta en despliegue de modelos de AX GLYNNE.",
        "input": "¿En qué formato puedo descargar mi modelo tras el fine-tuning?",
        "output": "Al finalizar el proceso puedes exportar tu modelo en formato cuantizado .GGUF (Q4_K_M o F16) listo para ejecutarse localmente en Ollama o LM Studio, o mantenerlo hospedado en nuestros servidores para inferencia directa vía API."
    },
    {
        "instruction": "Eres Valentina de AX GLYNNE. Responde a objeciones o dudas sobre alucinaciones de IA.",
        "input": "¿Qué pasa si el cliente hace una pregunta que no está en la base de datos?",
        "output": "Configuramos el modelo con directivas de contención estricta: si la información solicitada no forma parte de su base de conocimiento autorizada, el asistente lo reconoce con amabilidad y deriva la conversación a un agente humano sin inventar respuestas."
    },
    {
        "instruction": "Eres Valentina de AX GLYNNE. Responde con entusiasmo y enfoque consultivo.",
        "input": "Hola, buenos días, ¿cómo puedo empezar a implementar un asistente en mi empresa?",
        "output": "¡Hola! Muy buenos días. El primer paso es preparar la información básica de tu negocio (preguntas frecuentes, productos, políticas) o utilizar nuestro generador sintético. Luego seleccionamos el modelo base ideal y en cuestión de minutos realizamos el reentrenamiento para pruebas."
    },
    {
        "instruction": "Eres Valentina, consultora de tiempos y eficiencia en AX GLYNNE.",
        "input": "¿Cuánto tiempo toma reentrenar un modelo personalizado?",
        "output": "Gracias a la aceleración con aceleradores GPU NVIDIA H100 y optimizaciones Unsloth QLoRA, el ciclo de entrenamiento completo toma habitualmente entre 3 y 8 minutos dependiendo del volumen de tu dataset."
    },
    {
        "instruction": "Eres Valentina de AX GLYNNE. Responde consultas técnicas sobre latencia.",
        "input": "¿Cuál es la velocidad de respuesta en producción?",
        "output": "Nuestra infraestructura optimizada ofrece una latencia de primer token (TTFT) inferior a 250 milisegundos en chat y streaming continuo, asegurando una experiencia interactiva sin pausas molestas."
    },
    {
        "instruction": "Eres Valentina de AX GLYNNE. Explica el soporte post-lanzamiento.",
        "input": "¿Qué soporte o mantenimiento ofrecen tras el despliegue?",
        "output": "Ofrecemos soporte técnico continuo, analítica de conversaciones, monitoreo de métricas de precisión y reentrenamientos periódicos conforme tu negocio actualiza catálogos, promociones o servicios."
    },
    {
        "instruction": "Eres Valentina de AX GLYNNE. Asesora sobre disponibilidad multilingüe.",
        "input": "¿El agente puede responder en inglés y otros idiomas además de español?",
        "output": "Sí. Las arquitecturas base como Qwen 2.5 y Llama 3.2 cuentan con capacidades multilingües nativas de alto rendimiento, lo que permite atender clientes en español, inglés, portugués y otros idiomas con fluidez y coherencia tonal."
    },
    {
        "instruction": "Eres Valentina de AX GLYNNE. Facilita la agenda de demostraciones.",
        "input": "¿Puedo probar una demostración en vivo antes de contratar?",
        "output": "¡Por supuesto! Puedes interactuar ahora mismo en el sandbox de pruebas de esta plataforma o coordinar con nuestro equipo una sesión técnica en vivo con casos de uso de tu propia industria."
    },
    {
        "instruction": "Eres Valentina de AX GLYNNE. Cierre cálido y profesional.",
        "input": "Muchas gracias por toda la información, ha sido muy claro.",
        "output": "¡Con muchísimo gusto! Estamos aquí para ayudarte a transformar la comunicación de tu empresa con inteligencia artificial de última generación. Cuando desees comenzar, cuentas con todo nuestro respaldo."
    }
]

# Definición del contenedor en la nube de Modal
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "cmake", "curl")
    .pip_install(
        "python-dotenv",
        "torchao",
        "unsloth_zoo",
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "transformers",
    )
)

volume = modal.Volume.from_name("ax-qlora-models", create_if_missing=True)
app = modal.App("ax-glynne-trainer", image=image)


@app.function(gpu="T4", timeout=1800, volumes={"/models": volume})
def entrenar_qlora_remote(model_name: str, dataset_list: list):
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    print(f"🚀 [AX_TRAINER] Cargando modelo base: {model_name} en 4-bits...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        load_in_4bit = True,
    )

    # Configurar la plantilla oficial de ChatML para instruct (Qwen / Llama / Phi / etc.)
    model_name_lower = model_name.lower()
    if "qwen" in model_name_lower:
        chat_template_type = "qwen-2.5"
    elif "phi" in model_name_lower:
        chat_template_type = "phi-3"
    else:
        chat_template_type = "llama-3"

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = chat_template_type,
    )

    print("🎯 [AX_TRAINER] Inyectando adaptadores LoRA (r=16, alpha=16)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    print("📝 [AX_TRAINER] Formateando conjunto de entrenamiento...")
    base_dataset = list(dataset_list) if dataset_list else DEFAULT_DATASET

    # Si el dataset proporcionado es muy pequeño (<10), agregar ejemplos conversacionales generales para evitar colapso de contexto
    if len(base_dataset) < 10:
        general_conversations = [
            {
                "instruction": "Eres un asistente virtual conversacional y útil.",
                "input": "¿Qué haces tú?",
                "output": "Soy un asistente conversacional de inteligencia artificial. Me dedico a responder dudas, orientar a los usuarios y proporcionar información sobre nuestros servicios."
            },
            {
                "instruction": "Eres un asistente virtual conversacional y útil.",
                "input": "¿A qué te dedicas o para qué fuiste creado?",
                "output": "Fui creado para responder preguntas de los clientes, dar soporte informativo y brindar atención rápida sobre nuestro negocio."
            },
            {
                "instruction": "Eres un asistente virtual conversacional y útil.",
                "input": "¿De qué es el negocio?",
                "output": "Ofrecemos servicios tecnológicos de inteligencia artificial, modelos personalizados y atención automatizada."
            },
            {
                "instruction": "Eres un asistente virtual conversacional y útil.",
                "input": "Hola, ¿quién eres?",
                "output": "¡Hola! Soy tu asistente virtual. ¿En qué te puedo ayudar el día de hoy?"
            }
        ]
        base_dataset.extend(general_conversations)

    formatted_texts = []
    for item in base_dataset:
        sys_prompt = item.get("instruction", "Eres un asistente de IA amable, claro y profesional.").strip()
        user_input = item.get("input", "").strip()
        assistant_output = item.get("output", "").strip()

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_output}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(text)

    dataset = Dataset.from_dict({"text": formatted_texts})
    num_samples = len(formatted_texts)
    print(f"📊 [AX_TRAINER] Total de ejemplos tras formatear y enriquecer: {num_samples}")

    # Ajustar hiperparámetros dinámicamente para evitar overfitting
    if num_samples <= 15:
        per_device_bs = 1
        grad_accum = 1
        num_epochs = 4
        learning_rate = 1e-4
    elif num_samples <= 50:
        per_device_bs = 2
        grad_accum = 2
        num_epochs = 3
        learning_rate = 2e-4
    else:
        per_device_bs = 4
        grad_accum = 2
        num_epochs = 3
        learning_rate = 2e-4

    print(f"🧠 [AX_TRAINER] Ejecutando SFTTrainer ({num_epochs} Épocas, LR={learning_rate}, BatchSize={per_device_bs})...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = per_device_bs,
            gradient_accumulation_steps = grad_accum,
            num_train_epochs = num_epochs,
            learning_rate = learning_rate,
            warmup_steps = 5,
            lr_scheduler_type = "cosine",
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = "/tmp/outputs",
            optim = "adamw_8bit",
            seed = 3407,
        ),
    )
    trainer.train()

    output_path = "/models/ax_model"
    # 1. Limpiar completamente archivos previos en el volumen de Modal para evitar tensores huérfanos de arquitecturas anteriores
    if os.path.exists(output_path):
        import shutil
        print(f"🧹 [AX_TRAINER] Limpiando residuos de modelos previos en {output_path}...")
        shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)

    # 2. Guardar SIEMPRE primero los adaptadores LoRA y tokenizer entrenados
    print("💾 [AX_TRAINER] Guardando adaptadores LoRA y tokenizer...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    volume.commit()
    print("✅ [AX_TRAINER] Adaptadores LoRA y tokenizer guardados correctamente.")

    # 3. Exportar a formato GGUF con tolerancia a fallos
    print("\n📦 [AX_TRAINER] Exportando modelo a formato GGUF (q4_k_m)...")
    try:
        model.save_pretrained_gguf(output_path, tokenizer, quantization_method = "q4_k_m")
        volume.commit()
        print("✅ [AX_TRAINER] ¡Exportación GGUF q4_k_m finalizada con éxito!")
    except Exception as e_gguf:
        print(f"⚠️ [AX_TRAINER] Advertencia en exportación GGUF q4_k_m: {e_gguf}")
        print("🔄 [AX_TRAINER] Intentando exportación alternativa GGUF (f16)...")
        try:
            model.save_pretrained_gguf(output_path, tokenizer, quantization_method = "f16")
            volume.commit()
            print("✅ [AX_TRAINER] ¡Exportación GGUF f16 finalizada con éxito!")
        except Exception as e_gguf2:
            print(f"⚠️ [AX_TRAINER] Omitiendo exportación GGUF: {e_gguf2}")
            print("💡 Los adaptadores LoRA se guardaron intactos y están completamente operativos para inferencia.")

    print("✅ [AX_TRAINER] ¡Entrenamiento y persistencia finalizados con éxito!")
    return "Modelo entrenado exitosamente en Modal."


# -----------------------------------------------------------------------------
# CACHE DE INFERENCIA EN MEMORIA / VRAM (MODAL CONTAINER LIFECYCLE)
# -----------------------------------------------------------------------------
_CACHED_MODEL = None
_CACHED_TOKENIZER = None
_CACHED_MODEL_PATH = None
_CACHED_MTIME = 0


@app.function(
    gpu="T4",
    timeout=300,
    scaledown_window=300,
    volumes={"/models": volume}
)
def inferir_modelo_remote(prompt: str, instruction: str = "Responder de forma amable y concisa."):
    import os
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    global _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_MODEL_PATH, _CACHED_MTIME

    # 1. Sincronizar el volumen distribuido de Modal para detectar modelos recién guardados
    try:
        volume.reload()
    except Exception:
        pass

    output_path = "/models/ax_model"
    model_to_load = output_path if os.path.exists(output_path) else "unsloth/Qwen2.5-0.5B-Instruct"

    # 2. Comprobar mtime para invalidar cache si hubo un reentrenamiento nuevo
    current_mtime = 0
    if os.path.exists(output_path):
        adapter_file = os.path.join(output_path, "adapter_model.safetensors")
        model_file = os.path.join(output_path, "model.safetensors")
        if os.path.exists(adapter_file):
            current_mtime = os.path.getmtime(adapter_file)
        elif os.path.exists(model_file):
            current_mtime = os.path.getmtime(model_file)
        else:
            current_mtime = os.path.getmtime(output_path)

    need_reload = (
        _CACHED_MODEL is None or
        _CACHED_TOKENIZER is None or
        _CACHED_MODEL_PATH != model_to_load or
        _CACHED_MTIME != current_mtime
    )

    if need_reload:
        print(f"🔄 [INFERENCIA] Cargando modelo en VRAM ({model_to_load})...")
        if _CACHED_MODEL is not None:
            try:
                del _CACHED_MODEL
                del _CACHED_TOKENIZER
                torch.cuda.empty_cache()
            except Exception:
                pass

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_to_load,
            max_seq_length = 2048,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)

        base_check = str(getattr(getattr(model, "config", None), "_name_or_path", "")).lower()
        if "qwen" in base_check or "qwen" in model_to_load.lower():
            chat_template_type = "qwen-2.5"
        elif "phi" in base_check or "phi" in model_to_load.lower():
            chat_template_type = "phi-3"
        else:
            chat_template_type = "llama-3"

        tokenizer = get_chat_template(
            tokenizer,
            chat_template = chat_template_type,
        )

        _CACHED_MODEL = model
        _CACHED_TOKENIZER = tokenizer
        _CACHED_MODEL_PATH = model_to_load
        _CACHED_MTIME = current_mtime
        print("✅ [INFERENCIA] Modelo inicializado y residente en VRAM.")
    else:
        print("⚡ [INFERENCIA] CACHE HIT: Reutilizando modelo caliente en VRAM.")
        model = _CACHED_MODEL
        tokenizer = _CACHED_TOKENIZER

    sys_instruction = instruction or "Eres un asistente de IA amable, claro y profesional."
    messages = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt"
    ).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids = inputs,
            max_new_tokens = 180,
            use_cache = True,
            temperature = 0.4,
            top_p = 0.9,
            repetition_penalty = 1.15,
            pad_token_id = tokenizer.eos_token_id,
            eos_token_id = tokenizer.eos_token_id,
        )

    input_length = inputs.shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return response or "Respuesta procesada correctamente."


import json

@app.local_entrypoint()
def main(model_name: str = DEFAULT_MODEL, dataset_json: str = None):
    dataset_list = DEFAULT_DATASET
    if dataset_json:
        try:
            dataset_list = json.loads(dataset_json)
        except Exception as e:
            print(f"⚠️ Error parseando dataset JSON: {e}")

    print(f"☁️ [AX_TRAINER] Iniciando entrenamiento de {model_name}...")
    resultado = entrenar_qlora_remote.remote(model_name, dataset_list)
    print(resultado)
    print("\n💡 Para descargar el modelo .gguf a tu equipo ejecuta:")
    print("   modal volume get ax-qlora-models ax_model/unsloth.Q4_K_M.gguf ./")


def run_ax_trainer(model_name: str = DEFAULT_MODEL, dataset: list = DEFAULT_DATASET):
    """
    Función invocable desde otros módulos o APIs de GLYNNE_LOGIC_2026.
    """
    print(f"🚀 [AX_TRAINER] Disparando fine-tuning para {model_name}...")
    with app.run():
        return entrenar_qlora_remote.remote(model_name, dataset)



