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

# Dataset por defecto enriquecido para evitar sobreajuste (overfitting)
DEFAULT_DATASET = [
    {
        "instruction": "Eres un asistente virtual amable y profesional de atención al cliente.",
        "input": "¿Cuáles son los horarios de atención?",
        "output": "Atendemos de Lunes a Viernes de 9:00 AM a 6:00 PM."
    },
    {
        "instruction": "Eres un asistente virtual amable y profesional de atención al cliente.",
        "input": "¿Qué servicios ofrece la plataforma?",
        "output": "Ofrecemos agentes de voz inteligentes, chat automatizado, entrenamiento de modelos IA y soluciones personalizadas para empresas."
    },
    {
        "instruction": "Eres un asistente virtual amable y profesional de atención al cliente.",
        "input": "¿A qué te dedicas o cuál es tu función?",
        "output": "Soy un modelo de inteligencia artificial diseñado para atender consultas, responder dudas y brindar información sobre nuestro negocio de forma rápida y eficiente."
    },
    {
        "instruction": "Eres un asistente virtual amable y profesional de atención al cliente.",
        "input": "¿De qué es el negocio?",
        "output": "Nuestra empresa ofrece soluciones de automatización con inteligencia artificial, agentes conversacionales de voz y desarrollo de modelos privados."
    }
]

# Definición del contenedor en la nube de Modal
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
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

    # Configurar la plantilla oficial de ChatML para instruct (Qwen / Llama / etc.)
    chat_template_type = "qwen-2.5" if "qwen" in model_name.lower() else "llama-3"
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
        lora_dropout = 0.05,
        bias = "none",
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

    print("\n📦 [AX_TRAINER] Guardando modelo en formato GGUF...")
    output_path = "/models/ax_model"
    model.save_pretrained_gguf(output_path, tokenizer, quantization_method = "q4_k_m")
    volume.commit()
    print("✅ [AX_TRAINER] ¡Entrenamiento y exportación finalizados con éxito!")
    return "Modelo entrenado exitosamente en Modal."


@app.function(gpu="T4", timeout=300, volumes={"/models": volume})
def inferir_modelo_remote(prompt: str, instruction: str = "Responder de forma amable y concisa."):
    import os
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    output_path = "/models/ax_model"
    if not os.path.exists(output_path):
        output_path = "unsloth/Qwen2.5-0.5B-Instruct"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = output_path,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    chat_template_type = "qwen-2.5" if "qwen" in output_path.lower() else "llama-3"
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = chat_template_type,
    )

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

    outputs = model.generate(
        input_ids = inputs,
        max_new_tokens = 256,
        use_cache = True,
        temperature = 0.7,
        top_p = 0.9,
    )

    input_length = inputs.shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return response


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



