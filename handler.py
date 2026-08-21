import base64
import io
import json
import os
import random
import re

import numpy as np
import runpod
import torch
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = os.getenv("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
TRANSLATION_MODEL_ID = os.getenv(
    "TRANSLATION_MODEL_ID",
    "Helsinki-NLP/opus-mt-ROMANCE-en"
)

PROMPT_JSON_PATH = os.getenv("PROMPT_JSON_PATH", "prompt.json")
DETAILS_JSON_PATH = os.getenv("DETAILS_JSON_PATH", "details.json")

DEFAULT_NUM_IMAGES = int(os.getenv("DEFAULT_NUM_IMAGES", "1"))
MAX_NUM_IMAGES = int(os.getenv("MAX_NUM_IMAGES", "4"))
DEFAULT_RES_CHOICE = os.getenv("DEFAULT_RES_CHOICE", "2")
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "18"))
DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "0.0"))
USE_CPU_OFFLOAD = os.getenv("USE_CPU_OFFLOAD", "0") == "1"

RESOLUTIONS = {
    "1": (1024, 1024),
    "2": (720, 1280),
    "3": (576, 1024),
}

PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
else:
    dtype = torch.float32

print("Device:", device)
print("dtype:", dtype)


# ============================================================
# LOAD TRANSLATOR
# ============================================================

print("Loading translator...")

translator_tokenizer = AutoTokenizer.from_pretrained(
    TRANSLATION_MODEL_ID
)

translator_model = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL_ID
)

translator_model = translator_model.to("cpu")
translator_model.eval()

print("Translator loaded.")


# ============================================================
# LOAD IMAGE MODEL
# ============================================================

print("Loading Z-Image model...")

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)

if device == "cuda":
    if USE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
else:
    pipe = pipe.to(device)

print("Image model loaded.")


# ============================================================
# HELPERS
# ============================================================

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def traduzir_prompt(texto):
    if not texto or not texto.strip():
        return texto

    try:
        inputs = translator_tokenizer(
            texto,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.inference_mode():
            translated = translator_model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=4,
                early_stopping=True,
            )

        texto_en = translator_tokenizer.decode(
            translated[0],
            skip_special_tokens=True,
        )

        return texto_en.strip()

    except Exception as e:
        print(f"Aviso, erro ao traduzir prompt: {e}")
        return texto


def load_json_file(path, default):
    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Aviso, erro ao ler {path}: {e}")
        return default


def load_prompts_with_id(path):
    data = load_json_file(path, [])

    if not isinstance(data, list):
        return {}

    prompts = {}

    for item in data:
        if (
            isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("prompt"), str)
        ):
            prompts[item["id"].strip()] = item["prompt"]

    return prompts


def load_details(path):
    data = load_json_file(path, [])

    if not isinstance(data, list):
        return {}

    details = {}

    for item in data:
        if (
            isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("prompt"), str)
        ):
            opts = [s.strip() for s in item["prompt"].split(",")]
            opts = [s for s in opts if s]

            if opts:
                details[item["id"].strip()] = opts

    return details


def apply_details_overrides(prompt, details_overrides):
    if not details_overrides:
        return prompt, {}

    applied = {}

    def repl(match):
        key = match.group(1).strip()

        if key in details_overrides:
            val = details_overrides[key]

            if isinstance(val, str) and val.strip():
                val = val.strip()
                applied[key] = val
                return val

        return match.group(0)

    expanded = PLACEHOLDER_RE.sub(repl, prompt)
    return expanded, applied


def expand_prompt_with_details_random(prompt, details):
    if not details:
        return prompt, {}

    used = {}
    chosen = {}

    def repl(match):
        key = match.group(1).strip()

        if key not in details or not details[key]:
            return match.group(0)

        opts = details[key]

        used.setdefault(key, set())
        chosen.setdefault(key, [])

        available = [o for o in opts if o not in used[key]]

        if not available:
            available = opts

        val = random.choice(available)

        used[key].add(val)
        chosen[key].append(val)

        return val

    expanded = PLACEHOLDER_RE.sub(repl, prompt)
    return expanded, chosen


def resolve_dimensions(input_data):
    width = input_data.get("width")
    height = input_data.get("height")

    if width is not None and height is not None:
        return int(width), int(height)

    res_choice = str(input_data.get("res_choice", DEFAULT_RES_CHOICE))

    if res_choice not in RESOLUTIONS:
        res_choice = DEFAULT_RES_CHOICE

    return RESOLUTIONS[res_choice]


def get_prompt_base(input_data):
    prompt = input_data.get("prompt")
    prompt_id = input_data.get("prompt_id")

    if prompt_id:
        prompts = load_prompts_with_id(PROMPT_JSON_PATH)

        if prompt_id not in prompts:
            raise ValueError(
                f"prompt_id '{prompt_id}' não encontrado em {PROMPT_JSON_PATH}"
            )

        return prompts[prompt_id], prompt_id

    if prompt and isinstance(prompt, str) and prompt.strip():
        return prompt.strip(), None

    raise ValueError("Envie 'prompt' ou 'prompt_id'")


def build_prompt(prompt_base, details, details_mode, details_overrides):
    prompt_after_overrides, applied_overrides = apply_details_overrides(
        prompt_base,
        details_overrides
    )

    chosen_random = {}

    if details_mode == "random":
        prompt_final, chosen_random = expand_prompt_with_details_random(
            prompt_after_overrides,
            details
        )
    else:
        prompt_final = prompt_after_overrides

    return prompt_final, {
        "overrides": applied_overrides,
        "random_choices": chosen_random,
    }


def make_generator(seed):
    if device == "cuda":
        return torch.Generator(device="cuda").manual_seed(seed)

    return torch.Generator().manual_seed(seed)


def generate_images(
    prompt_base,
    prompt_id,
    width,
    height,
    num_images,
    steps,
    guidance_scale,
    details_mode,
    details_overrides,
    translate_to_english,
):
    details = load_details(DETAILS_JSON_PATH)
    results = []

    for i in range(num_images):
        prompt_final, details_info = build_prompt(
            prompt_base=prompt_base,
            details=details,
            details_mode=details_mode,
            details_overrides=details_overrides,
        )

        if translate_to_english:
            prompt_en = traduzir_prompt(prompt_final)
        else:
            prompt_en = prompt_final

        seed = int(np.random.randint(0, 2**31 - 1))
        generator = make_generator(seed)

        print(
            f"Gerando {i + 1}/{num_images}, seed={seed}, {width}x{height}"
        )
        print("PROMPT ORIGINAL:", prompt_final)
        print("PROMPT ENVIADO:", prompt_en)

        result = pipe(
            prompt=prompt_en,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
            guidance_scale=guidance_scale,
            max_sequence_length=1024,
            num_images_per_prompt=1,
            output_type="pil",
        )

        img = result.images[0]

        results.append({
            "index": i + 1,
            "prompt_id": prompt_id,
            "seed": seed,
            "width": width,
            "height": height,
            "prompt_original": prompt_final,
            "prompt_en": prompt_en,
            "details_info": details_info,
            "image_base64": image_to_base64(img),
            "mime_type": "image/png",
        })

        if device == "cuda":
            torch.cuda.empty_cache()

    return results


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(job):
    try:
        input_data = job.get("input", {})

        prompt_base, prompt_id = get_prompt_base(input_data)

        width, height = resolve_dimensions(input_data)

        num_images = int(input_data.get("num_images", DEFAULT_NUM_IMAGES))
        num_images = max(1, min(num_images, MAX_NUM_IMAGES))

        steps = int(input_data.get("steps", DEFAULT_STEPS))
        guidance_scale = float(
            input_data.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
        )

        details_mode = str(
            input_data.get("details_mode", "random")
        ).strip().lower()

        if details_mode not in ("random", "literal"):
            details_mode = "random"

        details_overrides = input_data.get("details", {})
        if not isinstance(details_overrides, dict):
            details_overrides = {}

        translate_to_english = bool(
            input_data.get("translate_to_english", True)
        )

        results = generate_images(
            prompt_base=prompt_base,
            prompt_id=prompt_id,
            width=width,
            height=height,
            num_images=num_images,
            steps=steps,
            guidance_scale=guidance_scale,
            details_mode=details_mode,
            details_overrides=details_overrides,
            translate_to_english=translate_to_english,
        )

        return {
            "status": "success",
            "model_id": MODEL_ID,
            "translation_model_id": TRANSLATION_MODEL_ID,
            "device": device,
            "dtype": str(dtype),
            "count": len(results),
            "results": results,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


runpod.serverless.start({"handler": handler})