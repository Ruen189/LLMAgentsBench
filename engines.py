# engines.py
import os
from types import SimpleNamespace

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    AwqConfig,
)

SYSTEM_STRICT = (
    "Ты строгий методист. "
    "Отвечай строго по контексту. "
    "Если ответа нет в контексте — напиши 'Не найдено в контексте'. "
    "Для ключевых утверждений приводи цитаты в формате [doc#p=N]."
)

# --- NEW: явный список исключений из vLLM (эти остаются на Transformers/bnb) ---
VLLM_EXCLUDE_MODELS = {
    "unsloth/Qwen3-8B-bnb-4bit",
    "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
}

def is_awq(model_id: str) -> bool:
    s = model_id.lower()
    return ("-awq" in s) or ("awq" in s)

def is_gptq(model_id: str) -> bool:
    s = model_id.lower()
    return ("-gptq" in s) or ("gptq" in s)

def is_bnb(model_id: str) -> bool:
    s = model_id.lower()
    return ("bnb-4bit" in s) or ("bitsandbytes" in s)

# --- NEW: vLLM по умолчанию для всех, кроме исключений ---
def is_vllm(model_id: str) -> bool:
    return model_id not in VLLM_EXCLUDE_MODELS

def make_bnb_config():
    mode = os.getenv("BNB_MODE", "4bit").lower().strip()
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

def make_awq_config():
    return AwqConfig(bits=4, do_fuse=False)

def build_prompt(model_id: str, tokenizer, user: str) -> str:
    messages = [{"role": "user", "content": f"{SYSTEM_STRICT}\n\n{user}"}]

    prompt = None
    try:
        has_template = bool(getattr(tokenizer, "chat_template", None))
        if has_template and hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = None

    if prompt is None:
        if "mistral" in model_id.lower() or "saiga" in model_id.lower():
            prompt = f"<s>[INST] {SYSTEM_STRICT}\n\n{user} [/INST]"
        else:
            prompt = f"{SYSTEM_STRICT}\n\n{user}\n\nОтвет:"
    return prompt


class TransformersEngine:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

        kwargs = dict(
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )

        if is_gptq(model_id):
            pass
        elif is_bnb(model_id):
            # bnb-4bit репозиторий: не передаём свой BitsAndBytesConfig
            pass
        elif is_awq(model_id):
            kwargs["quantization_config"] = make_awq_config()
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["quantization_config"] = make_bnb_config()

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, user: str, temperature: float = 0.2, max_new_tokens: int = 512) -> str:
        prompt = build_prompt(self.model_id, self.tokenizer, user)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.05,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        return text.strip()

class VllmEngine:
    def __init__(self, model_id: str):
        self.model_id = model_id
        if model_id in VLLM_EXCLUDE_MODELS:
            raise RuntimeError(f"Model is excluded from vLLM: {model_id}")

        max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
        max_num_seqs = int(os.getenv("VLLM_MAX_NUM_SEQS", "1"))
        enforce_eager = os.getenv("VLLM_ENFORCE_EAGER", "1").strip() == "1"
        target_util = float(os.getenv("VLLM_GPU_MEM_UTIL", "0.90"))

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        cfg.max_position_embeddings = max_model_len
        self.model = SimpleNamespace(config=cfg)

        try:
            self.tokenizer.model_max_length = max_model_len
        except Exception:
            pass

        from vllm import LLM

        load_format = None
        if is_awq(model_id):
            quant = "awq"
        elif is_gptq(model_id):
            quant = "gptq"
        else:
            quant = "bitsandbytes"
            load_format = "bitsandbytes"

        free_b, total_b = torch.cuda.mem_get_info()
        free_ratio = free_b / total_b
        safety = float(os.getenv("VLLM_GPU_MEM_SAFETY", "0.03"))
        gpu_mem_util = max(0.10, min(target_util, free_ratio - safety))

        llm_kwargs = dict(
            model=model_id,
            dtype=os.getenv("VLLM_DTYPE", "auto"),
            tensor_parallel_size=int(os.getenv("VLLM_TP", "1")),
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            quantization=quant,
        )
        if load_format is not None:
            llm_kwargs["load_format"] = load_format

        self.llm = LLM(**llm_kwargs)

    def generate(self, user: str, temperature: float = 0.2, max_new_tokens: int = 512) -> str:
        from vllm import SamplingParams

        prompt = build_prompt(self.model_id, self.tokenizer, user)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.05,
            max_tokens=max_new_tokens,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

def make_engine(model_name: str):
    if model_name in VLLM_EXCLUDE_MODELS:
        return TransformersEngine(model_name)

    # Все остальные — vLLM
    return VllmEngine(model_name)
