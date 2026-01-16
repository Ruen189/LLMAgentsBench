# bench.py
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Callable

import engines


# ----------------------------
# 1) Модели
# ----------------------------
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "TheBloke/saiga_mistral_7b-AWQ",
    "unsloth/Qwen3-8B-bnb-4bit",
    "Vikhrmodels/QVikhr-3-8B-Instruction",
    "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
]


# ----------------------------
# 2) Утилиты
# ----------------------------
def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:90]


def write_output(f, title: str, answer: str):
    f.write(f"=== {title} ===\n")
    f.write("MODEL OUTPUT:\n")
    f.write(answer.rstrip() + "\n\n")


# ----------------------------
# 3) Данные: out_parse/*/pdfplumber.jsonl -> PageChunk
# ----------------------------
@dataclass
class PageChunk:
    folder: str
    source_path: str
    page: int
    citation: str
    text: str


def load_page_chunks(out_parse_dir: str = "out_parse") -> List[PageChunk]:
    chunks: List[PageChunk] = []
    root = Path(out_parse_dir)
    if not root.exists():
        raise FileNotFoundError(f"Не найдена папка: {out_parse_dir}")

    for folder in sorted([p for p in root.iterdir() if p.is_dir()]):
        jsonl_path = folder / "pdfplumber.jsonl"
        if not jsonl_path.exists():
            continue

        by_page: Dict[int, List[dict]] = {}
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                page = int(obj.get("page", -1))
                if page < 0:
                    continue
                by_page.setdefault(page, []).append(obj)

        for page, items in sorted(by_page.items()):
            texts = [it.get("text", "") for it in items if it.get("text")]
            if not texts:
                continue
            source_path = items[0].get("source_path", "")
            citation = items[0].get("citation", f"{Path(source_path).name}#p={page}")
            chunks.append(
                PageChunk(
                    folder=folder.name,
                    source_path=source_path,
                    page=page,
                    citation=citation,
                    text="\n".join(texts).strip(),
                )
            )

    return chunks


# ----------------------------
# 4) Retrieval: BM25 (строим 1 раз)
# ----------------------------
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zа-яё0-9]+", text.lower())


def bm25_build(docs_tokens: List[List[str]], k1=1.5, b=0.75) -> Callable[[List[str]], List[float]]:
    import math

    N = len(docs_tokens)
    df = {}
    dl = [len(toks) for toks in docs_tokens]
    avgdl = sum(dl) / max(1, N)

    for toks in docs_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    idf = {t: math.log(1 + (N - df_t + 0.5) / (df_t + 0.5)) for t, df_t in df.items()}

    def score(query_tokens: List[str]) -> List[float]:
        scores = [0.0] * N
        q = query_tokens

        for i, doc in enumerate(docs_tokens):
            freqs = {}
            for t in doc:
                freqs[t] = freqs.get(t, 0) + 1

            denom_const = k1 * (1 - b + b * (dl[i] / avgdl))
            s = 0.0
            for t in q:
                if t not in freqs:
                    continue
                f = freqs[t]
                s += idf.get(t, 0.0) * (f * (k1 + 1)) / (f + denom_const)

            scores[i] = s

        return scores

    return score


def retrieve(chunks: List[PageChunk], scorer, query: str, top_k: int = 6) -> List[PageChunk]:
    q = tokenize(query)
    scores = scorer(q)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in ranked if scores[i] > 0]


def build_context_token_limited(passages: List[PageChunk], tokenizer, max_ctx_tokens: int) -> str:
    parts = []
    used = 0
    for p in passages:
        block = f"[{p.citation}] (folder={p.folder}, page={p.page})\n{p.text}\n"
        n = len(tokenizer(block, add_special_tokens=False).input_ids)
        if used + n > max_ctx_tokens:
            break
        parts.append(block)
        used += n
    return "\n---\n".join(parts).strip()


# ----------------------------
# 5) Бенч и аудит
# ----------------------------
BENCH_QUESTIONS = [
    "Какова цель проекта и на каком продукте он реализуется?",
    "Перечисли ожидаемые результаты/достижения по итогам семестра (пункты 1–3).",
    "Почему в архитектуре был реализован парсер как расширение для браузера и какая причина указана?",
    "Какая LLM модель указана как используемая в проекте, и почему выбрана 4-битная квантизация (AWQ INT4)?",
    "Какие типовые ошибки выявлялись при тестировании и какие меры перечислены для исправления?",
    "Перечисли функциональные требования FR-2.1–FR-2.6 для модуля поиска и скоринга компаний.",
    "Перечисли нефункциональные требования NFR-1–NFR-6.",
    "Перечисли KPI-1–KPI-6 критерии успеха проекта (с порогами/числами).",
    "Какие самые востребованные навыки и какие дефицитные компетенции в проектах перечислены?",
    "Какие задачи на следующую КТ перечислены (оба блока)?",
]


def make_qa_user_prompt(question: str, context: str) -> str:
    return f"""Ты строгий методист и проверяющий бенча.

ДАНО:
Ниже контекст из документа. Каждый фрагмент начинается с ссылки в квадратных скобках вида [CITATION].
Разрешено цитировать ТОЛЬКО эти ссылки, строго копируя их как есть.

ВОПРОС:
{question}

КОНТЕКСТ:
{context}

ПРАВИЛА:
- Отвечай строго по контексту. Никаких знаний "из головы".
- Если прямого ответа нет в контексте — напиши ровно: Не найдено в контексте.
- Запрещено придумывать цитаты/страницы/идентификаторы.
- Любое ключевое утверждение в ответе должно быть подтверждено 1–3 цитатами из контекста.

ФОРМАТ ОТВЕТА (строго, 2 строки):
Ответ: <1–3 коротких предложения или "Не найдено в контексте.">
Цитаты: <1–3 цитаты вида [....] из контекста, либо "—" если не найдено>
"""



def make_audit_user_prompt(doc_label: str, context: str, min_issues: int = 10) -> str:
    return f"""Ты строгий методист. Нужно провести аудит текста документа.

ДОКУМЕНТ:
{doc_label}

КОНТЕКСТ (каждый фрагмент начинается с [CITATION]):
{context}

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
- Запрещено придумывать факты и цитаты. Разрешены только ссылки [....], которые встречаются в начале фрагментов контекста.
- Каждое замечание должно опираться на конкретную цитату из контекста.
- Если в контексте недостаточно материала для {min_issues} замечаний, сделай столько, сколько возможно БЕЗ выдумок, и добавь строку "Контекста недостаточно для полного аудита".

ЧТО СДЕЛАТЬ:
1) Замечания: неоднозначности, непроверяемость, отсутствие критериев, логические разрывы, дубли, несогласованность терминов.
2) Для каждого замечания: (а) почему проблема, (б) как исправить — предложи конкретную замену текста/добавление (1–3 предложения).
3) 5 вопросов заказчику/куратору — каждый вопрос привяжи к цитате, которая вызвала вопрос.

ФОРМАТ (строго):
Замечания:
1) Цитата: [....]
   Проблема: ...
   Почему это важно: ...
   Как исправить (текст): "..."
2) ...

Вопросы заказчику:
1) Цитата: [....] Вопрос: ...
2) ...
"""



# ----------------------------
# 6) Запуск
# ----------------------------
def main():
    print("Выберите модель для тестирования:")
    for i, m in enumerate(MODELS, 1):
        print(f"{i}) {m}")
    idx = int(input("Введите номер модели: ").strip())
    model_name = MODELS[idx - 1]

    print(f"\nЗагрузка модели: {model_name}")
    engine = engines.make_engine(model_name)

    chunks = load_page_chunks("out_parse")
    if not chunks:
        raise RuntimeError("Не найдено ни одного out_parse/*/pdfplumber.jsonl")

    # BM25 индекс один раз на весь корпус
    docs_tokens = [tokenize(c.text) for c in chunks]
    scorer = bm25_build(docs_tokens)

    out_dir = Path("results") / slugify(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_path = out_dir / "bench.txt"
    audit_path = out_dir / "audit.txt"

    temperature = float(os.getenv("GEN_TEMPERATURE", "0.2"))
    max_new_tokens = int(os.getenv("GEN_MAX_NEW_TOKENS", "512"))

    # лимит контекста по токенам (чтобы не улетать в очень длинные промпты)
    # если у engine нет tokenizer/model (неожиданно) — упадет, но у TransformersEngine они есть
    tokenizer = engine.tokenizer
    model_max = getattr(engine.model.config, "max_position_embeddings", getattr(tokenizer, "model_max_length", 4096))
    reserve = int(os.getenv("CTX_RESERVE_TOKENS", "256"))
    max_ctx_tokens = max(256, int(model_max) - max_new_tokens - reserve)

    # явное обнуление (опционально, но наглядно)
    bench_path.write_text("", encoding="utf-8")
    audit_path.write_text("", encoding="utf-8")

    # -------- BENCH --------
    with bench_path.open("w", encoding="utf-8") as f:
        for q_i, q in enumerate(BENCH_QUESTIONS, 1):
            passages = retrieve(chunks, scorer, q, top_k=6)
            context = build_context_token_limited(passages, tokenizer, max_ctx_tokens)
            user_prompt = make_qa_user_prompt(q, context)

            ans = engine.generate(user_prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            write_output(f, f"Q{q_i}", ans)

    # -------- AUDIT --------
    with audit_path.open("w", encoding="utf-8") as f:
        by_folder: Dict[str, List[PageChunk]] = {}
        for c in chunks:
            by_folder.setdefault(c.folder, []).append(c)

        seed_queries = [
            "цель проекта ожидаемые результаты",
            "требования FR- NFR- KPI-",
            "риски ограничения методология тестирование",
            "фазы критическая точка эскалации",
        ]

        for audit_i, (folder, folder_chunks) in enumerate(sorted(by_folder.items()), 1):
            picked: List[PageChunk] = []
            seen = set()

            # retrieval делаем по всему корпусу (scorer общий), потом фильтруем по folder
            for sq in seed_queries:
                hits = retrieve(chunks, scorer, sq, top_k=30)
                hits = [h for h in hits if h.folder == folder][:4]

                for p in hits:
                    key = (p.page, p.citation)
                    if key not in seen:
                        picked.append(p)
                        seen.add(key)

            context = build_context_token_limited(picked, tokenizer, max_ctx_tokens)
            doc_label = folder_chunks[0].source_path or folder
            user_prompt = make_audit_user_prompt(doc_label, context, min_issues=10)

            ans = engine.generate(user_prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            write_output(f, f"AUDIT{audit_i} folder={folder}", ans)

    print(f"\nГотово. Результаты: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
