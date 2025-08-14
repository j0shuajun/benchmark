#!/usr/bin/env python3
"""Asynchronous LLM benchmark runner."""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm

ANSWER_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}


def load_cache(cache_file: Path | None) -> dict:
    if cache_file and cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache_file: Path | None, cache: dict) -> None:
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)


async def query_llm(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    payload = {"model": model, "seed": 1234}
    if url.endswith("/chat/completions"):
        payload["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    elif url.endswith("/responses"):
        payload["input"] = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload["reasoning"] = {"effort": "high"}
    else:
        raise ValueError("Unsupported URL. Use /chat/completions or /responses")

    resp = await client.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    return data["output"][0]["content"][0]["text"]


async def evaluate_subject(
    benchmark_name: str,
    subject_name: str,
    df: pd.DataFrame,
    system_prompt: str,
    client: httpx.AsyncClient,
    url: str,
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    cache: dict,
    cache_file: Path | None,
) -> list[str]:
    responses = ["" for _ in range(len(df))]

    async def process(idx: int, row: pd.Series, pbar: tqdm):
        question_key = f"{benchmark_name}/{subject_name}/{row['question']}"
        if question_key in cache and cache[question_key]:
            responses[idx] = cache[question_key]
            pbar.update(1)
            return

        parts = []
        if (
            "RAG_Contexts" in df.columns
            and isinstance(row.get("RAG_Contexts", ""), str)
            and row.get("RAG_Contexts", "").strip()
        ):
            parts.append("[Related Contexts]")
            parts.append(row["RAG_Contexts"])
        parts.append(f"Q. {row['question']}")
        parts.append(f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}")
        user_prompt = "\n".join(parts)

        for attempt in range(max_retries):
            try:
                async with semaphore:
                    ans = await query_llm(
                        client, url, model, system_prompt, user_prompt
                    )
                break
            except Exception:
                if attempt + 1 == max_retries:
                    ans = ""
                else:
                    await asyncio.sleep(1)
        cache[question_key] = ans
        save_cache(cache_file, cache)
        responses[idx] = ans
        pbar.update(1)

    pbar = tqdm(total=len(df), desc=subject_name, leave=False)
    tasks = [asyncio.create_task(process(i, row, pbar)) for i, row in df.iterrows()]
    await asyncio.gather(*tasks)
    pbar.close()
    return responses


async def evaluate_benchmark(
    name: str, args, cache: dict, cache_file: Path | None
) -> None:
    data_dir = Path("data") / name
    subjects = sorted([p for p in data_dir.glob("*.csv")])
    system_prompt_file = data_dir / "system_prompt.txt"
    system_prompt = (
        system_prompt_file.read_text(encoding="utf-8")
        if system_prompt_file.exists()
        else ""
    )

    sanitized_name = name.replace("/", "__")
    sanitized_model = getattr(args, "sanitized_model", args.model).replace("/", "__")
    timestamp = getattr(args, "timestamp", datetime.now().strftime("%y%m%d-%H%M%S"))

    print(f"Benchmark {name}: {len(subjects)} subjects")
    for sub in subjects:
        df = pd.read_csv(sub)
        print(f"  {sub.stem}: {len(df)} questions")

    result_root = Path("results") / f"{sanitized_name}_{sanitized_model}_{timestamp}"
    result_root.mkdir(parents=True, exist_ok=True)

    summary = []
    total_correct = 0
    total_questions = 0

    async with httpx.AsyncClient(
        timeout=None, headers={"Authorization": f"Bearer {args.api_key}"}
    ) as client:
        semaphore = asyncio.Semaphore(args.max_concurrency)
        for sub in tqdm(subjects, desc=name):
            df = pd.read_csv(sub)
            sub_keys = [
                f"{name}/{sub.stem}/{q}" for q in df["question"]
            ]
            if all((k in cache and cache[k]) for k in sub_keys):
                print(f"Skipping {sub.stem}: using cached responses")
                pbar = tqdm(total=len(df), desc=sub.stem, leave=False)
                pbar.update(len(df))
                pbar.close()
                responses = [cache[k] for k in sub_keys]
            else:
                responses = await evaluate_subject(
                    name,
                    sub.stem,
                    df,
                    system_prompt,
                    client,
                    args.url,
                    args.model,
                    semaphore,
                    args.max_retries,
                    cache,
                    cache_file,
                )
            df["response"] = responses
            df["answer_letter"] = df["answer"].map(ANSWER_MAP)

            pattern = r"(?i)(?:Answer\s*:|Answer\s*:​​​​​​|답변\s*:|정답\s*:|답\s*:|Answer\s*|So Answer\s*|That's\s*|The best answer is\s*|The correct choice is\s*|The answer is\s*)[ \t]*([A-D]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"

            df["parsed_response"] = df["response"].str.extract(pattern, expand=False)

            df["correct"] = (
                df["answer_letter"].str.upper()
                == df["parsed_response"].fillna("").str.strip().str.upper()
            )
            acc = df["correct"].mean()
            summary.append({"subject": sub.stem, "accuracy": acc, "total": len(df)})
            total_correct += int(df["correct"].sum())
            total_questions += len(df)
            out_df = df.drop(columns=["answer_letter", "correct"])
            out_df.to_csv(result_root / f"{sub.stem}_result.csv", index=False)

    overall_acc = total_correct / total_questions if total_questions else 0.0
    summary.append(
        {"subject": "overall", "accuracy": overall_acc, "total": total_questions}
    )
    pd.DataFrame(summary).to_csv(result_root / "summary.csv", index=False)

    print("\nFinal summary:")
    for item in summary:
        print(
            f"  {item['subject']}: {item['accuracy']:.2%} ({item['total']} questions)"
        )
    print("\n")
    save_cache(cache_file, cache)


def main() -> None:
    parser = argparse.ArgumentParser(description="Asynchronous LLM benchmark runner")
    parser.add_argument(
        "--runlist",
        type=str,
        required=True,
        help="Space separated list of benchmarks to run",
    )
    parser.add_argument("--url", type=str, required=True, help="LLM server URL")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--max_concurrency", type=int, default=5, help="Max concurrent requests"
    )
    parser.add_argument(
        "--max_retries", type=int, default=3, help="Max retries per request"
    )
    parser.add_argument(
        "--api_key", type=str, required=True, help="API key for Authorization header"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help=(
            "Path to cache file. If not provided, a new file is created under .cache."
        ),
    )
    args = parser.parse_args()

    sanitized_model = args.model.replace("/", "__")
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")

    if args.cache:
        cache_file = Path(args.cache)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache = load_cache(cache_file)
        parts = cache_file.stem.split("_")
        if len(parts) > 1:
            timestamp = parts[-1]
    else:
        cache_dir = Path(".cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{sanitized_model}_{timestamp}.json"
        cache = {}

    args.sanitized_model = sanitized_model
    args.timestamp = timestamp

    runlist = args.runlist.split()
    for idx, bench in enumerate(runlist, 1):
        print(f"\nStarting benchmark {idx}/{len(runlist)}: {bench}")
        asyncio.run(evaluate_benchmark(bench, args, cache, cache_file))
    save_cache(cache_file, cache)


if __name__ == "__main__":
    main()
