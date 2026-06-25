from __future__ import annotations

import json
import os
import re
from typing import Any

import requests


class QwenClient:
    def __init__(self):
        self.backend = os.getenv("LLM_BACKEND", "transformers").lower()
        self.model_id = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
        self._pipe = None

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 1000) -> str:
        if self.backend == "ollama":
            return self._generate_ollama(system_prompt, user_prompt)
        if self.backend == "mock":
            return self._generate_mock(user_prompt)
        return self._generate_transformers(system_prompt, user_prompt, max_new_tokens=max_new_tokens)

    def _generate_transformers(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
        if self._pipe is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            self._pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tokenizer = self._pipe.tokenizer
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = self._pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)[0]["generated_text"]
        return out[len(prompt):].strip()

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str:
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
        res = requests.post("http://localhost:11434/api/chat", json=payload, timeout=180)
        res.raise_for_status()
        return res.json()["message"]["content"]

    def _generate_mock(self, user_prompt: str) -> str:
        chunk_ids = re.findall(r"CHUNK_ID:\s*(C\d+)", user_prompt)
        return json.dumps({
            "user_query": _extract_between(user_prompt, "USER QUERY:", "RETRIEVED CHUNKS:").strip(),
            "retrieved_chunk_ids": chunk_ids[:10],
            "discrepancies_found": False,
            "discrepancies": [],
        })


def _extract_between(text: str, start: str, end: str) -> str:
    try:
        return text.split(start, 1)[1].split(end, 1)[0]
    except Exception:
        return ""


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {
        "user_query": "",
        "retrieved_chunk_ids": [],
        "discrepancies_found": True,
        "discrepancies": [
            {
                "inconsistency_id": "INC-PARSE",
                "severity": "High",
                "category": "Model Output Parsing",
                "title": "Model did not return valid JSON",
                "nicely_articulated_discrepancy": "The Qwen model response could not be parsed as JSON. The prompt or max token setting may need adjustment.",
                "evidence": [{"chunk_id": "N/A", "source": "Model output", "quote_or_evidence": text[:500]}],
                "suggested_resolution": "Retry with fewer chunks, lower max tokens, or stricter JSON formatting instruction.",
            }
        ],
    }


def sort_and_limit_discrepancies(data: dict[str, Any], limit: int = 3) -> dict[str, Any]:
    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    discrepancies = data.get("discrepancies") or []
    discrepancies = sorted(discrepancies, key=lambda d: order.get(d.get("severity", "Low"), 4))[:limit]
    data["discrepancies"] = discrepancies
    data["discrepancies_found"] = bool(discrepancies)
    return data
