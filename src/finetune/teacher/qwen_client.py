"""Thin adapter over the local Qwen (a.k.a. "openclaw"/Delphi) vLLM server.

Everything token-heavy in the teacher pipeline goes through THIS module — grounded
QA generation and the adversarial judge pass. Nothing else talks to the model directly.

The model is an OpenAI-compatible vLLM endpoint; the base_url and served model name
are supplied by the caller from configs/teacher.yaml (nothing here is hardcoded):
  * concurrency ceiling MAX_NUM_SEQS  -> we cap client concurrency to match; more
                        in-flight requests just queue server-side and add latency.
  * reasoning model     -> chat_template_kwargs.enable_thinking=false, temperature 0
                        for deterministic, verbatim output (thinking ON makes it
                        "reason"/rewrite instead of answering).

Design:
  * stdlib only (urllib + concurrent.futures) so it runs in any env without pip.
  * `chat()`            one blocking request with retry/backoff.
  * `generate_batch()`  many prompts -> many strings, saturating the GPU at
                        `max_concurrency` worker threads.
  * `classify_batch()`  many prompts -> many parsed JSON objects (best-effort
                        extraction), for label/score tasks.

The batch calls are the ONLY throughput path. They return results positionally
aligned with the input list; a failed item yields a QwenError sentinel (or None
for classify) rather than dropping it, so callers can persist partial progress.
"""
from __future__ import annotations

import json
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed as _as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ── Tuning defaults (endpoint + model are required, supplied from config) ─────
DEFAULT_MAX_CONCURRENCY = 32          # == vLLM MAX_NUM_SEQS on this box
DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_RETRIES = 5
RETRYABLE_HTTP = {408, 409, 429, 500, 502, 503, 504}


class QwenError(RuntimeError):
    """A request that exhausted retries. Carries the last error detail."""


@dataclass
class Usage:
    """Rolling token counters across all calls made by one client instance."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    retries: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, prompt: int, completion: int, retries: int = 0) -> None:
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.requests += 1
            self.retries += retries

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "requests": self.requests,
                "retries": self.retries,
            }


class QwenClient:
    """Batched, concurrency-limited, retrying client for the local Qwen server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        provider: str = "vllm",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # provider controls how thinking is disabled:
        #   vllm   -> OpenAI /chat/completions + chat_template_kwargs.enable_thinking
        #   ollama -> native /api/chat + "think" (ollama ignores chat_template_kwargs
        #             AND the /no_think soft switch, so a reasoning model returns EMPTY
        #             content unless think is disabled here).
        if provider not in ("vllm", "ollama"):
            raise ValueError(f"provider must be vllm|ollama, got {provider}")
        self.provider = provider
        import re as _re
        self._root = _re.sub(r"/v1/?$", "", self.base_url)  # host root, no /v1
        self.model = model
        # Never exceed the server's sequence ceiling; extra threads only add latency.
        self.max_concurrency = max(1, min(max_concurrency, DEFAULT_MAX_CONCURRENCY))
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.usage = Usage()

    # ── Low-level single request ──────────────────────────────────────────────
    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """One chat completion. Blocks, retries transient errors, returns the
        assistant text. Raises QwenError if all retries fail."""
        temp = self.temperature if temperature is None else temperature
        if self.provider == "ollama":
            url = self._root + "/api/chat"
            payload: Dict[str, Any] = {
                "model": self.model, "messages": messages, "stream": False,
                "think": self.enable_thinking,   # the only switch ollama honours
                "options": {"temperature": temp, "num_predict": max_tokens},
            }
        else:
            url = self.base_url + "/chat/completions"
            payload = {
                "model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temp,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            }
            if response_format is not None:
                payload["response_format"] = response_format

        body = json.dumps(payload).encode()
        last_err = ""
        retries = 0
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode())
                if self.provider == "ollama":
                    self.usage.add(data.get("prompt_eval_count", 0),
                                   data.get("eval_count", 0), retries=retries)
                    return (data.get("message", {}) or {}).get("content", "") or ""
                usage = data.get("usage", {}) or {}
                self.usage.add(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    retries=retries,
                )
                return data["choices"][0]["message"]["content"] or ""
            except Exception as exc:  # noqa: BLE001 - classify then decide retry
                status = getattr(exc, "code", None)
                detail = ""
                if hasattr(exc, "read"):
                    try:
                        detail = exc.read().decode()[:300]
                    except Exception:
                        pass
                last_err = f"{type(exc).__name__}: {exc} {detail}".strip()
                retryable = status is None or status in RETRYABLE_HTTP
                if attempt < self.max_retries and retryable:
                    retries += 1
                    self._sleep_backoff(attempt)
                    continue
                break
        self.usage.add(0, 0, retries=retries)
        raise QwenError(last_err)

    def _sleep_backoff(self, attempt: int) -> None:
        # Exponential backoff with full jitter, capped at 60s.
        delay = min(60.0, (2 ** attempt)) * (0.5 + random.random() / 2)
        time.sleep(delay)

    # ── Batch APIs (the throughput path) ──────────────────────────────────────
    def generate_batch(
        self,
        prompts: Sequence[str],
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        on_result=None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Run `prompts` concurrently (<= max_concurrency in flight). Returns a
        list aligned with `prompts`; a failed item is a QwenError instance (not
        raised) so a batch never loses good results to one bad prompt.

        `on_result(index, result)` is called as each item completes (for progress
        / incremental checkpointing)."""
        def _one(prompt: str) -> Any:
            messages: List[Dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            try:
                return self.chat(messages, max_tokens=max_tokens, temperature=temperature,
                                 response_format=response_format)
            except QwenError as e:
                return e

        return self._map(prompts, _one, on_result)

    def classify_batch(
        self,
        prompts: Sequence[str],
        system: Optional[str] = None,
        max_tokens: int = 256,
        on_result=None,
    ) -> List[Optional[Dict[str, Any]]]:
        """Like generate_batch, but each output is parsed into a JSON object
        (best-effort). Items that error or don't yield JSON become None so the
        caller can retry just those."""
        raw = self.generate_batch(
            prompts, system=system, max_tokens=max_tokens, on_result=None
        )
        out: List[Optional[Dict[str, Any]]] = []
        for i, r in enumerate(raw):
            parsed = None if isinstance(r, QwenError) else extract_json(r)
            out.append(parsed)
            if on_result is not None:
                on_result(i, parsed)
        return out

    def _map(self, items: Sequence[Any], fn, on_result) -> List[Any]:
        results: List[Any] = [None] * len(items)
        if not items:
            return results
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            futs = {ex.submit(fn, item): i for i, item in enumerate(items)}
            for fut in _as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()
                if on_result is not None:
                    on_result(i, results[i])
        return results

    # ── Health ────────────────────────────────────────────────────────────────
    def health(self) -> bool:
        """True if the server responds and serves `self.model`."""
        try:
            req = urllib.request.Request(self.base_url + "/models")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            return any(m.get("id") == self.model for m in data.get("data", []))
        except Exception:
            return False


def extract_json(text: str) -> Optional[Any]:
    """Best-effort parse of a JSON object/array from model text: whole string,
    then a ```json fence, then the first '{'/'[' … last '}'/']' span."""
    text = (text or "").strip()
    if not text:
        return None
    candidates = [text]
    if "```" in text:
        for part in text.split("```"):
            p = part.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p[:1] in "{[":
                candidates.append(p)
    for opener, closer in (("{", "}"), ("[", "]")):
        a, b = text.find(opener), text.rfind(closer)
        if a != -1 and b > a:
            candidates.append(text[a:b + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Return every top-level {...} object parseable from text, in order. Handles
    JSON arrays, JSONL (newline-delimited objects, as ollama emits), and objects
    embedded in prose. Empty list if none."""
    text = text or ""
    out: List[Dict[str, Any]] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            out.append(obj)
                    except Exception:
                        pass
                    start = -1
    return out
