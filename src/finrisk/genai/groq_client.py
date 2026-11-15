"""Groq client: retries with backoff, on-disk cache, clear errors.

    python -m finrisk.genai.groq_client --list-models     # models your key can use right now
    python -m finrisk.genai.groq_client --ping            # one tiny request to test the key

Groq retires and renames models over time. The model comes from GROQ_MODEL in .env (or --model);
if it is not available you get a message telling you how to list the current ones.
Responses are cached by (model, messages, temperature), so re-running the report costs nothing.
The free tier is rate limited: 429 errors are retried, honouring the Retry-After header.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from finrisk.config import PROJECT_ROOT, get_settings


class LLMError(RuntimeError):
    pass


class GroqLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 cache_dir: Path | None = None, max_retries: int = 6,
                 min_interval: float = 2.0, sleep=time.sleep, client=None):
        s = get_settings()
        self.model = model or s.groq_model
        self.max_retries = max_retries
        self.min_interval = min_interval
        self._sleep = sleep
        self._last_call = 0.0
        self.cache_dir = Path(cache_dir) if cache_dir else PROJECT_ROOT / ".cache" / "groq"
        self.calls = 0
        self.cache_hits = 0
        if client is None:
            key = api_key or s.groq_api_key
            if not key or key == "your_key_here":
                raise LLMError("No Groq API key. Create a free key at console.groq.com, put it in .env "
                               "as GROQ_API_KEY=..., or run with --no-llm.")
            from groq import Groq
            client = Groq(api_key=key)
        self.client = client

    # --------------------------------------------------------------------- cache
    def _cache_path(self, messages, temperature, max_tokens) -> Path:
        blob = json.dumps({"model": self.model, "messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens}, sort_keys=True)
        return self.cache_dir / (hashlib.sha256(blob.encode("utf-8")).hexdigest() + ".json")

    # ----------------------------------------------------------------------- chat
    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 700) -> str:
        path = self._cache_path(messages, temperature, max_tokens)
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]

        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                self._sleep(wait)
            try:
                self._last_call = time.time()
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                text = (resp.choices[0].message.content or "").strip()
                self.calls += 1
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"model": self.model, "text": text}), encoding="utf-8")
                return text
            except Exception as exc:  # noqa: BLE001
                kind = self._classify(exc)
                if kind in ("auth", "model", "bad"):
                    raise LLMError(self._explain(kind, exc)) from exc
                if kind == "unknown":
                    raise
                delay = self._retry_after(exc) or min(60.0, 2.0 ** (attempt + 1))
                self._sleep(delay)
        raise LLMError(f"Groq request failed after {self.max_retries} attempts. "
                       "Wait a minute (free-tier rate limit) and run again; cached answers are kept.")

    # --------------------------------------------------------------------- errors
    @staticmethod
    def _classify(exc: Exception) -> str:
        try:
            import groq
        except ImportError:
            return "unknown"
        is_ = lambda *names: any(isinstance(exc, getattr(groq, n)) for n in names if hasattr(groq, n))
        if is_("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"):
            return "retry"
        if is_("AuthenticationError", "PermissionDeniedError"):
            return "auth"
        if is_("NotFoundError"):
            return "model"
        if is_("BadRequestError"):
            msg = str(exc).lower()
            return "model" if ("model" in msg or "decommission" in msg) else "bad"
        return "unknown"

    def _explain(self, kind: str, exc: Exception) -> str:
        if kind == "auth":
            return "Groq rejected the API key. Check GROQ_API_KEY in .env (console.groq.com)."
        if kind == "model":
            return (f"Model '{self.model}' is not available on Groq (it may have been retired). "
                    "List current models with:  python -m finrisk.genai.groq_client --list-models  "
                    "then set GROQ_MODEL in .env.  Original error: " + str(exc))
        return f"Groq rejected the request: {exc}"

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        try:
            value = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
            return min(120.0, float(value)) if value else None
        except Exception:  # noqa: BLE001
            return None

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list().data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--ping", action="store_true")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    llm = GroqLLM(model=args.model)
    if args.list_models:
        print("\n".join(llm.list_models()))
    if args.ping or not args.list_models:
        print(f"[{llm.model}]", llm.chat([{"role": "user", "content": "Reply with the single word: ready"}],
                                         max_tokens=10))


if __name__ == "__main__":
    main()
