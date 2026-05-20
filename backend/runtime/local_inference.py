from __future__ import annotations
import requests


class LocalInferenceError(Exception):
    """Raised when local inference endpoint fails."""


class LocalInference:
    """
    Synchronous HTTP client for local OpenAI-compatible inference endpoints.
    Responsibilities:
    - call active vLLM/LM Studio runtime
    - build chat completion payload
    - extract and return content string
    - surface clean errors to caller
    """

    DEFAULT_TIMEOUT = 180
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_MAX_TOKENS = 300

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        system_prompt: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.system_prompt = system_prompt

    def infer(
        self,
        *,
        port: int,
        model: str,
        prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt: str | None = None,
        response_format: dict | None = None,
    ) -> str:
        """
        Call inference endpoint and return assistant content.
        system_prompt: per-call override — falls back to instance default.
        """
        if not prompt.strip():
            raise LocalInferenceError("prompt must not be blank.")

        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        messages = self._build_messages(prompt, system_prompt or self.system_prompt)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        try:
            response = self._post(url, payload)
            data = response.json()
            return self._extract_content(data, url)

        except LocalInferenceError:
            raise
        except requests.exceptions.ConnectionError as exc:
            raise LocalInferenceError(
                f"Cannot connect to inference endpoint at {url}. "
                f"Is the runtime running on port {port}?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LocalInferenceError(
                f"Inference timed out after {self.timeout}s at {url}."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise LocalInferenceError(
                f"HTTP {exc.response.status_code} from {url}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except Exception as exc:
            raise LocalInferenceError(
                f"Inference failed at {url}: {exc}"
            ) from exc

    def _post(self, url: str, payload: dict) -> requests.Response:
        response = requests.post(url, json=payload, timeout=self.timeout)
        if (
            response.status_code == 400
            and "response_format" in payload
            and "response_format" in response.text
        ):
            fallback_payload = dict(payload)
            fallback_payload.pop("response_format", None)
            response = requests.post(url, json=fallback_payload, timeout=self.timeout)
        response.raise_for_status()
        return response

    def health_check(self, port: int) -> bool:
        """
        Ping /health endpoint. Returns True if runtime is reachable.
        Non-raising — safe to call before inference.
        """
        try:
            response = requests.get(
                f"http://127.0.0.1:{port}/health",
                timeout=5,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: str | None,
    ) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _extract_content(
        data: dict,
        url: str,
    ) -> str:
        try:
            choices = data.get(
                "choices",
                [],
            )

            if not choices:
                raise LocalInferenceError(
                    f"Empty choices from {url}."
                )

            message = choices[0].get(
                "message",
                {},
            )

            content = message.get(
                "content"
            )

            if isinstance(
                content,
                str,
            ) and content.strip():
                return content

            reasoning = (
                message.get(
                    "reasoning"
                )
                or message.get(
                    "reasoning_content"
                )
            )

            if isinstance(
                reasoning,
                str,
            ) and reasoning.strip():
                return reasoning

            if isinstance(
                content,
                list,
            ):
                parts: list[str] = []

                for item in content:
                    if (
                        isinstance(
                            item,
                            dict,
                        )
                        and item.get("text")
                    ):
                        parts.append(
                            item["text"]
                        )

                merged = "\n".join(
                    parts
                ).strip()

                if merged:
                    return merged

            raise LocalInferenceError(
                f"No usable content "
                f"returned from {url}."
            )

        except Exception as exc:
            raise LocalInferenceError(
                f"Malformed response "
                f"from {url}: {exc}"
            ) from exc
