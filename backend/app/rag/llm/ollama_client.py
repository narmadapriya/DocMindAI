import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from app.core.logging import get_logger, log_event
from app.core.performance import OLLAMA_NUM_PREDICT

logger = get_logger(__name__)


class OllamaModelError(RuntimeError):
    """Raised when the local Ollama model fails."""


class OllamaClient:
    """
    Local Ollama client for DocMindAI.

    Default model:
        qwen2.5vl:3b

    Uses:
        POST /api/chat

    Supports:
        - text prompts
        - multimodal image prompts
        - optional Ollama JSON output mode
        - bounded generation/context options
        - keep-alive for faster repeated local inference

    The client remains dependency-free and communicates directly
    with the local Ollama HTTP API.
    """

    _SUPPORTED_IMAGE_SUFFIXES = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | float | None = None,
        keep_alive: str | None = None,
    ):
        self.base_url = (
            base_url
            or os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            )
        ).rstrip("/")

        self.model = (
            model
            or os.getenv(
                "OLLAMA_MODEL",
                "qwen2.5vl:3b",
            )
        )

        if timeout is None:
            timeout = float(
                os.getenv(
                    "OLLAMA_TIMEOUT",
                    "600",
                )
            )

        self.timeout = float(timeout)

        # Ollama normally unloads idle models after a short period.
        # Keeping the 3B model resident avoids repeated cold-load costs
        # on normal Chat & Ask turns. This is configurable for users who
        # need a smaller memory footprint.
        self.keep_alive = (
            keep_alive
            if keep_alive is not None
            else os.getenv(
                "OLLAMA_KEEP_ALIVE",
                "30m",
            )
        )

    # =========================================================
    # AVAILABILITY
    # =========================================================

    def is_available(self) -> bool:
        """Check whether the local Ollama server is reachable."""

        try:
            with urllib.request.urlopen(
                self.base_url + "/api/tags",
                timeout=5,
            ) as response:
                return response.status == 200

        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ):
            return False

        except Exception:
            return False

    # =========================================================
    # MODEL AVAILABILITY
    # =========================================================

    def is_model_available(self) -> bool:
        """Check whether the configured model exists in Ollama."""

        try:
            with urllib.request.urlopen(
                self.base_url + "/api/tags",
                timeout=5,
            ) as response:
                if response.status != 200:
                    return False

                data = json.loads(
                    response.read().decode("utf-8")
                )

        except Exception:
            return False

        for model in data.get("models", []):
            name = (
                model.get("name")
                or model.get("model")
                or ""
            )

            if name == self.model:
                return True

        return False

    # =========================================================
    # IMAGE ENCODING
    # =========================================================

    @staticmethod
    @lru_cache(maxsize=16)
    def _encode_resized_image_cached(
        path_text: str,
        mtime_ns: int,
        file_size: int,
        max_edge: int,
    ) -> str:
        """Resize only the model-request copy; never modify the source."""

        del mtime_ns, file_size  # included only to invalidate the cache
        path = Path(path_text)

        try:
            from PIL import Image

            with Image.open(path) as image:
                image.load()
                width, height = image.size
                largest = max(width, height)

                if largest > max_edge:
                    scale = float(max_edge) / float(largest)
                    size = (
                        max(1, int(round(width * scale))),
                        max(1, int(round(height * scale))),
                    )
                    resampling = getattr(
                        getattr(Image, "Resampling", Image),
                        "LANCZOS",
                    )
                    image = image.resize(size, resampling)

                if image.mode not in {"RGB", "RGBA"}:
                    image = image.convert("RGB")

                buffer = io.BytesIO()
                image.save(buffer, format="PNG", optimize=True)
                raw = buffer.getvalue()

        except (ImportError, OSError, ValueError):
            raw = path.read_bytes()

        return base64.b64encode(raw).decode("ascii")

    @classmethod
    def _encode_images(
        cls,
        images: Sequence[str] | None,
        *,
        max_edge: int | None = None,
    ) -> list[str]:
        """
        Convert genuine image assets to base64.

        ``max_edge=None`` preserves the frozen behavior exactly. A resized
        model-input copy is created only when the internal caller explicitly
        requests it; the stored source image is never changed.
        """

        encoded_images: list[str] = []

        for image in images or []:
            path = Path(str(image))

            if not path.exists() or not path.is_file():
                continue

            if path.suffix.lower() not in cls._SUPPORTED_IMAGE_SUFFIXES:
                continue

            try:
                if max_edge is None:
                    encoded = base64.b64encode(
                        path.read_bytes()
                    ).decode("ascii")
                else:
                    bounded_edge = max(640, int(max_edge))
                    stat = path.stat()
                    encoded = cls._encode_resized_image_cached(
                        str(path.resolve()),
                        int(stat.st_mtime_ns),
                        int(stat.st_size),
                        bounded_edge,
                    )

                encoded_images.append(encoded)

            except (OSError, ValueError, TypeError):
                continue

        return encoded_images

    # =========================================================
    # CHAT
    # =========================================================

    def chat(
        self,
        *,
        prompt: str,
        images: Sequence[str] | None = None,
        temperature: float = 0.0,
        num_predict: int | None = None,
        num_ctx: int | None = None,
        json_mode: bool = False,
        keep_alive: str | None = None,
        image_max_edge: int | None = None,
        retry_json_on_images: bool = True,
    ) -> str:
        """
        Send a text or multimodal request to Ollama.

        ``json_mode`` uses Ollama's native JSON output constraint.
        It does not change DocMindAI's reasoning workflow; it only
        constrains serialization of the model output expected by the
        existing ReasoningAgent.

        Normal requests keep the frozen generation budget. If Ollama
        returns syntactically incomplete JSON on the first attempt
        (most commonly because the local 3B model reached the generation
        limit while closing the structured response), one recovery
        attempt is made with a slightly larger output budget. This is
        serialization recovery only; the prompt, evidence, model, RAG
        flow, API contract, and answer rules remain unchanged.
        """

        if not prompt or not prompt.strip():
            raise OllamaModelError(
                "Ollama prompt cannot be empty."
            )

        encoded_images = self._encode_images(
            images,
            max_edge=image_max_edge,
        )

        message: dict = {
            "role": "user",
            "content": prompt,
        }

        if encoded_images:
            message["images"] = encoded_images

        if num_predict is None:
            num_predict = OLLAMA_NUM_PREDICT

        if int(num_predict) <= 0:
            raise OllamaModelError(
                "num_predict must be greater than zero."
            )

        if num_ctx is not None and int(num_ctx) <= 0:
            raise OllamaModelError(
                "num_ctx must be greater than zero."
            )

        base_num_predict = int(num_predict)

        # Keep the normal frozen budget untouched. Recovery is used only
        # after malformed JSON from a JSON-mode request. Cap the recovery
        # budget so a bad local response cannot create a very long retry.
        recovery_num_predict = min(
            512,
            max(
                base_num_predict,
                base_num_predict * 2,
            ),
        )

        started = time.perf_counter()

        transient_http_codes = {
            500,
            502,
            503,
            504,
        }

        data: dict = {}
        content = ""
        max_attempts = 2
        final_num_predict = base_num_predict

        for attempt in range(1, max_attempts + 1):
            attempt_num_predict = (
                recovery_num_predict
                if attempt > 1
                else base_num_predict
            )
            final_num_predict = attempt_num_predict

            options: dict = {
                "temperature": float(temperature),
                "num_predict": attempt_num_predict,
            }

            if num_ctx is not None:
                options["num_ctx"] = int(num_ctx)

            payload: dict = {
                "model": self.model,
                "stream": False,
                "messages": [message],
                "options": options,
                "keep_alive": (
                    self.keep_alive
                    if keep_alive is None
                    else keep_alive
                ),
            }

            if json_mode:
                payload["format"] = "json"

            request = urllib.request.Request(
                self.base_url + "/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout,
                ) as response:
                    raw_response = (
                        response.read()
                        .decode("utf-8")
                    )

                    data = json.loads(
                        raw_response
                    )

                content = (
                    (data.get("message") or {}).get("content")
                    or ""
                ).strip()

                if not content:
                    ollama_error = (
                        data.get("error")
                        or ""
                    )

                    if ollama_error:
                        raise OllamaModelError(
                            "Ollama returned an error "
                            f"for '{self.model}': "
                            f"{ollama_error}"
                        )

                    if attempt < max_attempts:
                        log_event(
                            logger,
                            "llm_retry",
                            model=self.model,
                            attempt=attempt,
                            reason="empty_response",
                        )
                        time.sleep(0.20)
                        continue

                    raise OllamaModelError(
                        "Ollama returned an empty "
                        f"response for '{self.model}'."
                    )

                # Native JSON mode can still produce an incomplete object
                # if generation stops at the output-token boundary. The
                # ReasoningAgent already has repair logic for minor format
                # deviations, so only retry the clearly malformed first
                # response. On the final attempt, return the content and let
                # the existing parser/repair path decide whether it is usable.
                if (
                    json_mode
                    and attempt < max_attempts
                    and (retry_json_on_images or not encoded_images)
                ):
                    try:
                        json.loads(content)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        log_event(
                            logger,
                            "llm_retry",
                            model=self.model,
                            attempt=attempt,
                            reason="incomplete_model_json",
                            done_reason=data.get("done_reason"),
                            num_predict=attempt_num_predict,
                            recovery_num_predict=recovery_num_predict,
                        )
                        time.sleep(0.20)
                        continue

                break

            except urllib.error.HTTPError as exc:
                try:
                    error_body = (
                        exc.read()
                        .decode("utf-8")
                    )
                except Exception:
                    error_body = ""

                if (
                    exc.code in transient_http_codes
                    and attempt < max_attempts
                ):
                    log_event(
                        logger,
                        "llm_retry",
                        model=self.model,
                        attempt=attempt,
                        reason=f"http_{exc.code}",
                    )
                    time.sleep(0.20)
                    continue

                raise OllamaModelError(
                    "Ollama HTTP request failed "
                    f"for '{self.model}' "
                    f"(HTTP {exc.code}). "
                    f"{error_body}"
                ) from exc

            except urllib.error.URLError as exc:
                if attempt < max_attempts:
                    log_event(
                        logger,
                        "llm_retry",
                        model=self.model,
                        attempt=attempt,
                        reason="url_error",
                    )
                    time.sleep(0.20)
                    continue

                raise OllamaModelError(
                    "Unable to connect to Ollama "
                    f"at '{self.base_url}'. "
                    f"Model: '{self.model}'. "
                    f"Error: {exc.reason}"
                ) from exc

            except TimeoutError as exc:
                # Do not retry timeouts. Retrying a long local-model timeout
                # can double an already slow Chat & Ask request.
                raise OllamaModelError(
                    "Ollama request timed out "
                    f"after {self.timeout:.0f} seconds "
                    f"for model '{self.model}'. "
                    "Qwen2.5-VL may still be loading "
                    "or processing the multimodal input."
                ) from exc

            except json.JSONDecodeError as exc:
                if attempt < max_attempts:
                    log_event(
                        logger,
                        "llm_retry",
                        model=self.model,
                        attempt=attempt,
                        reason="invalid_response_json",
                    )
                    time.sleep(0.20)
                    continue

                raise OllamaModelError(
                    "Ollama returned invalid JSON "
                    f"for model '{self.model}'."
                ) from exc

            except OSError as exc:
                if attempt < max_attempts:
                    log_event(
                        logger,
                        "llm_retry",
                        model=self.model,
                        attempt=attempt,
                        reason="os_error",
                    )
                    time.sleep(0.20)
                    continue

                raise OllamaModelError(
                    "Ollama network/OS error "
                    f"for model '{self.model}': "
                    f"{exc}"
                ) from exc

            except OllamaModelError:
                raise

            except Exception as exc:
                raise OllamaModelError(
                    "Ollama request failed "
                    f"for '{self.model}': "
                    f"{exc}"
                ) from exc

        if not content:
            raise OllamaModelError(
                "Ollama returned an empty "
                f"response for '{self.model}'."
            )

        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        log_event(
            logger,
            "llm",
            model=self.model,
            image_count=len(encoded_images),
            prompt_chars=len(prompt),
            num_predict=final_num_predict,
            num_ctx=(
                int(num_ctx)
                if num_ctx is not None
                else None
            ),
            json_mode=bool(json_mode),
            latency_ms=latency_ms,
            ollama_total_ms=self._ollama_duration_ms(
                data.get("total_duration")
            ),
            ollama_load_ms=self._ollama_duration_ms(
                data.get("load_duration")
            ),
            prompt_eval_ms=self._ollama_duration_ms(
                data.get("prompt_eval_duration")
            ),
            eval_ms=self._ollama_duration_ms(
                data.get("eval_duration")
            ),
            prompt_eval_count=self._safe_int(
                data.get("prompt_eval_count")
            ),
            eval_count=self._safe_int(
                data.get("eval_count")
            ),
            done_reason=data.get("done_reason"),
        )

        return content

    @staticmethod
    def _ollama_duration_ms(
        value: object,
    ) -> float | None:
        """Convert Ollama nanosecond timing fields to milliseconds."""

        try:
            if value is None:
                return None

            return round(
                float(value) / 1_000_000.0,
                2,
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(
        value: object,
    ) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None
