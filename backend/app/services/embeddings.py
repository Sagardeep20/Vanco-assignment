"""Embedding provider abstraction: real local 512-d model with fallback.

Providers (selected via ``EMBEDDING_PROVIDER``):

- ``deterministic`` — the original hashed bag-of-words embedding
  (512-d, L2-normalized). No dependencies, no downloads. Kept as the
  reliable fallback and for tests.
- ``local`` — a CLIP-compatible text encoder (``clip-ViT-B-32``, 512-d)
  loaded **only** from packages and weight files already present on this
  machine. Raises :class:`EmbeddingUnavailableError` with a clear message
  when unavailable. NEVER downloads anything. The same OpenCLIP model
  also serves image embeddings via :func:`embed_image`
  (``encode_image`` + the model's own preprocessing, L2-normalized).
- ``auto`` (default) — ``local`` when available, else ``deterministic``.

Asset processing and search both go through :func:`embed_text`, so the
query and the stored vectors always come from the SAME provider.
Images are stored from actual pixels (:func:`embed_image`) while text
queries use :func:`embed_text` — both live in the one shared CLIP
joint space, which is what makes cross-modal search work.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from pathlib import Path
from typing import Protocol

from app.core.config import settings
from app.models.asset import EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
DIMENSION = EMBEDDING_DIMENSION  # 512; must match the pgvector column.


class EmbeddingUnavailableError(RuntimeError):
    """The requested embedding provider cannot be used on this machine."""


class EmbeddingProvider(Protocol):
    """Interface every embedding provider implements."""

    name: str
    dim: int

    def embed(self, text: str) -> list[float] | None:
        """Return an L2-normalized embedding, or None for blank input."""
        ...

    def embed_image(self, image: object) -> list[float] | None:
        """Return an L2-normalized embedding for a PIL image (None input -> None).

        Providers without an image encoder raise
        :class:`EmbeddingUnavailableError`; callers fall back to text.
        """
        ...


def deterministic_embedding(
    text: str, dim: int = DIMENSION
) -> list[float] | None:
    """Hashed bag-of-words embedding (the original implementation).

    Each token increments one hashed bucket; the vector is L2-normalized
    so pgvector cosine distance is meaningful. Deterministic, dependency
    free. Returns None for blank input so callers keep the column NULL.
    """
    if not text or not text.strip():
        return None
    vec = [0.0] * dim
    for token in _TOKEN_RE.findall(text.lower()):
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return None
    return [v / norm for v in vec]


def l2_normalize(vec: list[float]) -> list[float]:
    """Return the L2-normalized copy of a vector."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return [v / norm for v in vec]


class DeterministicProvider:
    """Fallback provider: hashed embeddings, always available."""

    name = "deterministic"
    dim = DIMENSION

    def embed(self, text: str) -> list[float] | None:
        return deterministic_embedding(text, self.dim)

    def embed_image(self, image: object) -> list[float] | None:
        if image is None:
            return None
        raise EmbeddingUnavailableError(
            "The deterministic provider has no image encoder; "
            "callers should fall back to text embeddings."
        )


class LocalSemanticProvider:
    """CLIP-compatible local provider (512-d). Never downloads.

    Loads ``clip-ViT-B-32`` (or ``LOCAL_EMBEDDING_MODEL_PATH``) from
    already-installed packages and already-cached weights only: the
    Hugging Face offline flags are forced during load, so missing files
    raise instead of triggering a download.
    """

    name = "local"
    dim = DIMENSION

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = (
            model_name or getattr(settings, "local_embedding_model", "clip-ViT-B-32")
        )
        self._model: object | None = None
        self._loaded = False

    def _load(self) -> object:
        """Load and cache the underlying encoder (offline only)."""
        if self._loaded:
            if self._model is None:
                raise EmbeddingUnavailableError("Local embedding model failed to load.")
            return self._model
        self._loaded = True
        offline = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        previous = {k: os.environ.get(k) for k in offline}
        os.environ.update(offline)
        try:
            self._model = self._load_offline()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if self._model is None:
            raise EmbeddingUnavailableError(
                f"Local embedding model '{self.model_name}' is not available "
                "on this machine (offline check, no download attempted)."
            )
        return self._model

    def _load_offline(self) -> object | None:
        """Try known local backends. Returns None when nothing is usable."""
        explicit = getattr(settings, "local_embedding_model_path", "") or ""
        if explicit and Path(explicit).exists():
            loaded = self._try_open_clip(str(explicit))
            if loaded is not None:
                return loaded
            loaded = self._try_sentence_transformers(str(explicit))
            if loaded is not None:
                return loaded
            return self._try_transformers_clip(str(explicit))
        loaded = self._try_open_clip(self.model_name)
        if loaded is not None:
            return loaded
        loaded = self._try_sentence_transformers(self.model_name)
        if loaded is not None:
            return loaded
        return self._try_transformers_clip(self.model_name)

    @staticmethod
    def _resolve_open_clip_model(model_ref: str) -> tuple[str, str] | None:
        """Map a model ref to an open_clip (arch, pretrained) pair.

        Only the already-cached ViT-B-32/OpenAI weights (or an explicit
        local weight file) are ever used, so no other model is downloaded.
        Returns None for unknown names (caller then tries other backends).
        """
        explicit = getattr(settings, "local_embedding_model_path", "") or ""
        if explicit and Path(explicit).is_file():
            return ("ViT-B-32", explicit)
        if model_ref and Path(model_ref).is_file():
            return ("ViT-B-32", model_ref)
        normalized = (model_ref or "").strip().lower().replace("/", "-").replace("_", "-")
        if normalized in {"", "clip-vit-b-32", "vit-b-32", "vit-b32"}:
            return ("ViT-B-32", "openai")
        return None

    def _try_open_clip(self, model_ref: str) -> object | None:
        """Load ViT-B-32/openai via the installed open_clip package (offline).

        Runs under the HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE flags set by
        :meth:`_load`, so a missing cached checkpoint raises (offline) and
        is caught below instead of triggering any download.
        """
        try:
            import open_clip
        except ImportError:
            logger.info("open_clip is not installed; skipping.")
            return None
        resolved = self._resolve_open_clip_model(model_ref)
        if resolved is None:
            logger.info(
                "open_clip backend only serves cached ViT-B-32/openai weights; "
                "skipping model ref '%s' (no download attempted).", model_ref,
            )
            return None
        arch, pretrained = resolved
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained, device="cpu"
            )
            tokenizer = open_clip.get_tokenizer(arch)
            model.eval()
        except Exception as exc:
            logger.info("open_clip '%s/%s' not loadable offline: %s", arch, pretrained, exc)
            return None
        proj = getattr(model, "text_projection", None)
        if proj is not None:
            try:
                if int(proj.shape[-1]) != self.dim:
                    logger.info("open_clip '%s/%s' is not 512-d; skipping.", arch, pretrained)
                    return None
            except Exception:
                pass
        return {
            "backend": "open_clip",
            "model": model,
            "tokenizer": tokenizer,
            "preprocess": preprocess,
        }

    def _try_sentence_transformers(self, model_ref: str) -> object | None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.info("sentence-transformers is not installed; skipping.")
            return None
        try:
            model = SentenceTransformer(model_ref, device="cpu")
        except Exception as exc:
            logger.info("Local model '%s' not loadable offline: %s", model_ref, exc)
            return None
        try:
            model_dim = model.get_sentence_embedding_dimension()
        except Exception:
            model_dim = None
        if model_dim != self.dim:
            logger.info(
                "Local model '%s' has dim=%s, need %d; skipping.",
                model_ref, model_dim, self.dim,
            )
            return None
        return model

    def _try_transformers_clip(self, model_ref: str) -> object | None:
        try:
            from transformers import CLIPModel, CLIPTokenizer
        except ImportError:
            logger.info("transformers is not installed; skipping.")
            return None
        try:
            model = CLIPModel.from_pretrained(model_ref, local_files_only=True)
            tokenizer = CLIPTokenizer.from_pretrained(model_ref, local_files_only=True)
        except Exception as exc:
            logger.info("Local CLIP '%s' not loadable offline: %s", model_ref, exc)
            return None
        if getattr(model.config, "projection_dim", self.dim) != self.dim:
            logger.info("Local CLIP '%s' is not 512-d; skipping.", model_ref)
            return None
        return (model, tokenizer)

    def _encode(self, model: object, text: str) -> list[float]:
        if isinstance(model, dict) and model.get("backend") == "open_clip":
            import torch

            clip_model = model["model"]
            tokenizer = model["tokenizer"]
            tokens = tokenizer([text])
            with torch.no_grad():
                features = clip_model.encode_text(tokens)[0]
            return [float(v) for v in features.tolist()]
        if hasattr(model, "encode"):  # sentence-transformers
            vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=False)[0]
            return [float(v) for v in vec]
        # transformers CLIP (model, tokenizer) tuple, CPU, no grad.
        import torch

        clip_model, tokenizer = model
        inputs = tokenizer([text], padding=True, return_tensors="pt")
        with torch.no_grad():
            features = clip_model.get_text_features(**inputs)[0]
        return [float(v) for v in features.tolist()]

    def _encode_image(self, model: object, image: object) -> list[float]:
        """Encode raw pixels with the OpenCLIP image tower (CPU, no grad)."""
        if not (isinstance(model, dict) and model.get("backend") == "open_clip"):
            raise EmbeddingUnavailableError(
                "Image embeddings need the open_clip backend; "
                "this loaded model has no image encoder."
            )
        import torch

        clip_model = model["model"]
        preprocess = model["preprocess"]
        pixels = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = clip_model.encode_image(pixels)[0]
        return [float(v) for v in features.tolist()]

    def _finalize(self, raw: list[float]) -> list[float]:
        if len(raw) != self.dim:
            raise EmbeddingUnavailableError(
                f"Local model returned dim={len(raw)}, expected {self.dim}."
            )
        return l2_normalize(raw)

    def embed(self, text: str) -> list[float] | None:
        if not text or not text.strip():
            return None
        try:
            model = self._load()
        except EmbeddingUnavailableError:
            raise
        except Exception as exc:
            raise EmbeddingUnavailableError(
                f"Local embedding model '{self.model_name}' failed: {exc}"
            ) from exc
        try:
            raw = self._encode(model, text)
        except Exception as exc:
            raise EmbeddingUnavailableError(
                f"Local embedding inference failed: {exc}"
            ) from exc
        return self._finalize(raw)

    def embed_image(self, image: object) -> list[float] | None:
        """Embed a PIL image's pixels (512-d, L2-normalized; None -> None)."""
        if image is None:
            return None
        try:
            model = self._load()
        except EmbeddingUnavailableError:
            raise
        except Exception as exc:
            raise EmbeddingUnavailableError(
                f"Local embedding model '{self.model_name}' failed: {exc}"
            ) from exc
        try:
            raw = self._encode_image(model, image)
        except EmbeddingUnavailableError:
            raise
        except Exception as exc:
            raise EmbeddingUnavailableError(
                f"Local image embedding inference failed: {exc}"
            ) from exc
        return self._finalize(raw)


_provider_cache: dict[str, object] = {}


def clear_provider_cache() -> None:
    """Drop cached provider instances (mainly for tests)."""
    _provider_cache.clear()


def get_provider(name: str | None = None) -> EmbeddingProvider:
    """Resolve a provider by name: ``auto`` | ``local`` | ``deterministic``.

    - ``auto``: local model when already available, else deterministic.
    - ``local``: local model or raise :class:`EmbeddingUnavailableError`.
    - ``deterministic``: always the hash fallback.
    Unknown names raise ValueError. Never downloads anything.
    """
    requested = (name if name is not None else settings.embedding_provider or "auto")
    requested = requested.lower().strip()
    if requested == "deterministic":
        return DeterministicProvider()
    if requested == "local":
        provider = _local_singleton()
        if provider is None:
            raise EmbeddingUnavailableError(
                "EMBEDDING_PROVIDER=local but no 512-d CLIP-compatible model "
                "is available locally (checked open_clip / "
                "sentence-transformers / transformers caches offline; "
                "no download attempted). "
                "Install weights or switch to auto/deterministic."
            )
        return provider
    if requested == "auto":
        provider = _local_singleton()
        if provider is not None:
            return provider
        logger.info("No local embedding model; using deterministic fallback.")
        return DeterministicProvider()
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={requested!r}; "
        "expected one of: auto, local, deterministic."
    )


def _local_singleton() -> LocalSemanticProvider | None:
    """Cached local provider, or None when unavailable (never raises)."""
    cached = _provider_cache.get("local")
    if isinstance(cached, LocalSemanticProvider):
        return cached
    if cached == "unavailable":
        return None
    candidate = LocalSemanticProvider()
    try:
        candidate._load()
    except EmbeddingUnavailableError as exc:
        logger.info("Local embedding model unavailable: %s", exc)
        _provider_cache["local"] = "unavailable"
        return None
    _provider_cache["local"] = candidate
    return candidate


def embed_text(text: str, provider_name: str | None = None) -> list[float] | None:
    """Embed text with the configured provider (blank -> None).

    Used by both asset processing and search so queries and stored
    vectors always come from the SAME provider.
    """
    if not text or not text.strip():
        return None
    return get_provider(provider_name).embed(text)


def embed_image(image: object, provider_name: str | None = None) -> list[float] | None:
    """Embed a PIL image's pixels with the configured provider (None -> None).

    Uses the OpenCLIP image tower (``encode_image`` + the model's own
    preprocessing), 512-d and L2-normalized, so image vectors share the
    joint space with :func:`embed_text` query vectors. Raises
    :class:`EmbeddingUnavailableError` when the resolved provider has no
    image encoder (e.g. the deterministic fallback); callers fall back
    to text embeddings in that case.
    """
    if image is None:
        return None
    return get_provider(provider_name).embed_image(image)
