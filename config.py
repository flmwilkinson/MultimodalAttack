# -*- coding: utf-8 -*-
"""COMET global configuration (Ollama-first for local use, all values overridable via environment variables).

All model calls use the OpenAI-compatible /chat/completions endpoint, so local Ollama
(http://localhost:11434/v1) can replace DeepSeek / OpenAI / Anthropic directly.

Common overrides:
    OLLAMA_BASE_URL   default http://localhost:11434/v1
    TEXT_MODEL        text model      default qwen3-vl:30b-a3b-instruct
    VISION_MODEL      vision model    default qwen3-vl:30b-a3b-instruct
    ADVERSARIAL_MODEL adversarial/jailbreak default huihui_ai/qwen3-vl-abliterated:30b-a3b-instruct-q8_0
    JUDGE_MODEL       StrongREJECT judge default qwen3-vl:32b
    OUTPUT_LANGUAGE   output language default en
    RUBRIC_LANGUAGE   rubric language default en
    IMAGE_GEN_BACKEND image backend   default local_pillow (local_pillow | draw_things | remote)
"""

import os


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no third-party dependency).

    Existing environment variables always win, so shell exports override the
    file. Supports ``KEY=value``, ``export KEY=value`` and ``#`` comments.
    """
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_load_dotenv()


# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------


def _env_str(key, default=""):
    value = os.getenv(key)
    return default if value is None or value == "" else value


def _env_int(key, default):
    try:
        return int(os.getenv(key, "") or default)
    except (TypeError, ValueError):
        return default


def _env_float(key, default):
    try:
        return float(os.getenv(key, "") or default)
    except (TypeError, ValueError):
        return default


def _env_bool(key, default):
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Basic connection parameters
# ---------------------------------------------------------------------------

OLLAMA_HOST = _env_str("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_BASE_URL = _env_str("OLLAMA_BASE_URL", f"{OLLAMA_HOST.rstrip('/')}/v1")
# The OpenAI-compatible base URL must end with /v1 (Ollama serves the chat
# endpoint at /v1/chat/completions). Auto-correct a common .env mistake where
# OLLAMA_BASE_URL is set to the bare host without /v1 (causes 404 on calls).
if not OLLAMA_BASE_URL.rstrip("/").endswith("/v1"):
    OLLAMA_BASE_URL = OLLAMA_BASE_URL.rstrip("/") + "/v1"
OLLAMA_API_KEY = _env_str("OLLAMA_API_KEY", "ollama")

# Default models
DEFAULT_TEXT_MODEL = _env_str("TEXT_MODEL", "qwen3-vl:30b-a3b-instruct")
DEFAULT_VISION_MODEL = _env_str("VISION_MODEL", "qwen3-vl:30b-a3b-instruct")
DEFAULT_ADVERSARIAL_MODEL = _env_str(
    "ADVERSARIAL_MODEL", "huihui_ai/qwen3-vl-abliterated:30b-a3b-instruct-q8_0"
)
DEFAULT_JUDGE_MODEL = _env_str("JUDGE_MODEL", "qwen3-vl:32b")

# Shared text / vision clients (core/llm_client.py)
TEXT_API_KEY = _env_str("TEXT_API_KEY", OLLAMA_API_KEY)
TEXT_BASE_URL = _env_str("TEXT_BASE_URL", OLLAMA_BASE_URL)

VISION_API_KEY = _env_str("VISION_API_KEY", OLLAMA_API_KEY)
VISION_BASE_URL = _env_str("VISION_BASE_URL", OLLAMA_BASE_URL)

# Image generation (tools/image_generator.py)
IMAGE_GEN_API_KEY = _env_str("IMAGE_GEN_API_KEY", OLLAMA_API_KEY)
IMAGE_GEN_BASE_URL = _env_str("IMAGE_GEN_BASE_URL", OLLAMA_BASE_URL)
IMAGE_GEN_MODEL = _env_str("IMAGE_GEN_MODEL", "x/z-image-turbo:latest")
# local_pillow: deterministic local entity board rendering; remote: use IMAGE_GEN_BASE_URL; ollama_image: use Ollama native /api/generate
IMAGE_GEN_BACKEND = _env_str("IMAGE_GEN_BACKEND", "local_pillow")
IMAGE_GEN_WIDTH = _env_int("IMAGE_GEN_WIDTH", 1024)
IMAGE_GEN_HEIGHT = _env_int("IMAGE_GEN_HEIGHT", 768)
IMAGE_GEN_TIMEOUT = _env_int("IMAGE_GEN_TIMEOUT", 300)

# Draw Things local API (IMAGE_GEN_BACKEND=draw_things)
DRAW_THINGS_ENDPOINT = _env_str("DRAW_THINGS_ENDPOINT", "http://127.0.0.1:7860/api/v1/generate")
DRAW_THINGS_STEPS = _env_int("DRAW_THINGS_STEPS", 20)
DRAW_THINGS_SAMPLER = _env_str("DRAW_THINGS_SAMPLER", "DPM++ 2M Karras")
DRAW_THINGS_CFG_SCALE = _env_float("DRAW_THINGS_CFG_SCALE", 7.0)
DRAW_THINGS_NEGATIVE_PROMPT = _env_str("DRAW_THINGS_NEGATIVE_PROMPT", "")

# ---------------------------------------------------------------------------
# Task-level LLM configuration
# ---------------------------------------------------------------------------

def _task(model, temperature=0.2, max_tokens=2048, retry_limit=3):
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "retry_limit": retry_limit,
    }


_TEXT = DEFAULT_TEXT_MODEL
_ADV = DEFAULT_ADVERSARIAL_MODEL
_VIS = DEFAULT_VISION_MODEL

LLM_TASKS = {
    # Stage 3: entity extraction / distractor entity generation
    "entity_extraction": _task(_TEXT, 0.2, 2048, 3),
    # Quality assessment / entity matching (text-only judgment)
    "quality_assessment": _task(_TEXT, 0.1, 1536, 3),
    "entity_matching": _task(_TEXT, 0.1, 1536, 3),
    # Stage 7: visual grounding / visual polishing
    "vlm_recognition": _task(_VIS, 0.1, 4096, 3),
    "vlm_element_location": _task(_VIS, 0.1, 4096, 3),
    "vlm_text_polishing": _task(_VIS, 0.2, 4096, 3),
    # Text rewriting / polishing
    "text_rewriting": _task(_ADV, 0.4, 4096, 3),
    "text_polishing": _task(_TEXT, 0.3, 4096, 3),
    "text_redaction": _task(_TEXT, 0.2, 3072, 3),
    "adversarial_rewrite": _task(_ADV, 0.5, 4096, 3),
    # Image prompt composition / payload assembly
    "narrative_composition": _task(_TEXT, 0.4, 4096, 3),
    "payload_assembly": _task(_ADV, 0.3, 4096, 3),
    # Rubric parsing and language normalization
    "rubric_parsing": _task(_TEXT, 0.1, 3072, 3),
    "rubric_language_normalization": _task(_TEXT, 0.1, 3072, 3),
}

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE = _env_float("DEFAULT_TEMPERATURE", 0.2)
MAX_TOKENS = _env_int("MAX_TOKENS", 4096)
OUTPUT_LANGUAGE = _env_str("OUTPUT_LANGUAGE", "en").lower()
RUBRIC_LANGUAGE = _env_str("RUBRIC_LANGUAGE", "en").lower()

# Entity extraction strategy (transmuter/services/system_prompts.py)
ENTITY_EXTRACTION_MODE = {
    "mode": _env_str("ENTITY_EXTRACTION_MODE", "dynamic"),
    "limits": {"target_object": 2, "action": 3, "concept": 2, "default": 3},
    "soft_limit": 5,
}

# Distraction entities (core/entity_extractor.py)
DISTRACTION_ENTITY_CONFIG = {
    "enabled": _env_bool("DISTRACTION_ENTITY_ENABLED", True),
    "mode": "ratio",
    "count_ratio": 0.3,
    "min_count": 2,
    "max_count": 5,
    "categories": ["natural", "everyday", "abstract"],
}

# Location lexicon: leave empty for generalization (core/entity_migration.py)
LOCATION_LEXICON = {"allowed": [], "banned": []}
NEUTRAL_LOCATION_HEADS = ["device", "structure", "element", "unit", "substance", "object", "container", "area"]
LOCATION_NEUTRALIZATION = {"enabled": _env_bool("LOCATION_NEUTRALIZATION", True)}

# Polishing backend: vlm | text (core/entity_migration.py)
POLISHING_MODE = _env_str("POLISHING_MODE", "vlm")

# Recognition quality thresholds (core/vlm_recognizer.py)
QUALITY_THRESHOLDS = {
    "recognition_confidence": _env_float("RECOGNITION_CONFIDENCE", 0.6),
    "coherence_score": _env_float("COHERENCE_SCORE", 0.7),
}

# Prompt versions (core/entity_extractor.py)
PROMPT_VERSIONS = {
    "image_composition": _env_str("IMAGE_COMPOSITION_PROMPT_VERSION", "v2"),
}

# ---------------------------------------------------------------------------
# Evaluation configuration (process_utils/*)
# ---------------------------------------------------------------------------

API_KEYS = {
    "ollama": OLLAMA_API_KEY,
    "openai": _env_str("OPENAI_API_KEY", ""),
    "anthropic": _env_str("ANTHROPIC_API_KEY", ""),
    "siliconcloud": _env_str("SILICONCLOUD_API_KEY", ""),
    "openrouter": _env_str("OPENROUTER_API_KEY", ""),
}

API_URLS = {
    "ollama": OLLAMA_BASE_URL,
    "openai": _env_str("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "anthropic": _env_str("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
    "siliconcloud": _env_str("SILICONCLOUD_BASE_URL", "https://api.siliconflow.cn/v1"),
    "openrouter": _env_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
}

# Local model list -> registered under the "ollama" type for eval.py --target_model/--judge_model
LOCAL_MODELS = [
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    DEFAULT_ADVERSARIAL_MODEL,
    DEFAULT_JUDGE_MODEL,
    "qwen3-vl:32b",
    "qwen3-vl:30b-a3b-instruct",
    "qwen2.5vl:72b-q8_0",
    "gpt-oss:20B",
]

MODEL_REGISTRY = {
    "ollama": list(dict.fromkeys(LOCAL_MODELS)),
    "openai": [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
    ],
    "anthropic": ["anthropic/claude-haiku-4.5"],
    "siliconcloud": [
        "siliconcloud/Qwen/Qwen3-VL-235B-A22B-Instruct",
        "siliconcloud/MiniMaxAI/MiniMax-M2",
    ],
    "openrouter": [
        # Registered name = "openrouter/<exact OpenRouter model id>". The
        # "openrouter/" prefix selects the provider; it is stripped before the
        # API call (see process_utils/model_initializer.api_model_name), so the
        # part after it must be the EXACT id from https://openrouter.ai/models .
        "openrouter/openai/gpt-4o-mini",          # strong, cheap judge via OpenRouter
        "openrouter/qwen/qwen3-vl-235b-a22b-instruct",  # large vision model (verify the slug on openrouter.ai)
    ],
}

EVAL_MAX_TOKENS = _env_int("EVAL_MAX_TOKENS", 2048)
EVAL_RETRY_TEMPERATURE = _env_float("EVAL_RETRY_TEMPERATURE", 0.7)

# ---------------------------------------------------------------------------
# Translation configuration (transmuter/services/translation.py)
# ---------------------------------------------------------------------------

DEEPL_API_KEY = _env_str("DEEPL_API_KEY", "")
DEEPL_BASE_URL = _env_str("DEEPL_BASE_URL", "https://api.deepl.com/v2/translate")

TRANSLATION_PROVIDER = _env_str("TRANSLATION_PROVIDER", "none")
TRANSLATION_RETRY_LIMIT = _env_int("TRANSLATION_RETRY_LIMIT", 3)
TRANSLATION_TIMEOUT = _env_int("TRANSLATION_TIMEOUT", 30)
TRANSLATION_SPLIT_SENTENCES = _env_str("TRANSLATION_SPLIT_SENTENCES", "1")
TRANSLATION_PRESERVE_FORMATTING = _env_str("TRANSLATION_PRESERVE_FORMATTING", "1")
TRANSLATION_FORMALITY = _env_str("TRANSLATION_FORMALITY", "default")

__all__ = [name for name in dir() if not name.startswith("_")]
