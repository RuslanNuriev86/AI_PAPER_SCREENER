"""Settings: environment + YAML (§3, §11).

One field per YAML file, so every key has exactly one home. `config_hash` covers all of them
and is stored on every run, which is what makes a past digest reproducible (§13.5).
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from screener.domain.models import (
    OutcomeScale,
    Profile,
    RevisitConfig,
    Selection,
)
from screener.domain.types import RUBRIC_WEIGHTS, TRIAGE_WEIGHTS

CONFIG_DIR = Path("config")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # Validation aliases above are the env spellings; this lets code and YAML construct
        # Settings with the plain field name too.
        populate_by_name=True,
    )

    # --- secrets (env only) ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- LLM provider (DeepSeek by default) ---
    # Field names are deliberately provider-neutral: the adapter speaks OpenAI-shaped
    # chat-completions, which DeepSeek serves at https://api.deepseek.com. Accepting several
    # env spellings means an existing OPENAI_* key still works without a code change.
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"),
    )
    llm_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias=AliasChoices("DEEPSEEK_BASE_URL", "LLM_BASE_URL", "OPENAI_BASE_URL"),
    )
    #: Thinking mode is ON by default on DeepSeek and **ignores `temperature`**. Scoring must
    #: be repeatable (the rubric is meant to be auditable), so it is off by default: a
    #: reasoning trace would add latency and cost while silently discarding temperature=0.
    llm_thinking: bool = Field(
        default=False, validation_alias=AliasChoices("SCREENER_LLM_THINKING", "LLM_THINKING")
    )
    semantic_scholar_api_key: str = ""
    github_token: str = ""
    contact_email: str = "anonymous@example.com"
    heartbeat_url: str = ""

    # --- runtime ---
    screener_db: str = "./screener.db"
    #: Where an undeliverable digest is parked. Configurable so tests cannot touch
    #: (or consume) a real outbox in the working directory.
    screener_outbox: str = "./outbox"
    screener_mode: str = "live"  # live | dry
    screener_llm_fast: str = "deepseek-flash"
    screener_llm_deep: str = "deepseek-flash"
    screener_budget_usd: float = 2.0
    #: How many gate-passers reach the review tier. At v0 the deterministic gate
    #: hint orders them (a stand-in for the v1 triage cascade); it is also the
    #: dominant cost lever, which is why it is bounded rather than "all".
    screener_review_top_k: int = 16
    #: §4 stage-5 failure behaviour: at or above this many per-paper review failures the run
    #: is recorded `degraded`, not `ok`. Without it a run where every single review failed
    #: still reports success — which is the silent-degradation mode §1.1 exists to prevent.
    screener_degrade_after_review_failures: int = 3
    screener_config_dir: str = "config"

    # --- loaded from YAML (not env) ---
    profile: Profile = Field(default_factory=Profile)
    selection: Selection = Field(default_factory=Selection)
    revisit: RevisitConfig = Field(default_factory=RevisitConfig)
    outcome_scale: OutcomeScale = Field(default_factory=OutcomeScale)

    @property
    def review_top_k(self) -> int:
        return self.screener_review_top_k

    @property
    def dry_run(self) -> bool:
        return self.screener_mode.lower() in {"dry", "dry-run"}

    @property
    def llm_fast(self) -> str:
        return self.screener_llm_fast

    @property
    def llm_deep(self) -> str:
        return self.screener_llm_deep

    def config_hash(self) -> str:
        """Stable digest of everything that can change what a run produces."""
        blob = {
            "profile": self.profile.model_dump(mode="json"),
            "selection": self.selection.model_dump(mode="json"),
            "revisit": self.revisit.model_dump(mode="json"),
            "outcome_scale": self.outcome_scale.model_dump(mode="json"),
            "rubric_weights": RUBRIC_WEIGHTS,
            "triage_weights": TRIAGE_WEIGHTS,
            "llm_fast": self.llm_fast,
            "llm_deep": self.llm_deep,
            "llm_base_url": self.llm_base_url,
            "llm_thinking": self.llm_thinking,
            "budget_usd": self.screener_budget_usd,
        }
        raw = json.dumps(blob, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()[:16]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_settings(config_dir: str | Path = CONFIG_DIR) -> Settings:
    """Load env, then overlay the YAML profiles.

    A missing config file is not fatal: the defaults in `models.py` are the documented
    starting point (DESIGN.md §16 lists them as the confirmed choices), so v0 runs out of the
    box and `screener doctor` reports which files were absent.
    """
    cfg = Settings()
    d = Path(config_dir)

    profile_data = _read_yaml(d / "profile.yaml")
    selection_data = _read_yaml(d / "selection.yaml")
    revisit_data = _read_yaml(d / "revisit.yaml")
    scale_data = _read_yaml(d / "outcome_scale.yaml")

    if profile_data:
        cfg.profile = Profile.model_validate(profile_data)
    if selection_data:
        cfg.selection = Selection.model_validate(selection_data)
    if revisit_data:
        cfg.revisit = RevisitConfig.model_validate(revisit_data)
    if scale_data:
        # v0 reads the t14 block; the full ladder lives in the same file from v1.5.
        t14 = scale_data.get("t14", scale_data)
        if t14:
            cfg.outcome_scale = OutcomeScale.model_validate(
                {k: v for k, v in t14.items() if k in {"stars", "hf_upvotes", "weights"}}
            )
    return cfg


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def prompt_path(name: str, *, base: Path | None = None) -> Path:
    root = base or Path(__file__).parent / "prompts"
    return root / name


def read_prompt(name: str) -> str:
    """Read a versioned prompt file. Prompts are files, never string literals (§3)."""
    return prompt_path(name).read_text()
