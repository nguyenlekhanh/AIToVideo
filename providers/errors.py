"""Provider errors: always carry provider + workflow context, chained cause."""
from __future__ import annotations


class UnknownModelError(ValueError):
    """Requested model is not in the registry."""


class ProviderError(RuntimeError):
    """A provider failed. Never retry silently: raise with full context."""

    def __init__(self, provider: str, message: str, workflow: str = "",
                 detail: str = ""):
        self.provider = provider
        self.message = message
        self.workflow = workflow
        self.detail = detail
        super().__init__(message)

    def __str__(self) -> str:
        parts = [f"[provider:{self.provider}]"]
        if self.workflow:
            parts.append(f"[workflow:{self.workflow}]")
        parts.append(self.message)
        text = " ".join(p for p in parts if p)
        if self.detail:
            text += f"\n{self.detail}"
        return text
