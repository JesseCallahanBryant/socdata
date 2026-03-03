"""registry.py — Provider registry with @register decorator."""

from __future__ import annotations

from socdata.datasets.base import DatasetProvider

_REGISTRY: dict[str, DatasetProvider] = {}


def register(provider: DatasetProvider) -> DatasetProvider:
    """Register a dataset provider instance."""
    _REGISTRY[provider.name.lower()] = provider
    return provider


def get_provider(name: str) -> DatasetProvider | None:
    """Look up a provider by name (case-insensitive)."""
    return _REGISTRY.get(name.lower())


def list_providers() -> list[DatasetProvider]:
    """Return all registered providers."""
    return list(_REGISTRY.values())
