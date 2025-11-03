from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
)


@dataclass(frozen=True)
class LoaderParam:
    """Describes a configuration parameter for a dataset loader."""

    name: str
    label: str
    kind: str = "string"
    description: str | None = None
    required: bool = True
    default: Any | None = None
    choices: Sequence[Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "required": self.required,
        }
        if self.description is not None:
            payload["description"] = self.description
        if self.default is not None:
            payload["default"] = self.default
        if self.choices is not None:
            payload["choices"] = list(self.choices)
        return payload


class DatasetLoader(ABC):
    """Base class for dataset loaders."""

    id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str] = ""
    params: ClassVar[Sequence[LoaderParam]] = ()

    def __init__(self, **config: Any) -> None:
        self.config = config

    @classmethod
    def metadata(cls) -> Dict[str, Any]:
        return {
            "id": cls.id,
            "name": cls.name,
            "description": cls.description,
            "params": [param.to_dict() for param in cls.params],
        }

    @abstractmethod
    def scan(self) -> Iterable[Dict[str, Any]]:
        """Return lightweight dataset descriptors for the current configuration."""

    @abstractmethod
    def iter_records(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Yield preview records for the configured dataset."""


LoaderT = TypeVar("LoaderT", bound=DatasetLoader)


class LoaderRegistry:
    def __init__(self) -> None:
        self._registry: MutableMapping[str, Type[DatasetLoader]] = {}

    def register(self, loader_cls: Type[LoaderT]) -> Type[LoaderT]:
        loader_id = getattr(loader_cls, "id", None)
        if not loader_id:
            raise ValueError("DatasetLoader subclasses must define an 'id' attribute")
        if loader_id in self._registry:
            raise ValueError(f"Dataset loader '{loader_id}' already registered")
        self._registry[loader_id] = loader_cls
        return loader_cls

    def get(self, loader_id: str) -> Type[DatasetLoader]:
        try:
            return self._registry[loader_id]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset loader '{loader_id}'") from exc

    def all(self) -> List[Type[DatasetLoader]]:
        return list(self._registry.values())


registry = LoaderRegistry()


def register_loader(loader_cls: Type[LoaderT]) -> Type[LoaderT]:
    return registry.register(loader_cls)
