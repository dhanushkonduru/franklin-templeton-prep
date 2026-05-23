from __future__ import annotations

from abc import ABC, abstractmethod

from app.types import RawNewsEvent


class BaseSource(ABC):
    name: str

    @abstractmethod
    async def fetch_events(self) -> list[RawNewsEvent]:
        raise NotImplementedError
