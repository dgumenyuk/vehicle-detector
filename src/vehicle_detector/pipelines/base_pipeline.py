from abc import ABC, abstractmethod


class BasePipeline(ABC):
    @abstractmethod
    def initialize(self, config: dict) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass
