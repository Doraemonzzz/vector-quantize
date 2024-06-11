from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @abstractmethod
    def resume(
        self,
    ):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass
