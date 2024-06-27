from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    @abstractmethod
    def eval(self):
        pass
