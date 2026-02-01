from abc import ABC, abstractmethod

class Postprocessor(ABC):
    @abstractmethod
    def train(self, num_epochs: int):
        """Entrena el modelo."""
        pass

    @abstractmethod
    def validate(self):
        """Valida el modelo."""
        pass

    @abstractmethod
    def save_model(self, path: str):
        """Guarda el modelo entrenado."""
        pass

    @abstractmethod
    def load_model(self, path: str):
        """Carga un modelo previamente entrenado."""
        pass