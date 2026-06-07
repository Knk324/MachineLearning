import math
from dataclasses import dataclass


def sigmoid(z: float) -> float:
    """Numerically stable sigmoid."""
    if z >= 0:
        ez = math.exp(-z)
        return 1 / (1 + ez)
    ez = math.exp(z)
    return ez / (1 + ez)


@dataclass
class LogisticRegressionGD:
    learning_rate: float = 0.1
    epochs: int = 3000
    weight: float = 0.0
    bias: float = 0.0

    def fit(self, x: list[float], y: list[int]) -> None:
        if len(x) != len(y) or not x:
            raise ValueError("x and y must be non-empty and have the same length")
        if any(label not in (0, 1) for label in y):
            raise ValueError("y must only contain 0/1 labels")

        n = len(x)
        for _ in range(self.epochs):
            preds = [sigmoid(self.weight * xi + self.bias) for xi in x]
            dw = sum((p - yi) * xi for p, xi, yi in zip(preds, x, y)) / n
            db = sum(p - yi for p, yi in zip(preds, y)) / n
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, x: list[float]) -> list[float]:
        return [sigmoid(self.weight * xi + self.bias) for xi in x]

    def predict(self, x: list[float], threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(x)]


if __name__ == "__main__":
    # Simple dataset: points > 2.5 should be class 1
    train_x = [0.5, 1.0, 1.5, 3.0, 3.5, 4.0]
    train_y = [0, 0, 0, 1, 1, 1]

    model = LogisticRegressionGD()
    model.fit(train_x, train_y)
    test_x = [2.0, 2.8, 4.5]
    print("probabilities =", [round(p, 3) for p in model.predict_proba(test_x)])
    print("predictions   =", model.predict(test_x))
