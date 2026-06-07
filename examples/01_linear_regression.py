from dataclasses import dataclass


@dataclass
class LinearRegressionGD:
    learning_rate: float = 0.01
    epochs: int = 2000
    weight: float = 0.0
    bias: float = 0.0

    def fit(self, x: list[float], y: list[float]) -> None:
        if len(x) != len(y) or not x:
            raise ValueError("x and y must be non-empty and have the same length")

        n = len(x)
        for _ in range(self.epochs):
            dw = sum(((self.weight * xi + self.bias) - yi) * xi for xi, yi in zip(x, y)) * (2 / n)
            db = sum((self.weight * xi + self.bias) - yi for xi, yi in zip(x, y)) * (2 / n)
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, x: list[float]) -> list[float]:
        return [self.weight * xi + self.bias for xi in x]


if __name__ == "__main__":
    train_x = [1, 2, 3, 4, 5]
    train_y = [3, 5, 7, 9, 11]

    model = LinearRegressionGD()
    model.fit(train_x, train_y)
    preds = model.predict([6, 7])
    print("weight=", round(model.weight, 3), "bias=", round(model.bias, 3))
    print("predictions for [6, 7] =", [round(v, 3) for v in preds])
