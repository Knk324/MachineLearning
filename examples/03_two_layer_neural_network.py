import math
import random
from dataclasses import dataclass


def sigmoid(x: float) -> float:
    if x >= 0:
        e = math.exp(-x)
        return 1 / (1 + e)
    e = math.exp(x)
    return e / (1 + e)


@dataclass
class TwoLayerNN:
    hidden_size: int = 4
    learning_rate: float = 0.5
    epochs: int = 6000
    seed: int = 42

    def __post_init__(self) -> None:
        random.seed(self.seed)
        self.w1 = [[random.uniform(-1, 1) for _ in range(self.hidden_size)] for _ in range(2)]
        self.b1 = [0.0 for _ in range(self.hidden_size)]
        self.w2 = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
        self.b2 = 0.0

    def _forward(self, x: list[float]) -> tuple[list[float], float]:
        hidden_raw = [
            x[0] * self.w1[0][j] + x[1] * self.w1[1][j] + self.b1[j]
            for j in range(self.hidden_size)
        ]
        hidden = [sigmoid(v) for v in hidden_raw]
        out_raw = sum(hidden[j] * self.w2[j] for j in range(self.hidden_size)) + self.b2
        out = sigmoid(out_raw)
        return hidden, out

    def fit(self, x_data: list[list[float]], y_data: list[int]) -> None:
        if len(x_data) != len(y_data) or not x_data:
            raise ValueError("x_data and y_data must be non-empty and have the same length")

        for _ in range(self.epochs):
            for x, y in zip(x_data, y_data):
                hidden, out = self._forward(x)

                d_out = (out - y) * out * (1 - out)

                d_w2 = [d_out * h for h in hidden]
                d_b2 = d_out

                d_hidden = [
                    d_out * self.w2[j] * hidden[j] * (1 - hidden[j])
                    for j in range(self.hidden_size)
                ]

                d_w1_0 = [d_hidden[j] * x[0] for j in range(self.hidden_size)]
                d_w1_1 = [d_hidden[j] * x[1] for j in range(self.hidden_size)]
                d_b1 = d_hidden

                for j in range(self.hidden_size):
                    self.w2[j] -= self.learning_rate * d_w2[j]
                    self.w1[0][j] -= self.learning_rate * d_w1_0[j]
                    self.w1[1][j] -= self.learning_rate * d_w1_1[j]
                    self.b1[j] -= self.learning_rate * d_b1[j]
                self.b2 -= self.learning_rate * d_b2

    def predict_proba(self, x_data: list[list[float]]) -> list[float]:
        return [self._forward(x)[1] for x in x_data]

    def predict(self, x_data: list[list[float]], threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(x_data)]


if __name__ == "__main__":
    # XOR dataset
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [0, 1, 1, 0]

    model = TwoLayerNN()
    model.fit(x_train, y_train)

    probs = model.predict_proba(x_train)
    preds = model.predict(x_train)

    print("probabilities =", [round(p, 3) for p in probs])
    print("predictions   =", preds)
