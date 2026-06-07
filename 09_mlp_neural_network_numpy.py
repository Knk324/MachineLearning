"""
09_mlp_neural_network_numpy.py

主题：两层神经网络 MLP
难度：偏难

说明：
- 用 numpy 实现一个简单的多层感知机
- 任务：学习 XOR 异或问题
- 重点：前向传播、反向传播、链式法则
"""

import numpy as np


class MLP:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.1, epochs=5000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0, 1, size=(hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.sigmoid(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)

        cache = {
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
        }
        return a2, cache

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        n = len(X)

        for epoch in range(self.epochs):
            y_pred, cache = self.forward(X)

            eps = 1e-12
            loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

            # 输出层梯度：sigmoid + BCE 的组合梯度可以简化为 y_pred - y
            dz2 = y_pred - y
            dW2 = cache["a1"].T @ dz2 / n
            db2 = np.mean(dz2, axis=0, keepdims=True)

            da1 = dz2 @ self.W2.T
            dz1 = da1 * self.sigmoid_derivative(cache["a1"])
            dW1 = X.T @ dz1 / n
            db1 = np.mean(dz1, axis=0, keepdims=True)

            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1

            if epoch % 1000 == 0:
                acc = np.mean(self.predict(X) == y.reshape(-1))
                print(f"Epoch {epoch:4d} | Loss = {loss:.4f} | Acc = {acc:.4f}")

    def predict_proba(self, X):
        y_pred, _ = self.forward(np.asarray(X))
        return y_pred.reshape(-1)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def main():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])

    y = np.array([0, 1, 1, 0])

    model = MLP(input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.8, epochs=6000)
    model.fit(X, y)

    print("\nXOR 预测结果：")
    for sample, prob, pred in zip(X, model.predict_proba(X), model.predict(X)):
        print(f"x = {sample}, prob = {prob:.4f}, predicted = {pred}")


if __name__ == "__main__":
    main()
