"""
02_logistic_regression_binary.py

主题：二分类逻辑回归
难度：入门到基础

说明：
- 使用 sigmoid 函数输出类别概率
- 使用二元交叉熵损失
- 使用梯度下降训练
"""

import numpy as np


class LogisticRegressionBinary:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            logits = X @ self.w + self.b
            probs = self.sigmoid(logits)

            eps = 1e-12
            loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))

            dw = (1 / n_samples) * (X.T @ (probs - y))
            db = np.mean(probs - y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if epoch % 200 == 0:
                acc = np.mean(self.predict(X) == y)
                print(f"Epoch {epoch:4d} | Loss = {loss:.4f} | Acc = {acc:.4f}")

    def predict_proba(self, X):
        X = np.asarray(X)
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def make_data():
    np.random.seed(0)

    class0 = np.random.normal(loc=[-2, -2], scale=1.0, size=(60, 2))
    class1 = np.random.normal(loc=[2, 2], scale=1.0, size=(60, 2))

    X = np.vstack([class0, class1])
    y = np.array([0] * len(class0) + [1] * len(class1))

    return X, y


def main():
    X, y = make_data()

    model = LogisticRegressionBinary(learning_rate=0.2, epochs=1200)
    model.fit(X, y)

    print("\n最终参数：")
    print("w =", model.w)
    print("b =", model.b)

    test_points = np.array([
        [-3, -2],
        [0, 0],
        [3, 2],
    ])

    print("\n测试样本预测：")
    for point, prob, label in zip(test_points, model.predict_proba(test_points), model.predict(test_points)):
        print(f"x = {point}, P(class=1) = {prob:.4f}, predicted label = {label}")


if __name__ == "__main__":
    main()
