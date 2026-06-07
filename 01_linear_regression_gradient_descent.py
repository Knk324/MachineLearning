"""
01_linear_regression_gradient_descent.py

主题：线性回归 + 梯度下降
难度：入门

说明：
- 用 y = wx + b 拟合一组带噪声的一维数据
- 不依赖 sklearn，主要用于理解损失函数和梯度下降
"""

import numpy as np


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0
        self.loss_history = []

    def fit(self, X, y):
        X = X.reshape(-1)
        y = y.reshape(-1)
        n = len(X)

        for epoch in range(self.epochs):
            y_pred = self.w * X + self.b
            error = y_pred - y

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            dw = (2 / n) * np.sum(error * X)
            db = (2 / n) * np.sum(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if epoch % 200 == 0:
                print(f"Epoch {epoch:4d} | Loss = {loss:.4f}")

    def predict(self, X):
        X = np.asarray(X).reshape(-1)
        return self.w * X + self.b


def main():
    np.random.seed(42)

    X = np.linspace(0, 10, 80)
    true_w, true_b = 3.0, 2.0
    noise = np.random.normal(0, 2, size=X.shape)
    y = true_w * X + true_b + noise

    model = LinearRegressionGD(learning_rate=0.01, epochs=1200)
    model.fit(X, y)

    print("\n训练完成")
    print(f"学习到的 w = {model.w:.4f}, b = {model.b:.4f}")
    print(f"真实参数   w = {true_w:.4f}, b = {true_b:.4f}")

    test_X = np.array([2, 5, 8])
    print("\n预测结果：")
    for x, pred in zip(test_X, model.predict(test_X)):
        print(f"x = {x}, predicted y = {pred:.4f}")


if __name__ == "__main__":
    main()
