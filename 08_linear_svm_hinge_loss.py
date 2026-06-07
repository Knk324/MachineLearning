"""
08_linear_svm_hinge_loss.py

主题：线性支持向量机 SVM + Hinge Loss
难度：中等到偏难

说明：
- 用梯度下降优化线性 SVM
- 标签需要转换为 -1 和 +1
- 目标函数包含 hinge loss 和 L2 正则化
"""

import numpy as np


class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        y_transformed = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y_transformed[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - y_transformed[idx] * x_i
                    db = -y_transformed[idx]

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

            if epoch % 200 == 0:
                acc = np.mean(self.predict(X) == y)
                print(f"Epoch {epoch:4d} | Acc = {acc:.4f}")

    def predict(self, X):
        X = np.asarray(X)
        linear_output = X @ self.w + self.b
        return np.where(linear_output >= 0, 1, 0)


def main():
    np.random.seed(11)

    class0 = np.random.normal(loc=[-2, -2], scale=0.8, size=(50, 2))
    class1 = np.random.normal(loc=[2, 2], scale=0.8, size=(50, 2))

    X = np.vstack([class0, class1])
    y = np.array([0] * 50 + [1] * 50)

    model = LinearSVM(learning_rate=0.001, lambda_param=0.01, epochs=1000)
    model.fit(X, y)

    print("\n最终参数：")
    print("w =", model.w)
    print("b =", model.b)

    test_points = np.array([
        [-3, -1],
        [0, 0],
        [3, 2],
    ])

    print("\n测试样本预测：")
    for point, label in zip(test_points, model.predict(test_points)):
        print(f"x = {point}, predicted label = {label}")


if __name__ == "__main__":
    main()
