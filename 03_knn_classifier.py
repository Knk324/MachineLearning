"""
03_knn_classifier.py

主题：K近邻分类 KNN
难度：基础

说明：
- KNN 不需要显式训练参数
- 预测时寻找距离最近的 K 个样本
- 适合理解“基于实例的学习”
"""

import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, k=3):
        if k <= 0:
            raise ValueError("k must be positive.")
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def _predict_one(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = self.y_train[nearest_indices]

        most_common = Counter(nearest_labels).most_common(1)[0][0]
        return most_common

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x) for x in X])


def main():
    np.random.seed(1)

    X0 = np.random.normal(loc=[0, 0], scale=0.8, size=(30, 2))
    X1 = np.random.normal(loc=[4, 4], scale=0.8, size=(30, 2))
    X2 = np.random.normal(loc=[0, 5], scale=0.8, size=(30, 2))

    X = np.vstack([X0, X1, X2])
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)

    model = KNNClassifier(k=5)
    model.fit(X, y)

    test_points = np.array([
        [0.2, 0.1],
        [4.2, 3.7],
        [-0.5, 5.2],
        [2.0, 2.0],
    ])

    predictions = model.predict(test_points)

    print("KNN 预测结果：")
    for point, pred in zip(test_points, predictions):
        print(f"x = {point}, predicted class = {pred}")


if __name__ == "__main__":
    main()
