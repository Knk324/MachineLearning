"""
10_adaboost_decision_stump.py

主题：AdaBoost + 决策树桩
难度：偏难

说明：
- AdaBoost 是经典集成学习算法
- 通过不断提高错分样本的权重，让多个弱分类器组合成强分类器
- 本文件使用一维阈值决策树桩作为弱分类器
"""

import numpy as np


class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)

        feature_values = X[:, self.feature_index]

        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values > self.threshold] = -1

        return predictions


class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        y = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape
        sample_weights = np.full(n_samples, 1 / n_samples)

        for estimator_idx in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float("inf")

            for feature_index in range(n_features):
                thresholds = np.unique(X[:, feature_index])

                for threshold in thresholds:
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)

                        if polarity == 1:
                            predictions[X[:, feature_index] < threshold] = -1
                        else:
                            predictions[X[:, feature_index] > threshold] = -1

                        misclassified = predictions != y
                        weighted_error = np.sum(sample_weights[misclassified])

                        if weighted_error < min_error:
                            min_error = weighted_error
                            stump.feature_index = feature_index
                            stump.threshold = threshold
                            stump.polarity = polarity

            eps = 1e-12
            alpha = 0.5 * np.log((1 - min_error + eps) / (min_error + eps))

            predictions = stump.predict(X)

            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)

            self.stumps.append(stump)
            self.alphas.append(alpha)

            train_acc = np.mean(self.predict(X) == y)
            print(
                f"Estimator {estimator_idx + 1:2d} | "
                f"error = {min_error:.4f} | alpha = {alpha:.4f} | acc = {train_acc:.4f}"
            )

    def predict(self, X):
        X = np.asarray(X)

        final_scores = np.zeros(X.shape[0])
        for alpha, stump in zip(self.alphas, self.stumps):
            final_scores += alpha * stump.predict(X)

        return np.sign(final_scores)


def main():
    np.random.seed(21)

    class_neg = np.random.normal(loc=[-1.5, -1.5], scale=1.0, size=(50, 2))
    class_pos = np.random.normal(loc=[1.5, 1.5], scale=1.0, size=(50, 2))

    X = np.vstack([class_neg, class_pos])
    y = np.array([-1] * 50 + [1] * 50)

    model = AdaBoost(n_estimators=12)
    model.fit(X, y)

    test_points = np.array([
        [-2, -1],
        [0, 0],
        [2, 2],
    ])

    print("\n测试样本预测：")
    for point, pred in zip(test_points, model.predict(test_points)):
        print(f"x = {point}, predicted label = {int(pred)}")


if __name__ == "__main__":
    main()
