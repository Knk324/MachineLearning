"""
04_kmeans_clustering.py

主题：K-Means 聚类
难度：基础到中等

说明：
- 无监督学习算法
- 通过“分配样本到最近中心”和“更新中心”反复迭代
"""

import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X)
        rng = np.random.default_rng(self.random_state)

        initial_indices = rng.choice(len(X), size=self.n_clusters, replace=False)
        self.centroids = X[initial_indices]

        for iteration in range(self.max_iter):
            distances = self._pairwise_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.labels_ = labels

            print(f"Iter {iteration:3d} | centroid shift = {shift:.6f}")

            if shift < 1e-6:
                break

    @staticmethod
    def _pairwise_distances(X, centroids):
        return np.sqrt(((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))

    def predict(self, X):
        X = np.asarray(X)
        distances = self._pairwise_distances(X, self.centroids)
        return np.argmin(distances, axis=1)


def main():
    np.random.seed(10)

    cluster1 = np.random.normal(loc=[0, 0], scale=0.6, size=(50, 2))
    cluster2 = np.random.normal(loc=[5, 5], scale=0.6, size=(50, 2))
    cluster3 = np.random.normal(loc=[0, 6], scale=0.6, size=(50, 2))

    X = np.vstack([cluster1, cluster2, cluster3])

    model = KMeans(n_clusters=3, max_iter=50, random_state=0)
    model.fit(X)

    print("\n最终聚类中心：")
    print(model.centroids)

    test_points = np.array([
        [0, 0],
        [5, 5],
        [0, 6],
    ])

    print("\n测试点所属簇：")
    for point, label in zip(test_points, model.predict(test_points)):
        print(f"x = {point}, cluster = {label}")


if __name__ == "__main__":
    main()
