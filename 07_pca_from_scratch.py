"""
07_pca_from_scratch.py

主题：主成分分析 PCA
难度：中等

说明：
- PCA 是常见的无监督降维算法
- 核心思想：寻找数据方差最大的方向
- 本文件用 numpy 从零实现 PCA
"""

import numpy as np


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance_ratio = None

    def fit(self, X):
        X = np.asarray(X)
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        covariance_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components = eigenvectors[:, :self.n_components]

        total_variance = eigenvalues.sum()
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_variance

    def transform(self, X):
        X = np.asarray(X)
        X_centered = X - self.mean
        return X_centered @ self.components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def main():
    np.random.seed(7)

    n = 120
    x1 = np.random.normal(0, 3, size=n)
    x2 = 0.7 * x1 + np.random.normal(0, 0.8, size=n)
    x3 = -0.2 * x1 + np.random.normal(0, 0.5, size=n)

    X = np.column_stack([x1, x2, x3])

    model = PCA(n_components=2)
    X_reduced = model.fit_transform(X)

    print("原始数据形状：", X.shape)
    print("降维后形状：", X_reduced.shape)

    print("\n主成分方向：")
    print(model.components)

    print("\n解释方差比例：")
    print(model.explained_variance_ratio)

    print("\n前 5 个降维后的样本：")
    print(X_reduced[:5])


if __name__ == "__main__":
    main()
