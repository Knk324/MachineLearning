# Machine Learning Algorithms Learning Code

这是一组适合学习机器学习基础算法的 Python 独立脚本。

## 使用方式

进入该目录后，直接运行：

```bash
python 01_linear_regression_gradient_descent.py
```

如果脚本使用了 `numpy`，请先安装：

```bash
pip install numpy
```

## 文件列表

| 文件名 | 算法 | 难度 |
|---|---|---|
| `01_linear_regression_gradient_descent.py` | 线性回归 + 梯度下降 | 入门 |
| `02_logistic_regression_binary.py` | 逻辑回归二分类 | 入门到基础 |
| `03_knn_classifier.py` | K近邻分类 KNN | 基础 |
| `04_kmeans_clustering.py` | K-Means 聚类 | 基础到中等 |
| `05_naive_bayes_text_classifier.py` | 朴素贝叶斯文本分类 | 中等 |
| `06_decision_tree_id3.py` | 决策树 ID3 | 中等 |
| `07_pca_from_scratch.py` | PCA 主成分分析 | 中等 |
| `08_linear_svm_hinge_loss.py` | 线性 SVM + Hinge Loss | 中等到偏难 |
| `09_mlp_neural_network_numpy.py` | 两层神经网络 MLP | 偏难 |
| `10_adaboost_decision_stump.py` | AdaBoost 集成学习 | 偏难 |

## 学习建议

建议按照编号顺序阅读和运行：

1. 先理解有监督学习中的回归和分类；
2. 再理解无监督学习中的聚类和降维；
3. 然后学习概率模型、树模型、最大间隔分类；
4. 最后阅读神经网络和集成学习。
