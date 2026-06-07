"""
06_decision_tree_id3.py

主题：决策树 ID3
难度：中等

说明：
- 使用信息熵和信息增益选择划分特征
- 适用于离散特征
- 示例任务：根据天气条件判断是否打球
"""

import math
from collections import Counter, defaultdict


def entropy(labels):
    counts = Counter(labels)
    total = len(labels)

    result = 0.0
    for count in counts.values():
        p = count / total
        result -= p * math.log2(p)

    return result


def information_gain(rows, labels, feature_index):
    base_entropy = entropy(labels)
    total = len(rows)

    subsets = defaultdict(list)
    subset_labels = defaultdict(list)

    for row, label in zip(rows, labels):
        value = row[feature_index]
        subsets[value].append(row)
        subset_labels[value].append(label)

    weighted_entropy = 0.0
    for value in subsets:
        weight = len(subsets[value]) / total
        weighted_entropy += weight * entropy(subset_labels[value])

    return base_entropy - weighted_entropy


class DecisionTreeID3:
    def __init__(self):
        self.tree = None
        self.feature_names = None

    def fit(self, rows, labels, feature_names):
        self.feature_names = feature_names
        feature_indices = list(range(len(feature_names)))
        self.tree = self._build_tree(rows, labels, feature_indices)

    def _build_tree(self, rows, labels, feature_indices):
        if len(set(labels)) == 1:
            return labels[0]

        if not feature_indices:
            return Counter(labels).most_common(1)[0][0]

        gains = [
            (information_gain(rows, labels, idx), idx)
            for idx in feature_indices
        ]

        best_gain, best_feature = max(gains)

        if best_gain == 0:
            return Counter(labels).most_common(1)[0][0]

        tree = {
            "feature": best_feature,
            "feature_name": self.feature_names[best_feature],
            "branches": {},
            "default": Counter(labels).most_common(1)[0][0],
        }

        values = sorted(set(row[best_feature] for row in rows))

        for value in values:
            sub_rows = []
            sub_labels = []

            for row, label in zip(rows, labels):
                if row[best_feature] == value:
                    sub_rows.append(row)
                    sub_labels.append(label)

            remaining_features = [idx for idx in feature_indices if idx != best_feature]
            tree["branches"][value] = self._build_tree(sub_rows, sub_labels, remaining_features)

        return tree

    def predict_one(self, row):
        node = self.tree

        while isinstance(node, dict):
            feature_index = node["feature"]
            value = row[feature_index]

            if value in node["branches"]:
                node = node["branches"][value]
            else:
                return node["default"]

        return node

    def predict(self, rows):
        return [self.predict_one(row) for row in rows]

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.tree

        if not isinstance(node, dict):
            print(indent + "=> " + str(node))
            return

        print(indent + f"[Feature: {node['feature_name']}]")
        for value, child in node["branches"].items():
            print(indent + f"  If {node['feature_name']} == {value}:")
            self.print_tree(child, indent + "    ")


def main():
    feature_names = ["Outlook", "Temperature", "Humidity", "Wind"]

    rows = [
        ["sunny", "hot", "high", "weak"],
        ["sunny", "hot", "high", "strong"],
        ["overcast", "hot", "high", "weak"],
        ["rain", "mild", "high", "weak"],
        ["rain", "cool", "normal", "weak"],
        ["rain", "cool", "normal", "strong"],
        ["overcast", "cool", "normal", "strong"],
        ["sunny", "mild", "high", "weak"],
        ["sunny", "cool", "normal", "weak"],
        ["rain", "mild", "normal", "weak"],
        ["sunny", "mild", "normal", "strong"],
        ["overcast", "mild", "high", "strong"],
        ["overcast", "hot", "normal", "weak"],
        ["rain", "mild", "high", "strong"],
    ]

    labels = [
        "no", "no", "yes", "yes", "yes", "no", "yes",
        "no", "yes", "yes", "yes", "yes", "yes", "no"
    ]

    model = DecisionTreeID3()
    model.fit(rows, labels, feature_names)

    print("学习到的决策树：")
    model.print_tree()

    test_rows = [
        ["sunny", "cool", "high", "strong"],
        ["overcast", "mild", "high", "weak"],
        ["rain", "mild", "normal", "strong"],
    ]

    print("\n预测结果：")
    for row, pred in zip(test_rows, model.predict(test_rows)):
        print(f"{row} => play = {pred}")


if __name__ == "__main__":
    main()
