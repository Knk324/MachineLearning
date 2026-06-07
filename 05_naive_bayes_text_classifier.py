"""
05_naive_bayes_text_classifier.py

主题：朴素贝叶斯文本分类
难度：中等

说明：
- 用简单的词袋模型表示文本
- 用多项式朴素贝叶斯进行分类
- 示例任务：判断一句话偏“体育”还是“科技”
"""

import math
from collections import defaultdict, Counter


class MultinomialNaiveBayesText:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior = {}
        self.word_log_prob = {}
        self.vocab = set()
        self.classes = []

    @staticmethod
    def tokenize(text):
        return text.lower().split()

    def fit(self, texts, labels):
        self.classes = sorted(set(labels))

        class_counts = Counter(labels)
        total_docs = len(labels)

        word_counts_by_class = {c: defaultdict(int) for c in self.classes}
        total_words_by_class = {c: 0 for c in self.classes}

        for text, label in zip(texts, labels):
            tokens = self.tokenize(text)
            for token in tokens:
                self.vocab.add(token)
                word_counts_by_class[label][token] += 1
                total_words_by_class[label] += 1

        vocab_size = len(self.vocab)

        for c in self.classes:
            self.class_log_prior[c] = math.log(class_counts[c] / total_docs)
            self.word_log_prob[c] = {}

            for word in self.vocab:
                count = word_counts_by_class[c][word]
                prob = (count + self.alpha) / (total_words_by_class[c] + self.alpha * vocab_size)
                self.word_log_prob[c][word] = math.log(prob)

    def predict_one(self, text):
        tokens = self.tokenize(text)
        scores = {}

        for c in self.classes:
            score = self.class_log_prior[c]

            for token in tokens:
                if token in self.vocab:
                    score += self.word_log_prob[c][token]

            scores[c] = score

        return max(scores, key=scores.get), scores

    def predict(self, texts):
        return [self.predict_one(text)[0] for text in texts]


def main():
    texts = [
        "team win football match",
        "player score goal",
        "basketball team champion",
        "coach and player train",
        "computer chip ai model",
        "software system data",
        "python code algorithm",
        "machine learning model",
    ]

    labels = [
        "sports",
        "sports",
        "sports",
        "sports",
        "technology",
        "technology",
        "technology",
        "technology",
    ]

    model = MultinomialNaiveBayesText(alpha=1.0)
    model.fit(texts, labels)

    test_texts = [
        "football player score",
        "python machine learning",
        "team data model",
    ]

    print("朴素贝叶斯文本分类结果：")
    for text in test_texts:
        label, scores = model.predict_one(text)
        print(f"\nText: {text}")
        print(f"Predicted label: {label}")
        print(f"Log scores: {scores}")


if __name__ == "__main__":
    main()
