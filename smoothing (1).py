from nltk import ngrams, FreqDist
from collections import defaultdict


class Smoothing:
    def __init__(self, n, laplace, tokens):
        self.n = n
        self.laplace = laplace
        self.tokens = tokens
        self.vocab = FreqDist(self.tokens)
        self.vocab_size = len(self.vocab)

        self.n_grams = ngrams(self.tokens, self.n)
        self.n_vocab = FreqDist(self.n_grams)

        self.m_grams = ngrams(self.tokens, self.n - 1)
        self.m_vocab = FreqDist(self.m_grams)

        self.n_vocab_size = len(self.n_vocab)
        self.m_vocab_size = len(self.m_vocab)

    def smoothed(self, n_gram, n_count):
        pass

    def get_model(self):
        pass

    def get_train_prob_smoothed(self):
        return {n_gram: self.smoothed(n_gram, count) for n_gram, count in self.n_vocab.items()}

    def get_test_prob_smoothed(self, test_ngrams):
        pass


class LaplacianSmoothing(Smoothing):
    def __init__(self, n, laplace, tokens):
        super(LaplacianSmoothing, self).__init__(n, tokens)
        self.laplace = laplace
        self.train_probs = self.get_train_prob_smoothed()

    def smoothed(self, n_gram, n_count):
        m_gram = n_gram[:-1]
        m_count = self.m_vocab[m_gram] if m_gram in self.m_vocab.keys() else 0
        return (n_count + self.laplace) / (m_count + self.laplace * self.vocab_size)

    def get_model(self):
        return self.train_probs

    def get_test_prob_smoothed(self, test_ngrams):
        probs = []
        for n_gram in test_ngrams:
            if n_gram not in self.n_vocab.keys():
                probs.append(self.smoothed(n_gram, 0))
            else:
                probs.append(self.train_probs[n_gram])
        return probs


class GoodTuringSmoothing(Smoothing):
    def __init__(self, n, laplace, tokens):
        super(GoodTuringSmoothing, self).__init__(n, laplace,tokens)

        self.N_c = defaultdict(int)
        for _, count in self.n_vocab.items():
            self.N_c[count] += 1

        self.train_probs = self.get_train_prob_smoothed()

    def smoothed(self, n_gram, n_count):
        m_gram = n_gram[:-1]
        if self.N_c[n_count + 1] == 0 or self.N_c[n_count] == 0:
            n_count = n_count / self.n_vocab_size
        else:
            n_count = (n_count + 1) * self.N_c[n_count + 1] / (self.n_vocab_size * self.N_c[n_count])
        if m_gram in self.m_vocab.keys():
            m_count = self.m_vocab[m_gram] / self.m_vocab_size
            return n_count / m_count
        else:
            return (n_count + self.laplace) / self.laplace * self.vocab_size

    def get_model(self):
        return self.train_probs

    def get_test_prob_smoothed(self, test_ngrams):
        probs = []
        N_c_i = 0
        for i in range(1, self.n_vocab_size + 1):
            if self.N_c[i] > 0:
                N_c_i = self.N_c[i]
                break  # 找第一个不为 0 的 N_c，避免 N_c[1] 为 0 的情况

        count_not_in_vocab = 0
        test_ngrams = list(test_ngrams)  # 避免迭代器被消耗，无法遍历两次
        for n_gram in test_ngrams:
            if n_gram not in self.n_vocab.keys():
                count_not_in_vocab += 1
        
        for n_gram in test_ngrams:
            if n_gram not in self.n_vocab.keys():
                probs.append(N_c_i / (self.n_vocab_size * count_not_in_vocab))
            else:
                probs.append(self.train_probs[n_gram])

        return probs
