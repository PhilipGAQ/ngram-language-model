from nltk import ngrams, FreqDist
from collections import defaultdict

class Smoothing:
    def __init__(self, n, tokens):
        self.n = n
        self.tokens = tokens
        self.vocab = FreqDist(self.tokens)
        self.vocab_size = len(self.vocab)

        self.n_grams = ngrams(self.tokens, self.n)
        self.n_vocab = FreqDist(self.n_grams)

    def get_train_prob_smoothed(self):
        pass

    def get_test_prob_smoothed(self, test_ngrams):
        pass

class LaplacianSmoothing(Smoothing):
    def __init__(self, n, laplace, tokens):
        super().__init__(n, tokens)
        self.laplace = laplace

        self.m_grams = ngrams(self.tokens, self.n - 1)
        self.m_vocab = FreqDist(self.m_grams)

        self.train_probs = self.get_train_prob_smoothed()

    def smoothed(self, n_gram, n_count):
        m_gram = n_gram[:-1]
        m_count = self.m_vocab[m_gram] if m_gram in self.m_vocab else 0
        return (n_count + self.laplace) / (m_count + self.laplace * self.vocab_size)

    def get_train_prob_smoothed(self):
        return {n_gram: self.smoothed(n_gram, count) for n_gram, count in self.n_vocab.items()}

    def get_test_prob_smoothed(self, test_ngrams):
        probs = []
        for n_gram in test_ngrams:
            if n_gram not in self.n_vocab:
                m_gram = n_gram[:-1]
                m_count = self.m_vocab[m_gram] if m_gram in self.m_vocab else 0
                probs.append(self.laplace / (m_count + self.laplace * self.vocab_size))
            else:
                probs.append(self.train_probs[n_gram])
        return probs

class GoodTuringSmoothing(Smoothing):
    def __init__(self, n, tokens):
        super().__init__(n, tokens)
        self.n_vocab_size = len(self.n_vocab)

        self.N_c = defaultdict(int)
        for _, count in self.n_vocab.items():
            self.N_c[count] += 1

        self.train_probs = self.get_train_prob_smoothed()

    def smoothed(self, n_gram, n_count):
        if self.N_c[n_count + 1] == 0 or self.N_c[n_count] == 0:
            return n_count

        return (n_count + 1) * self.N_c[n_count + 1] / (self.n_vocab_size * self.N_c[n_count])

    def get_train_prob_smoothed(self):
        return {n_gram: self.smoothed(n_gram, count) for n_gram, count in self.n_vocab.items()}

    def get_test_prob_smoothed(self, test_ngrams):
        probs = []
        N_c_i = next((i for i in range(1, self.n_vocab_size + 1) if self.N_c[i] > 0), 0)

        for n_gram in test_ngrams:
            if n_gram not in self.n_vocab:
                probs.append(N_c_i / self.n_vocab_size)
            else:
                probs.append(self.train_probs[n_gram])
        return probs
