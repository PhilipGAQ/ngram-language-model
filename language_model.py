#!/bin/env python

import argparse
from itertools import product
import math
import nltk
from pathlib import Path
import os
from collections import Counter
from preprocess import preprocess
import matplotlib.pyplot as plt
from evalue import top_k_ngrams
from smoothing import LaplacianSmoothing,GoodTuringSmoothing

def load_data(data_dir):
    """Load train and test corpora from a directory.

    Directory must contain two files: train.txt and test.txt.
    Newlines will be stripped out. 

    Args:
        data_dir (Path) -- pathlib.Path of the directory to use. 

    Returns:
        The train and test sets, as lists of sentences.

    """
    train_path = data_dir.joinpath('train.txt').absolute().as_posix()
    test1_path  = data_dir.joinpath('test.1.txt').absolute().as_posix()
    test2_path  = data_dir.joinpath('test.2.txt').absolute().as_posix()

    with open(train_path, 'r',encoding='utf-8') as f:
        train = [l.strip() for l in f.readlines()]
    with open(test1_path, 'r',encoding='utf-8') as f:
        test1 = [l.strip() for l in f.readlines()]
    with open(test2_path, 'r',encoding='utf-8') as f:
        test2 = [l.strip() for l in f.readlines()]
    return train, test1,test2


class LanguageModel(object):
    """An n-gram language model trained on a given corpus.
    
    For a given n and given training corpus, constructs an n-gram language
    model for the corpus by:
    1. preprocessing the corpus (adding SOS/EOS/UNK tokens)
    2. calculating (smoothed) probabilities for each n-gram

    Also contains methods for calculating the perplexity of the model
    against another corpus, and for generating sentences.

    Args:
        train_data (list of str): list of sentences comprising the training corpus.
        n (int): the order of language model to build (i.e. 1 for unigram, 2 for bigram, etc.).
        laplace (int): lambda multiplier to use for laplace smoothing (default 1 for add-1 smoothing).

    """

    def __init__(self, train_data, n, laplace=1,good_turing=False,no_smoothing=False):
        self.n = n
        self.laplace = laplace
        self.tokens = preprocess(train_data, n)
        self.vocab  = nltk.FreqDist(self.tokens)
        self.good_turing = good_turing
        self.no_smoothing = no_smoothing
        self.model  = self._create_model()
        self.masks  = list(reversed(list(product((0,1), repeat=n))))


    def _no_smoothing(self):
        print("no smoothing")
        n_grams=nltk.ngrams(self.tokens,self.n)
        n_vocab = nltk.FreqDist(n_grams)
        # prob= 每一个ngram出现的次数 / 总的ngrams的数量
        m_grams=nltk.ngrams(self.tokens,self.n-1)
        m_vocab=nltk.FreqDist(m_grams)
        prob={}

        prob={n_gram : count/m_vocab[n_gram[:-1]] for n_gram, count in n_vocab.items()}
       
        return prob


    def _smooth_laplace(self):
        print('laplace smoothing')
        """Apply Laplace smoothing to n-gram frequency distribution.
        
        Here, n_grams refers to the n-grams of the tokens in the training corpus,
        while m_grams refers to the first (n-1) tokens of each n-gram.

        Returns:
            dict: Mapping of each n-gram (tuple of str) to its Laplace-smoothed 
            probability (float).

        """
        vocab_size = len(self.vocab)

        n_grams = nltk.ngrams(self.tokens, self.n)
        # print(n_grams.items()[10])
        n_vocab = nltk.FreqDist(n_grams)



        m_grams = nltk.ngrams(self.tokens, self.n-1)
        m_vocab = nltk.FreqDist(m_grams)

        def smoothed_count(n_gram, n_count):
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            return (n_count + self.laplace) / (m_count + self.laplace * vocab_size)
        prob= { n_gram: smoothed_count(n_gram, count) for n_gram, count in n_vocab.items() }

        
        return prob

    def _good_turing_smoothing(self):
        # 统计不同出现次数的n-grams数量
        ngrams_freq = Counter(nltk.ngrams(self.tokens, self.n))
        counts = Counter(ngrams_freq.values())
        
        mgram_freq=Counter(nltk.ngrams(self.tokens,self.n-1))
        mcounts=Counter(mgram_freq.values())

        smoothed_counts = {}
        for count in ngrams_freq.values():
            if count + 1 in counts:
                smoothed_count = (count + 1) * counts[count + 1] / counts[count]
            else:
                smoothed_count = count
            smoothed_counts[count] = smoothed_count

        m_smoothed_counts={}
        for mcount in mgram_freq.values():
            if mcount+1 in mcounts:
                m_smoothed_count=(mcount+1)*mcounts[mcount+1]/mcounts[mcount]
            else:
                m_smoothed_count=mcount
            m_smoothed_counts[mcount]=m_smoothed_count
        # list=[smoothed_counts[count] / m_smoothed_counts[mgram_freq[ngram[:-1]]] for ngram, count in ngrams_freq.items()]
        # total=sum(list)
        prob_smoothed={}
        for ngram, count in ngrams_freq.items():
            if smoothed_counts[count] / (m_smoothed_counts[mgram_freq[ngram[:-1]]]) >1:
                prob_smoothed[ngram]=1
            else:
                prob_smoothed[ngram]=smoothed_counts[count] / (m_smoothed_counts[mgram_freq[ngram[:-1]]]) 
        smoothedsum=0
        for ngram,cnt in ngrams_freq.items():
            smoothedsum+=smoothed_counts[cnt]
        prob_smoothed['<zero>']=counts[1]/len(self.vocab) #(len(self.tokens)-smoothedsum)/len(self.tokens)
        # print(prob_smoothed['<zero>'])
        return prob_smoothed

    def _create_model(self):
        """Create a probability distribution for the vocabulary of the training corpus.
        
        If building a unigram model, the probabilities are simple relative frequencies
        of each token with the entire corpus.

        Otherwise, the probabilities are Laplace-smoothed relative frequencies.

        Returns:
            A dict mapping each n-gram (tuple of str) to its probability (float).

        """
        if self.n == 1:
            num_tokens = len(self.tokens)
            return { (unigram,): count / num_tokens for unigram, count in self.vocab.items() }
        else:
            if self.good_turing:
                return self._good_turing_smoothing()
            elif self.no_smoothing:
                return self._no_smoothing()
            else:    
                return self._smooth_laplace()

    def _convert_oov(self, ngram):
        """Convert, if necessary, a given n-gram to one which is known by the model.

        Starting with the unmodified ngram, check each possible permutation of the n-gram
        with each index of the n-gram containing either the original token or <UNK>. Stop
        when the model contains an entry for that permutation.

        This is achieved by creating a 'bitmask' for the n-gram tuple, and swapping out
        each flagged token for <UNK>. Thus, in the worst case, this function checks 2^n
        possible n-grams before returning.

        Returns:
            The n-gram with <UNK> tokens in certain positions such that the model
            contains an entry for it.

        """
        mask = lambda ngram, bitmask: tuple((token if flag == 1 else "<UNK>" for token,flag in zip(ngram, bitmask)))

        ngram = (ngram,) if type(ngram) is str else ngram
        for possible_known in [mask(ngram, bitmask) for bitmask in self.masks]:
            if possible_known in self.model:
                return possible_known

    def perplexity(self, test_data):
        """Calculate the perplexity of the model against a given test corpus.
        
        Args:
            test_data (list of str): sentences comprising the training corpus.
        Returns:
            The perplexity of the model as a float.
        
        """
        test_tokens = preprocess(test_data, self.n)
        test_ngrams = nltk.ngrams(test_tokens, self.n)
        N = len(test_tokens)
        prob=[]
        # mgram_freq=nltk.FreqDist(nltk.ngrams(self.tokens,self.n-1))
        test_ngrams=list(test_ngrams)
        temp=0
        for ngram in test_ngrams:
            if ngram not in self.model.keys():
                temp+=1
        # print(temp)
        if self.good_turing:
            none_seen_prob=self.model['<zero>']/temp
        print(temp)
        for ngram in test_ngrams:
            if ngram in self.model.keys():
                prob.append(self.model[ngram])
            elif self.good_turing:
                prob.append(none_seen_prob)
            elif self.laplace:
                prob.append(1/len(self.vocab.keys()))
            else:
                continue
                #if temp>10: return False
        # if temp>1 : return False
        
        # known_ngrams  = (self._convert_oov(ngram) for ngram in test_ngrams)
        # probabilities = [self.model[ngram] for ngram in known_ngrams]

        return math.exp((-1/N) * sum(map(math.log, prob)))

    def _best_candidate(self, prev, i, without=[]):
        """Choose the most likely next token given the previous (n-1) tokens.

        If selecting the first word of the sentence (after the SOS tokens),
        the i'th best candidate will be selected, to create variety.
        If no candidates are found, the EOS token is returned with probability 1.

        Args:
            prev (tuple of str): the previous n-1 tokens of the sentence.
            i (int): which candidate to select if not the most probable one.
            without (list of str): tokens to exclude from the candidates list.
        Returns:
            A tuple with the next most probable token and its corresponding probability.

        """
        blacklist  = ["<UNK>"] + without
        candidates = ((ngram[-1],prob) for ngram,prob in self.model.items() if ngram[:-1]==prev)
        candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
        candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        if len(candidates) == 0:
            return ("</s>", 1)
        else:
            return candidates[0 if prev != () and prev[-1] != "<s>" else i]
     
    def generate_sentences(self, num, min_len=12, max_len=24):
        """Generate num random sentences using the language model.

        Sentences always begin with the SOS token and end with the EOS token.
        While unigram model sentences will only exclude the UNK token, n>1 models
        will also exclude all other words already in the sentence.

        Args:
            num (int): the number of sentences to generate.
            min_len (int): minimum allowed sentence length.
            max_len (int): maximum allowed sentence length.
        Yields:
            A tuple with the generated sentence and the combined probability
            (in log-space) of all of its n-grams.

        """
        for i in range(num):
            sent, prob = ["<s>"] * max(1, self.n-1), 1
            while sent[-1] != "</s>":
                prev = () if self.n == 1 else tuple(sent[-(self.n-1):])
                blacklist = sent + (["</s>"] if len(sent) < min_len else [])
                next_token, next_prob = self._best_candidate(prev, i, without=blacklist)
                sent.append(next_token)
                prob *= next_prob
                
                if len(sent) >= max_len:
                    sent.append("</s>")

            yield ' '.join(sent), -1/math.log(prob)
       
       

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser("N-gram Language Model")
    parser.add_argument('--data', type=str, required=True,
            help='Location of the data directory containing train.txt and test.txt')
    parser.add_argument('--n', type=int, required=True,
            help='Order of N-gram model to create (i.e. 1 for unigram, 2 for bigram, etc.)')
    parser.add_argument('--laplace', type=float, default=0.0,
            help='Lambda parameter for Laplace smoothing (default is 0.00 -- use 1 for add-1 smoothing)')
    parser.add_argument('--good_turing', type=bool, default=False,
            help='Use Good-Turing smoothing instead of Laplace (default False)')
    parser.add_argument('--num', type=int, default=10,
            help='Number of sentences to generate (default 10)')
    parser.add_argument('--no_smoothing',type=bool,default=False,
            help='no smoothing')
    args = parser.parse_args()

    # Load and prepare train/test data
    data_path = Path(args.data)
    train, test1, test2 = load_data(data_path)
    print("choose {}-gram models".format(args.n))
    lm = LanguageModel(train, args.n, laplace=args.laplace,good_turing=args.good_turing,no_smoothing=args.no_smoothing)
    print("Vocabulary size: {}".format(len(lm.vocab)))

    print("Generating sentences...")
    for sentence, prob in lm.generate_sentences(args.num):
        print("{} ({:.5f})".format(sentence, prob))
    
    # plot_dict_values(lm.model)
    perplexity1 = lm.perplexity(test1)
    perplexity2 = lm.perplexity(test2)
    if perplexity1:
        print("Model perplexity for test.1.txt: {:.3f}".format(perplexity1))
    else:
        print("error")
    if perplexity2:
        print("Model perplexity for test.2.txt: {:.3f}".format(perplexity2))
    else:
        print("error")
    
    # print("Model perplexity for test.1.txt: False")
    # print("Model perplexity for test.2.txt: {:.3f}".format(perplexity2))
