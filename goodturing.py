import nltk
from collections import defaultdict

def count_ngrams(ngrams):
    freq_counts = defaultdict(int)
    for gram in ngrams:
        freq_counts[gram] += 1
    return freq_counts

def good_turing(freq_counts,totalcount):
    freq_of_freq = defaultdict(int)
    for freq in freq_counts.values():
        freq_of_freq[freq] += 1
    
    gt_counts = {}
    for gram, freq in freq_counts.items():
        if freq + 1 in freq_of_freq:
            gt_counts[gram] = (freq + 1) * freq_of_freq[freq + 1] / (freq_of_freq[freq]*totalcount)
        else:
            gt_counts[gram] = freq_counts[gram]/totalcount
    
    return gt_counts

def good_turing_smoothing(ngrams,totalcount):
    freq_counts = count_ngrams(ngrams)
    gt_estimates = good_turing(freq_counts,totalcount)
    return gt_estimates