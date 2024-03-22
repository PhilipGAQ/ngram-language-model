
import nltk
from collections import Counter
import heapq

def top_k_ngrams(ngrams_prob, k):
    # 使用 heapq.nlargest 函数获取字典中前 k 个最大值及其对应的键
    top_k_prob = heapq.nlargest(k, ngrams_prob.items(), key=lambda item: item[1])
    # 返回结果
    return top_k_prob
