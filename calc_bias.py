import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def read_file(embedding_file_name):
    with open(embedding_file_name, 'r', encoding='utf-8') as embedding_file:
        tokens = []
        token_to_id = dict()
        response = dict()
        for line in embedding_file:
            word, *values = line.strip().split(' ')
            ind = len(tokens)
            tokens.append(word)
            token_to_id[word] = ind
            response[ind]= np.array([np.float(x) for x in values])
        return token_to_id, response    

def get_tokens(values, token_to_id, e1):
    return [e1[token_to_id[x]] for x in values]

def g(S,A,B):
    results = []
    for s in S:
        s = np.reshape(s, (-1,1))
        A_mean = np.mean([cosine_similarity(s, np.reshape(x, (-1,1))) for x in A])
        B_Mean = np.mean([cosine_similarity(s, np.reshape(x, (-1,1))) for x in B])
        results.append(A_mean - B_Mean)
    return np.mean(results)


def main():
    A = ['math', 'calculus', 'science', 'physics']    
    B = ['poetry','literature', 'symphony', 'sculpture'] 
    S = ['male', 'man', 'boy', 'brother', 'he']
    T = ['female', 'woman', 'girl', 'sister', 'she']
    t1, em1 = read_file('plain.txt')
    t2, em2 = read_file('complex.txt')
    t3, em3 = read_file('glove.6B.300d.txt')
    A1, B1, S1, T1 = list(map(lambda x: get_tokens(x, t1, em1), [A,B,S,T]))
    plain_mean = g(S1,A1,B1) - g(T1,A1,B1)
    print('Plain Mean', plain_mean)
    A2, B2, S2, T2 = list(map(lambda x: get_tokens(x, t2, em2), [A,B,S,T]))
    complex_mean = g(S2,A2,B2) - g(T2,A2,B2)
    print('Plain Mean', complex_mean)
    

if __name__ == '__main__':
    main()