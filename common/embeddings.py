import numpy as np
from sklearn.decomposition import PCA

path_emb = 'vectors.txt'

def get_embeddings(words):
    with open(path_emb, 'r') as f:
        embs = [v.split() for v in f if v.split()[0] in words]
    return {v[0]:v[1:] for v in embs}

def get_random_embeddings(n):
    import random
    with open(path_emb, 'r') as f:
        embs = random.sample([v.split() for v in f], n)
    return {v[0]:v[1:] for v in embs}

def principal_components(embs):
    X = np.array([e for e in list(embs.values())])
    pca = PCA()
    pca.fit(X)
    return [(v,r) for v,r in zip(pca.explained_variance_, pca.explained_variance_ratio_)]
