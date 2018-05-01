import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA

path_emb = 'vectors.txt'

def get_embeddings(words=None):
    all = (words is None or words==[])
    with open(path_emb, 'r') as f:
        embs = [v.split() for v in f if all or v.split()[0] in words]
    return {v[0]:_normalize([float(x) for x in v[1:]]) for v in embs}

def get_random_embeddings(n=10):
    import random
    with open(path_emb, 'r') as f:
        embs = random.sample([v.split() for v in f], n)
    return {v[0]:_normalize([float(x) for x in v[1:]]) for v in embs}

def _normalize(v):
    return v / np.linalg.norm(v)

def principal_components(embs):
    X = np.array([e for e in list(embs.values())])
    pca = PCA()
    pca.fit(X)
    return [(v,r) for v,r in zip(pca.explained_variance_, pca.explained_variance_ratio_)]

def euclidean(v1, v2):
    return np.linalg.norm(np.array(v1)-np.array(v2), ord=2)

def cosine(v1, v2):
    return spatial.distance.cosine(v1,v2)
