import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA

path_emb = 'vectors.txt'

def get_embeddings(words=None, norm=False):
    all = (words is None or words==[])
    with open(path_emb, 'r') as f:
        embs = [v.split() for v in f if all or v.split()[0] in words]
    if norm:
        return {v[0]:_normalize([float(x) for x in v[1:]]) for v in embs}
    return {v[0]:[float(x) for x in v[1:]] for v in embs}

def get_random_embeddings(n=10, norm=False):
    import random
    with open(path_emb, 'r') as f:
        embs = random.sample([v.split() for v in f], n)
    if norm:
        return {v[0]:_normalize([float(x) for x in v[1:]]) for v in embs}
    return {v[0]:[float(x) for x in v[1:]] for v in embs}

def _normalize(v):
    return v / np.linalg.norm(v)

class PrincipalComponentsAnalysis():
    def __init__(self, embs=None):
        self.pca = PCA()
        if embs is not None:
            self.fit(embs)

    def fit(self, embs):
        X = np.array([e for e in list(embs)])
        self.pca.fit(X)

    def principal_components(self):
        return [(v,r) for v,r in zip(self.pca.explained_variance_, self.pca.explained_variance_ratio_)]

    def principal_direction(self, n=1):
        n = max(1, n)       # If n<=0, converts it to 1 so the method returns 1st dimension
        return self.pca.components_[n-1,:]

def principal_components(embs):
    X = np.array([e for e in list(embs)])
    pca = PCA()
    pca.fit(X)
    return [(v,r) for v,r in zip(pca.explained_variance_, pca.explained_variance_ratio_)]

def principal_direction(embs, n=1):
    n = max(1, n)	# If n<=0, converts it to 1 so the method returns 1st dimension
    X = np.array([e for e in list(embs)])
    pca = PCA()
    pca.fit(X)
    return pca.components[n-1,:]

def euclidean(v1, v2):
    return np.linalg.norm(np.array(v1)-np.array(v2), ord=2)

def cosine(v1, v2):
    return np.abs(1-spatial.distance.cosine(v1,v2))

def direct_bias(source, target):
    return 0

def gendered_bias(source, target1, target2):
    return 0
