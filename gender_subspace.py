import numpy as np
from common import embeddings as emb
from common import metadata

vms = []
vfs = []
diffs = []
males = metadata.source_males
females = metadata.source_females

for n in [True, False]:
    rnd = emb.get_random_embeddings(len(males), norm=n).values()
    embs = emb.get_embeddings(males+females, norm=n)
    for m,f in zip(males, females):
        vm = np.array(embs[m])
        vf = np.array(embs[f])
        vms.append(vm)
        vfs.append(vf)
        diffs.append(np.abs(vm-vf))
    print('Normalizing...', n)
    print('\tRandom:', emb.PrincipalComponentsAnalysis(rnd).principal_components())
    print('\tMales:', emb.PrincipalComponentsAnalysis(vms).principal_components())
    print('\tFemales:', emb.PrincipalComponentsAnalysis(vfs).principal_components())
    print('\tDifferences:', emb.PrincipalComponentsAnalysis(diffs).principal_components())
