import numpy as np
from common import embeddings
from common import metadata

N_REPS = 1
emb_size = 300
vms = []
vfs = []
diffs = []
males = metadata.source_males
females = metadata.source_females


emb = embeddings.Embeddings(emb_size)
rnd_result = 0
for i in range(N_REPS):
    rnd = emb.get_random_embeddings(len(males), norm=False).values()
    rnd_comp = embeddings.PrincipalComponentsAnalysis(rnd).principal_components()
    print(rnd_comp)
    rnd_result += rnd_comp[0][1] / rnd_comp[1][1]
rnd_result /= N_REPS

embs = emb.get_embeddings(males+females, norm=False)
for m,f in zip(males, females):
    vm = np.array(embs[m])
    vf = np.array(embs[f])
    diffs.append(np.abs(vm-vf))
diffs_comp = embeddings.PrincipalComponentsAnalysis(diffs).principal_components()
print(diffs_comp)
diffs_result = diffs_comp[0][1] / diffs_comp[1][1]

print('Random (100 runs):', rnd_result)
print('Gender subspace:', diffs_result)

for n in [True, False]:
    emb = embeddings.Embeddings(emb_size)
    rnd = emb.get_random_embeddings(len(males), norm=n).values()
    embs = emb.get_embeddings(males+females, norm=n)
    for m,f in zip(males, females):
        vm = np.array(embs[m])
        vf = np.array(embs[f])
        vms.append(vm)
        vfs.append(vf)
        diffs.append(np.abs(vm-vf))
    print('Normalizing...', n)
    print('\tRandom:', embeddings.PrincipalComponentsAnalysis(rnd).principal_components())
    print('\tMales:', embeddings.PrincipalComponentsAnalysis(vms).principal_components())
    print('\tFemales:', embeddings.PrincipalComponentsAnalysis(vfs).principal_components())
    print('\tDifferences:', embeddings.PrincipalComponentsAnalysis(diffs).principal_components())
