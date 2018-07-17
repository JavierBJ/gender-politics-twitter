import numpy as np
from common import embeddings
import configparser

N_REPS = 1
emb_size = 300
vms = []
vfs = []
diffs = []

# Part 1: Gender subspace
# ---- -- ------ --------
config = configparser.ConfigParser()
config.read('words.ini')
males = list(config['SOURCEMALES'].values())
females = list(config['SOURCEFEMALES'].values())

# Create random subspaces and average them for comparability
emb = embeddings.Embeddings(emb_size)
rnd_result = 0
for i in range(N_REPS):
    rnd = emb.get_random_embeddings(len(males), norm=False).values()
    rnd_comp = embeddings.PrincipalComponentsAnalysis(rnd).principal_components()
    rnd_result += rnd_comp[0][1] / rnd_comp[1][1]
rnd_result /= N_REPS

# Create gender subspace
embs = emb.get_embeddings(males+females, norm=False)
for m,f in zip(males, females):
    vm = np.array(embs[m])
    vf = np.array(embs[f])
    diffs.append(np.abs(vm-vf))
diffs_comp = embeddings.PrincipalComponentsAnalysis(diffs).principal_components()
diffs_result = diffs_comp[0][1] / diffs_comp[1][1]

print('Random:', rnd_result)
print('Gender subspace:', diffs_result)

# Part 2: Direct Bias
# ---- -- ------ ----
config = configparser.ConfigParser()
config.read('words.ini')

source_embs = emb.get_embeddings(list(config['SOURCEMALES'].values())+list(config['SOURCEFEMALES'].values()))
target_embs = emb.get_embeddings(list(config['TARGETMODELS'].values()))
gender_dir = embeddings.PrincipalComponentsAnalysis(source_embs.values()).principal_direction()
db = 0
for v in config['TARGETMODELS'].values():
    cos = embeddings.cosine(gender_dir, target_embs[v])
    db += cos
db /= len(target_embs)
print('Direct bias:', db)
