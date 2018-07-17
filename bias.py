import numpy as np
from common import embeddings
import configparser
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='create gender subspace in the embedding space and calculate direct bias.')
    parser.add_argument('-e', '--embsize', type=int, default=300, help='dimensionality of the embedding vectors')
    parser.add_argument('-r', '--repetitions', type=int, default=100, help='how many times to repeat the random subspace and avoid noise')
    
    args = parser.parse_args()
    print('embsize:', args.embsize)
    print('repetitions:', args.repetitions)
    print()
    
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
    emb = embeddings.Embeddings(args.embsize)
    rnd_result = 0
    for i in range(args.repetitions):
        rnd = emb.get_random_embeddings(len(males)).values()
        rnd_comp = embeddings.PrincipalComponentsAnalysis(rnd).principal_components()
        rnd_result += rnd_comp[0][1] / rnd_comp[1][1]
    rnd_result /= args.repetitions
    
    # Create gender subspace
    embs = emb.get_embeddings(males+females)
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
    targets = list(config['TARGETMODELS'].values())
    target_embs = emb.get_embeddings(targets)
    gender_dir = embeddings.PrincipalComponentsAnalysis(embs.values()).principal_direction()
    db = 0
    for v in config['TARGETMODELS'].values():
        cos = embeddings.cosine(gender_dir, target_embs[v])
        db += cos
    db /= len(target_embs)
    print('Direct bias:', db)
