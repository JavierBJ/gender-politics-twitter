from common import embeddings as emb

words = ['el', 'ella', 'hombre', 'mujer', 'tio', 'tia', 'niño', 'niña', 'chico', 'chica', 'guapo', 'guapa']
embs = emb.get_embeddings(words)
print('Embeddings of gender-words...')
for k,v in embs.items():
    print('\t', k, ':', v)
print('Explained variances...')
print(emb.PrincipalComponentsAnalysis(embs.values()).principal_components())

rembs = emb.get_random_embeddings(10)
print('Embeddings of random words...')
for k,v in rembs.items():
    print('\t', k, ':', v)
print('Explained variances...')
print(emb.PrincipalComponentsAnalysis(rembs.values()).principal_components())

print('Distances EL - ELLA:')
print('\tCosine:', emb.cosine(embs['el'], embs['ella']))
print('\tEuclidean:', emb.euclidean(embs['el'], embs['ella']))

print('Distances GUAPO - GUAPA:')
print('\tCosine:', emb.cosine(embs['guapo'], embs['guapa']))
print('\tEuclidean:', emb.euclidean(embs['guapo'], embs['guapa']))


print('EL - GUAPO:', emb.cosine(embs['el'], embs['guapo']))
print('ELLA - GUAPA:', emb.cosine(embs['ella'], embs['guapa']))
print('HOMBRE - GUAPO:', emb.cosine(embs['hombre'], embs['guapo']))
print('MUJER - GUAPA:', emb.cosine(embs['mujer'], embs['guapa']))
