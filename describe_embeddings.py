from common import embeddings as emb

words = ['el', 'ella', 'hombre', 'mujer', 'tio', 'tia', 'niño', 'niña', 'chico', 'chica']
embs = emb.get_embeddings(words)
print('Embeddings of gender-words...')
for k,v in embs.items():
    print('\t', k, ':', v)
print('Explained variances...')
print(emb.principal_components(embs))

rembs = emb.get_random_embeddings(10)
print('Embeddings of random words...')
for k,v in rembs.items():
    print('\t', k, ':', v)
print('Explained variances...')
print(emb.principal_components(rembs))
