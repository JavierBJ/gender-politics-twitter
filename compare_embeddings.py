from common import embeddings as emb

source = ['hombre', 'mujer', 'tio', 'tia', \
        'niño', 'niña', 'chico', 'chica', 'hijo', 'hija']
occupation = ['diputado', 'diputada', 'presidente', 'presidenta', \
        'ministro', 'ministra', 'profesor', 'profesora', 'alcalde', 'alcaldesa', \
        'juez', 'jueza']
target = ['guapo', 'guapa', 'tonto', 'tonta', 'cojones', 'huevos', 'gilipollas', 'corazón', \
        'cariño', 'besos', 'beso']


''' # All-vs-all
embs = emb.get_embeddings(source + occupation + target)
for w in source+occupation+target:
    for v in source+occupation+target:
        if v!=w:
            d = emb.cosine(embs[v], embs[w])
            print(v, '-', w, d) '''

# Group-wise
embs = emb.get_embeddings(source + occupation + target)
for s in source:
    for o in occupation:
        print(s, '-', o, emb.cosine(embs[s], embs[o]))
    for t in target:
        print(s, '-', t, emb.cosine(embs[s], embs[t]))
for o in occupation:
    for t in target:
        print(o, '-', t, emb.cosine(embs[o], embs[t]))
