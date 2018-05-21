from common import embeddings as emb
from common import metadata

def gendered_bias_neutral(source_males, source_females, target_neutral):
    gb = 0
    embs = emb.get_embeddings(source_males+source_females+target_neutral)
    for male, female in zip(source_males, source_females):
        for neutral in target_neutral:
            cos_male = emb.cosine(embs[male], embs[neutral])
            cos_female = emb.cosine(embs[female], embs[neutral])
            ratio = cos_male/cos_female
            print('\t', male, neutral, cos_male)
            print('\t', female, neutral, cos_female)
            print('\t\tRatio', ratio)
            gb += ratio
    gb /= (len(target_neutral)+len(source_males))
    return gb

# Direct bias (gender-neutral alignment to gender-direction)
'''embs = emb.get_embeddings(source_males+source_females)
gender_dir = emb.PrincipalComponentsAnalysis(source_embs.values()).principal_direction()
db = 0
print('Direct bias...')
for v in target_neutral:
    cos = emb.cosine(gender_dir, target_embs[v])
    db += cos
    print('\t', v, cos)
db /= len(target_embs)
print('Direct bias:', db, '\n')'''

# Gendered bias with gender-neutral qualificative words
print('Gendered bias on gender-neutral qualificative words...')
gb = gendered_bias_neutral(metadata.source_males, metadata.source_females, metadata.target_qualif_neutral)
print('Gendered bias gender-neutral:', gb, '\n')

# Gendered bias with gender-neutral gender words
print('Gendered bias on gender-neutral gender-issues words...')
gb = gendered_bias_neutral(metadata.source_males, metadata.source_females, metadata.target_gender_neutral)
print('Gendered bias gender-neutral:', gb, '\n')

# Gendered bias with gender-neutral other words
print('Gendered bias on gender-neutral other words...')
gb = gendered_bias_neutral(metadata.source_males, metadata.source_females, metadata.target_other_neutral)
print('Gendered bias gender-neutral:', gb, '\n')

import sys
sys.exit(0)

# Gendered bias with gender-paired words
target_males = metadata.target_males
target_females = metadata.target_females
target_embs = emb.get_embeddings(target_males+target_females)

gb *= (len(target_neutral)+len(source_males))
print('Gendered bias on gender-paired words...')
for male, female in zip(source_males, source_females):
    for m, f in zip(target_males, target_females):
        cos_male = emb.cosine(source_embs[male], target_embs[m])
        cos_female = emb.cosine(source_embs[female], target_embs[f])
        ratio = cos_male/cos_female
        print('\t', male, m, cos_male)
        print('\t', female, f, cos_female)
        print('\t\tRatio', ratio)
        gb += ratio
gb /= (len(target_males)+len(source_males))
print('Gendered bias gender-paired:', gb, '\n')

gb *= (len(target_males)+len(source_males))
gb /= (len(target_males)+len(source_males)*2+len(target_neutral))
print('Gendered bias aggregated:', gb)
