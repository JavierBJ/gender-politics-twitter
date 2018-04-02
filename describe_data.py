from data import descriptive

desc = descriptive.DBDescriptor()

m, f = desc.followers_per_gender()
print('Followers per gender:')
print('\tMale politicians:', m)
print('\tFemale politicians:', f)

m, f = desc.favorites_received_per_gender()
print('Favorites per gender:')
print('\tMale politicians:', m)
print('\tFemale politicians:', f)

m, f = desc.retweets_received_per_gender()
print('Retweets per gender:')
print('\tMale politicians:', m)
print('\tFemale politicians:', f)

results = desc.contingency_table()
print('Contingency table')
print('Replies by Author / Respondent gender:')
print('               Respondent')
print('             Male --- Female')
print('       Male |', str(results[(1,1)]),' | ', str(results[(1,-1)]))
print('Author')
print('       Female |', str(results[(-1,1)]),' | ', str(results[(-1,-1)]))
