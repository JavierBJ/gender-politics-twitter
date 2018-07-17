import pandas as pd

path = 'dump02t.csv'

csv = pd.read_csv(path, delimiter=';', dtype='str')

rts = 0
for tweet in csv['full_text']:
    if tweet.startswith('RT @'):
        rts +=1
    else:
        print(tweet)

total = len(csv)
print('Total tweets:', total)
print('\t RTs:', rts)
print('\t Non RTs:', (total-rts))