import numpy as np
import pandas as pd
import pickle

dump_name = 'dump03'


f1 = open(dump_name+'t.pkl', 'rb')
tweets_recovered, _ ,_ = pickle.load(f1)
print(len(tweets_recovered))
print(tweets_recovered[0].AsDict().keys())
print(tweets_recovered[0])

'''f2 = open(dump_name+'mr.pkl', 'rb')
_, mentions_recovered, replies_recovered = pickle.load(f2)
print(len(mentions_recovered),len(replies_recovered))
print(mentions_recovered[0].AsDict().keys())
print(mentions_recovered[0])
print(replies_recovered[0].AsDict().keys())
print(replies_recovered[0])'''