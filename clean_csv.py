import pandas as pd
from data import mongo
from common import text
import sys

# ----------------------------> RUN ON DUMPUSERS FIRST!

# Path names
num = 13
code = 't'
is_users = False

if len(sys.argv)==4:
    num = int(sys.argv[1])
    code = str(sys.argv[2])
    is_users = bool(sys.argv[3])

if num<10:
    num = '0' + str(num)
else:
    num = str(num)

if is_users:
    path_in = 'dumpusers' + str(num) + code + '.csv'
else:
    path_in = 'dump' + str(num) + code + '.csv'
path_out = path_in

# Options
remove_rts = True
reorder_cols = True
remove_multiple = True
add_fill_in_cols = True
add_origin = True
add_msg_type = True


ORDERED_COLS = ['id_str', 'in_reply_to_status_id', 'user_id', 'in_reply_to_user_id', 'created_at', 'retweet_count', 'favorite_count', 'full_text']

if 'users' in path_in:
    df = pd.read_csv(path_in, delimiter=';', dtype='str')
    accounts = text.retrieve_accounts({'diputados_congreso.csv':'handle', 'diputados_autonomicos.csv':'twitter account'})
    accounts = [acc.strip().lower() for acc in accounts]
    df['polit'] = 0
    df['polit'][df['screen_name'].str.lower().isin(accounts)] = 1
    dict_aut = text.retrieve_accounts_to_autonomy({'diputados_autonomicos.csv': 'twitter account'})
    df['autname'] = [dict_aut.get(sn) if sn in dict_aut else 'No' for sn in df['screen_name'].str.lower()]
    df = df.drop([x for x in df.columns if 'Unnamed' in x], axis=1)
else:
    df = pd.read_csv(path_in, delimiter=';', dtype='str')
    if remove_rts:
        df = df[~df['full_text'].astype(str).str.startswith('RT @')]
    if reorder_cols:
        df = df.drop([x for x in df.columns if 'Unnamed' in x], axis=1)
        df = df.reindex_axis(ORDERED_COLS, axis=1)
    if remove_multiple:
        drops = (df['in_reply_to_status_id'].isnull()) & (df['full_text'].str.count('@')>1)
        df = df.drop(df[drops].index, axis=0)
    if add_fill_in_cols and 'is_sexist' not in df:
        df['is_sexist'] = ''
    if add_origin:
        accounts = text.retrieve_accounts({'diputados_congreso.csv':'handle'}, low=True) # Accounts from Congreso
        print(accounts[0:10], len(accounts))
        # Add user data from csv
        path_users = 'dumpusers' + str(num) + code + '.csv'
        users = pd.read_csv(path_users, delimiter=';', dtype='str')
        ids = [id for id,name in zip(users['id'], users['screen_name']) if name.lower() in accounts] # User IDs from Congreso
        print(ids[0:10], len(ids))
        df['aut'] = 1
        df['aut'][df['user_id'].isin(ids)] = 0
        accounts_to_auts = text.retrieve_accounts_to_autonomy({'diputados_autonomicos.csv':'twitter account'})
        ids_to_auts = {id:aut for id,aut in zip(users['id'], users['autname'])}
        df['autname'] = [ids_to_auts.get(id) if id in ids_to_auts else 'No' for id in df['user_id']]
        df['week'] = int(path_in[4:6])
    if add_msg_type:
        if path_in[-5]=='t':
            df['msg'] = 'tweet'
        else:
            df['msg'] = 'mention'
            df['msg'][pd.notnull(df['in_reply_to_user_id'])] = 'reply'

df.to_csv(path_out, sep=';', encoding='utf-8')
