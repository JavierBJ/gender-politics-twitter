import pandas as pd
from common import text
import configparser

ORDERED_COLS = ['id_str', 'in_reply_to_status_id', 'user_id', 'in_reply_to_user_id', 'created_at', 'retweet_count', 'favorite_count', 'full_text']    


def clean_users(csv_path):
    print('Cleaning users csv...')
    config = configparser.ConfigParser()
    config.read('config.ini')
    df = pd.read_csv(csv_path, delimiter=';', dtype='str')
    accounts = text.retrieve_accounts({config['PATH']['PathCon']:config['PATH']['HandleCon'], 
                                       config['PATH']['PathAut']:config['PATH']['HandleAut']})
    accounts = [acc.strip().lower() for acc in accounts]
    df['polit'] = 0
    df['polit'][df['screen_name'].str.lower().isin(accounts)] = 1
    dict_aut = text.retrieve_accounts_to_autonomy({config['PATH']['PathAut']:config['PATH']['HandleAut']})
    df['autname'] = [dict_aut.get(sn) if sn in dict_aut else 'No' for sn in df['screen_name'].str.lower()]
    df = df.drop([x for x in df.columns if 'Unnamed' in x], axis=1)
    df.to_csv(csv_path, sep=';', encoding='utf-8')
    print('Cleaned users csv.')
    
def clean_tweets(csv_path, csv_path_users):
    print('Cleaning tweets csv...')
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    df = pd.read_csv(csv_path, delimiter=';', dtype='str')
    # Remove RTs
    df = df[~df['full_text'].astype(str).str.startswith('RT @')]
    
    # Reorder columns
    df = df.drop([x for x in df.columns if 'Unnamed' in x], axis=1)
    df = df.reindex_axis(ORDERED_COLS, axis=1)
    
    # Remove multiple mentions
    drops = (df['in_reply_to_status_id'].isnull()) & (df['full_text'].str.count('@')>1)
    df = df.drop(df[drops].index, axis=0)
    
    # Add empty is_sexist column to prepare for annotation
    df['is_sexist'] = ''
    
    # Add identification columns: aut, autname, week
    accounts = text.retrieve_accounts({config['PATH']['PathCon']:config['PATH']['HandleCon']}, low=True) # Accounts from Congreso
    print(accounts[0:10], len(accounts))
    # Add user tweets from csv
    users = pd.read_csv(csv_path_users, delimiter=';', dtype='str')
    ids = [ident for ident,name in zip(users['id'], users['screen_name']) if name.lower() in accounts] # User IDs from Congreso
    print(ids[0:10], len(ids))
    df['aut'] = 1
    df['aut'][df['user_id'].isin(ids)] = 0
    #accounts_to_auts = text.retrieve_accounts_to_autonomy({'diputados_autonomicos.csv':'twitter account'})
    ids_to_auts = {ident:aut for ident,aut in zip(users['id'], users['autname'])}
    df['autname'] = [ids_to_auts.get(ident) if ident in ids_to_auts else 'No' for ident in df['user_id']]
    df['week'] = int(csv_path[4:6])
    
    # Add message type column
    if csv_path[-5]=='t':
        df['msg'] = 'tweet'
    else:
        df['msg'] = 'mention'
        df['msg'][pd.notnull(df['in_reply_to_user_id'])] = 'reply'
    df.to_csv(csv_path, sep=';', encoding='utf-8')
    print('Cleaned tweets csv.')



