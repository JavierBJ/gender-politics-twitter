import pyfreeling as freeling
import pandas as pd
from random import shuffle as shf
import configparser

def retrieve_accounts(dict_files, low=False):
    accounts = pd.Series()
    for file, column in dict_files.items():
        dataset = pd.read_csv(file, sep=';', dtype='str')
        if low:
            accounts = accounts.append(dataset[column].dropna().str.lower(), ignore_index=True)
        else:
            accounts = accounts.append(dataset[column].dropna(), ignore_index=True)
    return accounts

def retrieve_accounts_by_gender(dict_files):
    dict_gender = {}
    for file, column in dict_files.items():
        dataset = pd.read_csv(file, sep=';', dtype='str')
        dict_gender.update({d[column]:d['gender'] for d in dataset})
    return dict_gender

def retrieve_names_by_gender(dict_files, extr=True, low=False):
    dict_gender = {}
    for file, column in dict_files.items():
        dataset = pd.read_csv(file, sep=';', dtype='str')
        dataset = dataset.to_dict('records')
        if low:
            apply = lambda x: x.lower() if x==x else x
        else:
            apply = lambda x:x
        if extr:
            dict_gender.update({apply(_extract_name(record[column])):int(record['gender']) for record in dataset})
        else:
            dict_gender.update({apply(record[column]):int(record['gender']) for record in dataset})
    return dict_gender

def retrieve_accounts_to_autonomy(dict_files):
    dict_aut = {}
    for file, column in dict_files.items():
        dataset = pd.read_csv(file, sep=';', dtype='str')
        for data, ccaa in zip(dataset[column],dataset['ccaa']):
            if data==data:
                data = data.lower()
            dict_aut.update({data:ccaa})
    return dict_aut

def _extract_name(name):
    if 'Sr.' in name or 'Sra.' in name:
        return (name.split('.')[-1]).split(' ')[0].strip()
    elif ',' in name:
        return (name.split(',')[-1]).strip()
    else:
        return (name.split(' ')[0]).strip()

def create_dataset(from_file, limit=None, shuffle=False):
    print('Creating dataset...')
    if shuffle: # Shuffle order of reading of indicated datasets
        shf(from_file)
    
    dataset = pd.DataFrame()
    for file in from_file:
        data = pd.read_csv(file, delimiter=';', encoding='utf-8', engine='python', dtype='str')
        if shuffle: # Shuffle each dataset indicated
            data = data.sample(frac=1)
        
        if limit is not None and len(dataset)+len(data)>limit:    # Take only the part that fits our limit
            dataset = dataset.append(data[:limit-len(dataset)-len(data)], ignore_index=True)
            break   # Stop iterating datasets, no more data2 fits our limit
        else:
            dataset = dataset.append(data, ignore_index=True)
    print('\tDataset created.')
    return dataset

def setup_freeling():
    config = configparser.ConfigParser()
    config.read('config.ini')
    freeling.util_init_locale('default')
    tk = freeling.tokenizer(config['FREELING']['Data'] + config['FREELING']['Lang'] + '/twitter/tokenizer.dat')
    sp = freeling.splitter(config['FREELING']['Data'] + config['FREELING']['Lang'] + '/splitter.dat')
    umap = freeling.RE_map(config['FREELING']['Data'] + config['FREELING']['Lang'] + '/twitter/usermap.dat')
    
    # maco options to be activated and their data2 files
    op= freeling.maco_options("es");
    op.set_data_files("", 
            config['FREELING']['Data'] + "common/punct.dat",
            config['FREELING']['Data'] + config['FREELING']['Lang'] + "/dicc.src",
            config['FREELING']['Data'] + config['FREELING']['Lang'] + "/afixos.dat",
            "",
            config['FREELING']['Data'] + config['FREELING']['Lang'] + "/locucions.dat", 
            config['FREELING']['Data'] + config['FREELING']['Lang'] + "/np.dat",
            config['FREELING']['Data'] + config['FREELING']['Lang'] + "/quantities.dat",
            config['FREELING']['Data'] + config['FREELING']['Lang'] + "/probabilitats.dat");
            
    mf=freeling.maco(op);
    mf.set_active_options(False, True, True, True, # User_Map is already done before maco
            True, True, False, True,
            True, True, True, True );
    return tk, sp, umap, mf

def preprocess(df, filter_lang=None):
    print('Preprocessing tweets...')
    if filter_lang is not None:
        ids = identify_language(df)
        out = [ident in filter_lang for ident in ids]
        df = df.drop(df[out].index)
        df = df.reset_index(drop=True) # TODO: make some column a real index with set_index() to avoid this problem
        print('\tFiltered out', str(len([o for o in out if o])), 'tweets due to language.')
    
    ls_tokens = []
    tk, sp, umap, mf = setup_freeling()
    for col in ['full_text', 'in_reply_to_text']:
        for _,row in df.iterrows():
            try:
                raw_text = row[col]
                #print(raw_text)
                tokens = tk.tokenize(raw_text)
                tokens = sp.split(tokens)
                tokens = umap.analyze(tokens)
                tokens = mf.analyze(tokens)
                ls_tokens.append(tokens)
            except Exception:
                ls_tokens.append(tokens)
        df = df.drop(col, axis=1)
        df[col] = pd.Series(ls_tokens)
    print('\tTweets preprocessed.')
    return df

def tokenize(df):
    print('Tokenizing tweets...')
    tk, _, _, _ = setup_freeling()
    all_tokens = []
    for _, row in df.iterrows():
        raw_text = row['full_text']
        tokens = tk.tokenize(raw_text)
        tokens = [w.get_form() for w in tokens]
        all_tokens.extend(tokens)
    print('\tTweets tokenized...')
    return ' '.join(all_tokens)	# String of whitespace-separated tokens

def identify_language(df):
    config = configparser.ConfigParser()
    config.read('config.ini')
    freeling.util_init_locale('default')
    ident = freeling.lang_ident(config['FREELING']['Data'] + "common/lang_ident/ident.dat")
    
    identifications = []
    for _,row in df.iterrows():
        raw_text = str(row['full_text'])
        identifications.append(ident.identify_language(raw_text))
    return identifications

