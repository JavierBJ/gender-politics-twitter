from common import metadata
import pyfreeling as freeling
import pandas as pd
from random import shuffle as shf
from data import mongo
from collections import Counter

def retrieve_accounts(dict_files):
    accounts = pd.Series()
    for file, column in dict_files.items():
        dataset = pd.read_csv(file, sep=';')
        accounts = accounts.append(dataset[column].dropna(), ignore_index=True)
    return accounts

def retrieve_accounts_by_gender(dict_files):
    dict_gender = {}
    for file, column in dict_files.items():
        dataset = pd.read_csv(file, sep=';')
        dict_gender.update({d[column]:d['gender'] for d in dataset})
    return dict_gender

def retrieve_names_by_gender(dict_files):
    dict_gender = {}
    for file, column in dict_files.items():
        dataset = pd.read_csv(file, sep=';')
        dataset = dataset.to_dict('records')
        dict_gender.update({_extract_name(record[column]):record['gender'] for record in dataset})
    return dict_gender

def retrieve_accounts_to_autonomy(dict_files):
    dict_aut = {}
    for file, column in dict_files.items():
        dataset = pd.read_csv(file, sep=';')
        for data, ccaa in zip(dataset[column],dataset['ccaa']):
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
        data = pd.read_csv(file, delimiter=';', encoding='utf-8', engine='python')
        if shuffle: # Shuffle each dataset indicated
            data = data.sample(frac=1)
        
        if limit is not None and len(dataset)+len(data)>limit:    # Take only the part that fits our limit
            dataset = dataset.append(data[:limit-len(dataset)-len(data)], ignore_index=True)
            break   # Stop iterating datasets, no more data fits our limit
        else:
            dataset = dataset.append(data, ignore_index=True)
    print('\tDataset created.')
    return dataset

def setup_freeling():
    freeling.util_init_locale('default')
    tk = freeling.tokenizer(metadata.DATA + metadata.LANG + '/twitter/tokenizer.dat')
    sp = freeling.splitter(metadata.DATA + metadata.LANG + '/splitter.dat')
    umap = freeling.RE_map(metadata.DATA + metadata.LANG + '/twitter/usermap.dat')
    
    # maco options to be activated and their data files
    op= freeling.maco_options("es");
    op.set_data_files("", 
            metadata.DATA + "common/punct.dat",
            metadata.DATA + metadata.LANG + "/dicc.src",
            metadata.DATA + metadata.LANG + "/afixos.dat",
            "",
            metadata.DATA + metadata.LANG + "/locucions.dat", 
            metadata.DATA + metadata.LANG + "/np.dat",
            metadata.DATA + metadata.LANG + "/quantities.dat",
            metadata.DATA + metadata.LANG + "/probabilitats.dat");
            
    mf=freeling.maco(op);
    mf.set_active_options(False, True, True, True, # User_Map is already done before maco
            True, True, False, True,
            True, True, True, True );
    return tk, sp, umap, mf

def preprocess(df, filter_lang=None):
    print('Preprocessing tweets...')
    if filter_lang is not None:
        ids = identify_language(df)
        out = [id in filter_lang for id in ids]
        df = df.drop(df[out].index)
        df = df.reset_index(drop=True) # TODO: make some column a real index with set_index() to avoid this problem
        print('\tFiltered out', str(len([o for o in out if o])), 'tweets due to language.')
    
    ls_tokens = []
    tk, sp, umap, mf = setup_freeling()
    for _,row in df.iterrows():
        try:
            raw_text = row['full_text']
            #print(raw_text)
            tokens = tk.tokenize(raw_text)
            tokens = sp.split(tokens)
            tokens = umap.analyze(tokens)
            tokens = mf.analyze(tokens)
            ls_tokens.append(tokens)
        except Exception:
            ls_tokens.append(tokens)
    df = df.drop('full_text', axis=1)
    df['full_text'] = pd.Series(ls_tokens)
    print('\tTweets preprocessed.')
    return df

def identify_language(df):
    freeling.util_init_locale('default')
    ident = freeling.lang_ident(metadata.DATA +  "common/lang_ident/ident.dat")
    
    identifications = []
    for _,row in df.iterrows():
        raw_text = str(row['full_text'])
        identifications.append(ident.identify_language(raw_text))
    return identifications

if __name__ == '__main__':
    df = db.import_tweets_mongodb(limit=1000)
    df = preprocess(df)
    for i in range(10):
        print([[x.get_form() for x in y.get_words()] for y in df['full_text'][i]])
        print([[x.get_lemma() for x in y.get_words()] for y in df['full_text'][i]])
        print([[x.get_tag() for x in y.get_words()] for y in df['full_text'][i]])
        print()

