from common import metadata
import pyfreeling
import pandas as pd
from random import shuffle as shf
from data import mongo

def retrieve_accounts(dict_files):
    accounts = pd.Series()
    for file, column in dict_files.items():
        dataset = pd.read_csv(file, sep=';')
        accounts = accounts.append(dataset[column].dropna(), ignore_index=True)
    return accounts

def retrieve_accounts_by_gender(dict_files):
    accounts = {}
    for file, column in dict_files.items():
        dataset = pd.read_csv(file, sep=';')
        accounts = accounts.append(dataset[column].dropna(), ignore_index=True)
    return accounts

def retrieve_names_by_gender(dict_files):
    pass

def create_dataset(from_file, limit=None, shuffle=False):
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
    
    return dataset

def setup_freeling():
    pyfreeling.util_init_locale('default')
    tk = pyfreeling.tokenizer(metadata.DATA + metadata.LANG + '/twitter/tokenizer.dat')
    sp = pyfreeling.splitter(metadata.DATA + metadata.LANG + '/splitter.dat')
    umap = pyfreeling.RE_map(metadata.DATA + metadata.LANG + '/twitter/usermap.dat')
    
    # maco options to be activated and their data files
    op= pyfreeling.maco_options("es");
    op.set_data_files("", 
            metadata.DATA + "common/punct.dat",
            metadata.DATA + metadata.LANG + "/dicc.src",
            metadata.DATA + metadata.LANG + "/afixos.dat",
            "",
            metadata.DATA + metadata.LANG + "/locucions.dat", 
            metadata.DATA + metadata.LANG + "/np.dat",
            metadata.DATA + metadata.LANG + "/quantities.dat",
            metadata.DATA + metadata.LANG + "/probabilitats.dat");
            
    mf=pyfreeling.maco(op);
    mf.set_active_options(False, True, True, True, # User_Map is already done before maco
            True, True, False, True,
            True, True, True, True );
    return tk, sp, umap, mf

def preprocess(df):
    ls_tokens = []
    tk, sp, umap, mf = setup_freeling()
    
    for _,row in df.iterrows():
        raw_text = row['full_text']
        tokens = tk.tokenize(raw_text)
        tokens = sp.split(tokens)
        tokens = umap.analyze(tokens)
        tokens = mf.analyze(tokens)
        ls_tokens.append(tokens)
    df = df.drop('full_text', axis=1)
    df['full_text'] = pd.Series(ls_tokens)
    return df


if __name__ == '__main__':
    df = db.import_tweets_mongodb(limit=1000)
    df = preprocess(df)
    for i in range(10):
        print([[x.get_form() for x in y.get_words()] for y in df['full_text'][i]])
        print([[x.get_lemma() for x in y.get_words()] for y in df['full_text'][i]])
        print([[x.get_tag() for x in y.get_words()] for y in df['full_text'][i]])
        print()

