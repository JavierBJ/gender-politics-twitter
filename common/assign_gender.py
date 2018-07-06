from common import text
import pandas as pd
import json
import requests
from urllib.request import urlopen
from urllib.parse import quote
import configparser

r_genders_path = '../r-genders.csv'
paths_to_names = {'diputados_autonomicos.csv':'diputado', 'diputados_congreso.csv':'nombre'}
paths_to_screennames={'diputados_autonomicos.csv':'twitter account', 'diputados_congreso.csv':'handle'}
paths_to_genders = {'diputados_autonomicos.csv':'gender', 'diputados_congreso.csv':'gender'}
associations_api = {'male': 1, 'female': -1}

def tag_politicians_gender(users):
    names_to_genders = text.retrieve_names_by_gender(paths_to_screennames, extr=False, low=True)
    is_male = []
    for username,polit,gen in zip(users['screen_name'], users['polit'], users['gender']):
        polit = int(polit)
        gen = int(gen)
        username = username.lower()
        if gen==0:
            if polit==1:
                if username in names_to_genders:
                    is_male.append(-2*names_to_genders[username]+1)	# csv are encoded as male:0 female:1. -2n+1 gives male:1 female:-1.
                else:
                    print('Failed at', username)	# Should not happen
                    is_male.append(0)
            else:
                is_male.append(0)	# Gender of individuals to be assigned by other methods
        else:
            is_male.append(gen)	# Already assigned by another method
    users['gender'] = is_male
    return users

def tag_gender_from_r(users, path, thr):
    genders = pd.read_csv(path)
    dict_genders = {name:gender for name, gender in zip(genders['name'], genders['proportion_male'])}
    
    if thr>0.5:
        thr = 1-thr
    
    is_male = []
    for user, gender in zip(users['name'], users['gender']):
        if gender==0:
            user = str(user).split(' ')[0]
            if user in dict_genders:
                if dict_genders[user]<thr:
                    is_male.append(-1)   # 1 == male
                elif dict_genders[user]>(1-thr):
                    is_male.append(1)  # -1 == female
                else:
                    is_male.append(0)   # 0 == don't know
            else:
                is_male.append(0)
        else:
            is_male.append(gender)
    users['gender'] = is_male
    return users

def tag_gender_from_gender_api(users, thr):
    if thr<1:
        thr *= 100
    
    for i, (user, gender) in enumerate(zip(users['name'], users['gender'])):
        if gender==0:
            allowed = _check_limit_gender_api()
            if allowed:
                results = _query_gender_api(user)
                try:
                    if results['gender'] in associations_api and results['accuracy']>=thr:
                        users['gender'].iloc[[i]] = associations_api[results['gender']]
                except Exception:
                    print('Failed at', results)
            else:
                print('Gender-API: monthly limit reached')
                break
    return users

def _check_limit_gender_api():
    config = configparser.ConfigParser()
    config.read('config.ini')
    url = 'https://gender-api.com/get-stats?&key=' + config.GenderKey
    response = urlopen(url)
    decoded = response.read().decode('utf-8')
    data = json.loads(decoded)
    return not bool(data['is_limit_reached'])

def _query_gender_api(user):
    config = configparser.ConfigParser()
    config.read('config.ini')
    user = quote(str(user.split(' ')[0]))
    url = 'https://gender-api.com/get?key=' + config.GenderKey + '&name=' + user + '&country=ES'
    response = urlopen(url)
    decoded = response.read().decode('utf-8')
    return json.loads(decoded)

def tag_gender_from_genderize_api(users, thr):
    results = _query_genderize_api(users['name'])
    name_to_res = {elem[0]:elem for elem in results}
    
    is_male = []
    for user, gender in zip(users['name'], users['gender']):
        if gender==0:
            user = str(user).split(' ')[0]
            if user in name_to_res:
                if name_to_res[user][1]<thr:
                    is_male.append(-1)   # 1 == male
                elif name_to_res[user][1]>(1-thr):
                    is_male.append(1)  # -1 == female
                else:
                    is_male.append(0)   # 0 == don't know
        else:
            is_male.append(gender)
    users['gender'] = is_male
    return users

def _query_genderize_api(names):
    '''
    Source: https://github.com/block8437/gender.py
    '''
    url = ""
    cnt = 0
    
    for name in list(names):
        name = quote(str(name).split(' ')[0])
        if url == "":
            url = "name[0]=" + str(name)
        else:
            cnt += 1
            url = url + "&name[" + str(cnt) + "]=" + str(name)
        
    req = requests.get("https://api.genderize.io?" + url)
    req.read().decode('utf-8')
    results = json.loads(req.text)
    
    retrn = []
    for result in results:
        if result["gender"] is not None:
            retrn.append((result["gender"], result["probability"], result["count"]))
        else:
            retrn.append((u'None',u'0.0',0.0))
    return retrn

def tag_gender_from_politicians(users):
    names_to_genders = text.retrieve_names_by_gender(paths_to_names, low=True)
    is_male = []
    for username,gen in zip(users['name'], users['gender']):
        if gen==0:
            if username != username: # Check NaN
                is_male.append(0)
            else:
                username = username.lower()
                subnames = _split_name(username)
                is_male.append(_gender_from_splits(subnames, names_to_genders))
        else:
            is_male.append(gen)	# Gender already assigned by another method
    users['gender'] = is_male
    return users

def _split_name(name):
    subnames = []
    uninames = name.split(' ') # Add unigram names
    l = len(uninames)
    binames = []
    for i in range(l-1): # Add bigram names by concatenating unigram names
        binames.append(str(uninames[i])+str(uninames[i+1]))
    subnames.extend(uninames)
    subnames.extend(binames)
    return subnames

def _gender_from_splits(subnames, names_to_genders):
    for sn in subnames:
        if sn in names_to_genders:
            return -2*names_to_genders[sn]+1	#Tagged as male:0 female:1. -2n+1 makes it male:1 female:-1
    return 0	# Default case for when no subname matched a known name

def tokenize_names(names):
    ls_tokens = []
    tk, sp, umap, mf = text.setup_freeling()
    
    for name in names:
        tokens = tk.tokenize(name)
        tokens = sp.split(tokens)
        tokens = umap.analyze(tokens)
        tokens = mf.analyze(tokens)
        ls_tokens.append(tokens)

