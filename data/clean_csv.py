import pandas as pd


# Path names
path_in = 'dump01mr.csv'
path_out = path_in

# Options
remove_rts = True
reorder_cols = True
remove_multiple = True
add_fill_in_cols = True


ORDERED_COLS = ['id_str', 'in_reply_to_status_id', 'user_id', 'in_reply_to_user_id', 'created_at', 'retweet_count', 'full_text']

df = pd.read_csv(path_in, delimiter=';')
if remove_rts:
    df = df[~df['full_text'].astype(str).str.startswith('RT @')]
if reorder_cols:
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.reindex_axis(ORDERED_COLS, axis=1)
if remove_multiple:
    drops = (df['in_reply_to_status_id'].isnull()) & (df['full_text'].str.count('@')>1)
    df = df.drop(df[drops].index, axis=0)
if add_fill_in_cols and 'is_sexist' not in df:
    df['is_sexist'] = ''

df.to_csv(path_out, sep=';', encoding='utf-8')