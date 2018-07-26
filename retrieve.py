from tweets import dump_tweets, dump_to_csv
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='retrieve political tweets from Twitter API.')
    parser.add_argument('-t', '--notweets', action='store_false', help='do not download original tweets by politicians')
    parser.add_argument('-m', '--nomentions', action='store_false', help='do not download mentions to politicians')
    parser.add_argument('-r', '--noreplies', action='store_false', help='do not download replies to politicians')
    parser.add_argument('-n', '--numfile', type=int, default=-1, help='number of the write-file. If not specified, max in directory + 1')
    parser.add_argument('-l', '--limit', type=int, default=1000, help='upper limit of every type of tweet to be downloaded for each user')
    parser.add_argument('-s', '--since', type=int, default=-1, help='recover only tweets since this id')

    args = parser.parse_args()
    print('notweets:', args.notweets)
    print('nomentions:', args.nomentions)
    print('noreplies:', args.noreplies)
    print('numfile:', args.numfile)
    print('limit:', args.limit)
    print('since:', args.since)
    
    pkl_path = dump_tweets.dump(args.notweets, args.nomentions, args.noreplies, args.numfile, args.limit, args.since)
    dump_to_csv.pkl_to_csv(pkl_path)
