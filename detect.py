import argparse
from common import detection

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='run experiments of detection of tweets.')
    parser.add_argument('dv', choices=['hostility', 'sexism', 'gender'], help='dependent variable, outcome to be predicted')
    parser.add_argument('alg', choices=['lasso','l1','ridge','l2','svm','nb','mlp','rf'], help='learning algorithm sklearn compatible')
    parser.add_argument('prep', choices=['lemma', 'form'], help='type of preprocessing of the text')
    parser.add_argument('how', choices=['binary', 'counts', 'tfidf'], help='how to count the features')
    parser.add_argument('-d', '--dbname', default='gender', help='name of the mongodb collection where the data is')
    parser.add_argument('-t', '--limit', type=int, default=0, help='maximum number of tweets recovered (0 for all)')
    parser.add_argument('-k', '--kfolds', type=int, default=10, help='k for k-fold cross-validation')
    parser.add_argument('-s', '--stopwords', action='store_true', help='remove stopwords if active')
    parser.add_argument('-q', '--keepwordsfreq', type=int, help='number of appearances in the corpus for a feature to be considered')
    parser.add_argument('-r', '--keepwordsrank', type=int, help='number of features to be used ranked by number of appearances')
    
    # Only -w or -c must be specified, but not both
    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument('-w', '--bagofwords', type=int, help='use bag-of-word-ngrams of the specified n')
    mutex.add_argument('-c', '--bagofchars', type=int, help='use bag-of-character-ngrams of the specified n')
    
    # Parameters for fine-tuning of algorithms (a list of values can be specified)
    parser.add_argument('-a', '--alpha', type=float, nargs='+', default = 1.0, help='regularization parameters to try (only in L1, L2, SVM, MLP)')
    parser.add_argument('-n', '--hidden', type=int, nargs='+', default = 500, help='number of hidden neurons (only in MLP)')
    parser.add_argument('-l', '--learningrate', type=float, nargs='+', default = 0.0001, help='learning rate (only in MLP)')
    parser.add_argument('-m', '--maxfeatures', type=float, nargs='+', default = 1.0, help='max percentage of features to consider in split (only in RF)')
    parser.add_argument('-f', '--minsamplesleaf', type=int, nargs='+', default = 1, help='min samples to make a leaf (only in RF)')
    
    args = parser.parse_args()
    print('dv:', args.dv)
    print('alg:', args.alg)
    print('prep:', args.prep)
    print('how:', args.how)
    print('dbname:', args.dbname)
    print('limit:', args.limit)
    print('kfolds:', args.kfolds)
    print('stopwords:', args.stopwords)
    print('keepwordsfreq:', args.keepwordsfreq)
    print('keepwordsrank:', args.keepwordsrank)
    print('bagofwords:', args.bagofwords)
    print('bagofchars:', args.bagofchars)
    print('alpha:', args.alpha)
    print('hidden:', args.hidden)
    print('learningrate:', args.learningrate)
    print('maxfeatures:', args.maxfeatures)
    print('minsamplesleaf:', args.minsamplesleaf)

    detection.detect(args.dv, args.alg, args.prep, args.how, args.dbname, args.limit, args.kfolds, args.stopwords, args.keepwordsfreq, args.keepwordsrank, args.bagofwords, args.bagofchars, args.alpha, args.hidden, args.learningrate, args.maxfeatures, args.minsamplesleaf)
    # TODO: get appropriate output
    