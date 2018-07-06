import argparse
from common import models

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='run experiments of language analysis.')
    parser.add_argument('who', choices=['author, receiver'], help='outcome to be predicted: author or receiver gender')
    parser.add_argument('alg', choices=['lasso','l1','ridge','l2','svm'], help='learning algorithm')
    parser.add_argument('-d', '--dbname', default='gender', help='name of the mongodb collection where the data is')
    parser.add_argument('-t', '--limit', type=int, default=0, help='maximum number of tweets recovered (0 for all)')
    parser.add_argument('-k', '--kfolds', type=int, default=10, help='k for k-fold cross-validation')
    parser.add_argument('-s', '--stopwords', action='store_true', help='remove stopwords if active')
    parser.add_argument('-q', '--keepwordsfreq', type=int, help='number of appearances in the corpus for a feature to be considered')
    parser.add_argument('-r', '--keepwordsrank', type=int, help='number of features to be used ranked by number of appearances')
    parser.add_argument('-a', '--alpha', type=float, nargs='+', default = 1.0, help='regularization parameters to try')
    
    args = parser.parse_args()
    print('who:', args.who)
    print('alg:', args.alg)
    print('dbname:', args.dbname)
    print('limit:', args.limit)
    print('kfolds:', args.kfolds)
    print('stopwords:', args.stopwords)
    print('keepwordsfreq:', args.keepwordsfreq)
    print('keepwordsrank:', args.keepwordsrank)
    print('alpha:', args.alpha)

    models.analyze(args.who, args.dbname, args.limit, args.kfolds, args.stopwords, args.keepwordsfreq, args.keepwordsrank, args.alpha)
    # Output of this?
