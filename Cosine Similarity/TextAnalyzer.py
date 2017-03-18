import sys
import argparse
import numpy as np
from pyspark import SparkContext

def toLowerCase(s):
    """ Convert a sting to lowercase. E.g., 'BaNaNa' becomes 'banana'
    """
    return s.lower()

def stripNonAlpha(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """
    return ''.join([c for c in s if c.isalpha()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text Analysis through TFIDF computation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('mode', help='Mode of operation',choices=['TF','IDF','TFIDF','SIM','TOP']) 
    parser.add_argument('input', help='Input file or list of files.')
    parser.add_argument('output', help='File in which output is stored')
    parser.add_argument('--master',default="local[20]",help="Spark Master")
    parser.add_argument('--idfvalues',type=str,default="idf", help='File/directory containing IDF values. Used in TFIDF mode to compute TFIDF')
    parser.add_argument('--other',type=str,help = 'Score to which input score is to be compared. Used in SIM mode')
    args = parser.parse_args()
  
    sc = SparkContext(args.master, 'Text Analysis', '/shared/apps/spark/spark-1.4.1-bin-hadoop2.4', None)


    if args.mode=='TF' :
        # Read text file at args.input, compute TF of each term, 
        # and store result in file args.output. All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings, i.e., "" 
        # are also removed


        lines = sc.textFile(args.input)
  
        lines.flatMap(lambda s: s.split())\
                .map(toLowerCase)\
                .map(stripNonAlpha)\
                .map(lambda word: (word, 1))\
                .reduceByKey(lambda x, y: x + y)\
                .filter(lambda (key,val):key!="")\
                .saveAsTextFile(args.output)

    if args.mode=='TOP':
        # Read file at args.input, comprizing strings representing pairs of the form (TERM,VAL), 
        # where TERM is a string and VAL is a numeric value. Find the pairs with the top 20 values,
        # and store result in args.output
        pairs =sc.textFile(args.input)
        
        top=pairs.map(eval)\
             .takeOrdered(20,key=lambda (term,val):-val)\
        
        with open(args.output,'w') as f:
             f.write("\n".join([str(x) for x in top]))                     
        
    if args.mode=='IDF':
        # Read list of files from args.input, compute IDF of each term,
        # and store result in file args.output.  All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings ""
        # are removed
        files = sc.wholeTextFiles(args.input).cache()
        
        corpusSize=files.keys().count()

        files.flatMapValues(lambda s:s.split())\
             .mapValues(toLowerCase)\
             .mapValues(stripNonAlpha)\
             .distinct()\
             .map(lambda x:(x[1],1))\
             .reduceByKey(lambda x,y: x+y)\
             .mapValues(lambda x:np.log(corpusSize/(1.*x)))\
             .filter(lambda (key,val): key!="")\
             .saveAsTextFile(args.output)

        print 'Corpus Size is',corpusSize

    if args.mode=='TFIDF':
        # Read  TF scores from file args.input the IDF scores from file args.idfvalues,
        # compute TFIDF score, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase letter-only string and VAL is a numeric value. 
        tf = sc.textFile(args.input).map(eval)
        idf = sc.textFile(args.idfvalues).map(eval)
        
        tfidf=tf.join(idf).mapValues(lambda (tf,idf): tf*idf)
        tfidf.saveAsTextFile(args.output)
        
    if args.mode=='SIM':
        # Read  scores from file args.input the scores from file args.other,
        # compute the cosine similarity between them, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL), 
        # where TERM is a lowercase, letter-only string and VAL is a numeric value. 
        
        scores1 = sc.textFile(args.input).map(eval).cache()
        scores2 = sc.textFile(args.other).map(eval).cache()
        
        norm1 = np.sqrt(scores1.values().map(lambda x : x**2).reduce(lambda x,y:x+y))
        norm2 = np.sqrt(scores2.values().map(lambda x : x**2).reduce(lambda x,y:x+y))
        
        product = scores1.join(scores2).mapValues(lambda (s1,s2): s1*s2).values().reduce(lambda x,y:x+y)
        similarity = product/(norm1*norm2)

        with open(args.output,'w') as f:
             f.write('\t'.join(map(str,[args.input,args.other,similarity]))+"\n")                     
