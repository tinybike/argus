"""Extract question|sentence|gs class"""
import csv
import numpy as np

# Used when tokenizing words
sentence_re = r'''(?x)      # set flag to allow verbose regexps
      (?:[A-Z])(?:\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
    | \w+(?:-\w+)*            # words with optional internal hyphens
    | \$?\d+(?:\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
    | \.\.\.                # ellipsis
    | [][.,;"'?():-_`]      # these are separate tokens
'''

import string
import nltk
import re
def tokenize(string):
    return nltk.regexp_tokenize(string, sentence_re)

def generate(name):
    data = []
    for line in csv.reader(open(name+'/all_features.tsv'), delimiter='\t'):
        if line[2] not in 'YES NO':
            continue
        q = ' '.join(tokenize(re.sub(r'[^\x00-\x7F]+',' ', line[0]))).decode('utf-8', )
        s = ' '.join(tokenize(re.sub(r'[^\x00-\x7F]+',' ', line[1]))).decode('utf-8', )
        l = '1' if line[2] == 'YES' else '0'
        data.append((q,l,s))
    writer = csv.writer(open('argus_'+name+'.csv', 'wb'), delimiter=',')
    writer.writerow(['htext', 'label', 'mtext'])
    for triplet in data:
        writer.writerow(triplet)


generate('train')
generate('val')
generate('test')


for line in csv.reader(open('argus_val.csv'), delimiter=','):
    print line[-1]