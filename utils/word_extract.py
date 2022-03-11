import datetime, nltk, warnings
from grpc import stream_stream_rpc_method_handler
import pandas as pd
from sympy import continued_fraction_convergents
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

is_noun = lambda pos: pos[:2] == 'NN'
def word_extracts(input, attr:str):
    '''
    Extract words in input[attr]
    :param input: data which extracting words from
    :param attr: column which we wanna extract attributes
    :return: extracted words and counts
    '''
    assert isinstance(attr,str)
    
    stemm = nltk.stem.SnowballStemmer("english")
    key_count = dict()
    word_count = dict()
    keyroot = dict()
    for desc in input[attr]:
        if pd.isnull(desc):
            continue
        nouns = []
        desc = desc.lower()
        tokens = nltk.word_tokenize(desc)
        pos_tag = nltk.pos_tag(tokens)
        for (word, pos) in pos_tag:
            if is_noun(pos): nouns.append(word)
        for noun in nouns:
            root = stemm.stem(noun)
            # print(root)
            if root in keyroot.keys(): 
                key_count[root]+=1
                keyroot[root].add(noun)
            else: 
                key_count[root]=1
                keyroot[root]={noun}
        # select proper root for representation
        for i in keyroot.keys():
            minlenth = 100
            word = ''
            for j in keyroot[i]:
                if len(j)<minlenth: 
                    word = j
                    minlenth = len(j)
            word_count[word]=key_count[i]
    return word_count
            
def list_extract(word_count, thresh=13):
    '''
    Extract avaiable words from word_count according to the threshold
    :param word_count: dictionary containing word and corresponding count
    :param thresh: only consider words appear more than thresh times
    :return: words list
    '''
    word_list = []
    for key, value in word_count.items():
        if value < thresh: continue
        if len(key) < 4: continue
        if key in ['tag','pink','blue','green','orange']: continue
        word_list.append([key,value])
    return word_list
            


