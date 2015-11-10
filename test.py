from pathlib import Path
import nltk
import gensim
import glove
from pprint import pprint

size = 100
window = 9
threads = 4

def clean_word(w):
    i = 0
    while i < len(w):
        if not w[i].isalnum():
            w = w[:i] + w[i+1:] 
        else:
            i += 1
    return w.lower()

def get_data():
    for s in nltk.corpus.gutenberg.sents():
        yield [w for w in map(clean_word, s) if w != '']

if Path('word2vec_model').exists():
    word2vec_model = gensim.models.word2vec.Word2Vec.load('word2vec_model')
    print('loaded word2vec_model')
else:
    print('word2vec_model not found, buidling...')
    word2vec_model = gensim.models.word2vec.Word2Vec(
        list(get_data()),
        size=size,
        window=window // 2,
        min_count=5,
        workers=threads
    )
    word2vec_model.save('word2vec_model')
    print('done, saved word2vec_model')

if Path('glove_model').exists():
    glove_model = glove.Glove.load('glove_model')
    print('loaded glove_model')
else:
    print('glove_model not found, buidling...')
    glove_corpus = glove.Corpus()
    glove_corpus.fit(
        get_data(),
        window=window
    )  
    print('corpus ready...')
    glove_model = glove.Glove(no_components=size, max_count=20)
    glove_model.fit(glove_corpus.matrix, no_threads=threads)
    glove_model.add_dictionary(glove_corpus.dictionary)
    glove_model.save('glove_model')
    print('done, saved glove_model')

def use(word, n=10):
    print('word2vec:')
    pprint(word2vec_model.most_similar(positive=[word], negative=[], topn=n))
    print('glove:')
    pprint(glove_model.most_similar(word, n))


use('man', 50)



