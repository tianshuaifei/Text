#coding=utf-8
from gensim.models.word2vec import LineSentence, Word2Vec


sentences = LineSentence("../../w2v_corpus.txt")
model = Word2Vec(sentences, size=200, window=5, min_count=1, workers=4)
model.save_word2vec_format("../../vec.txt", binary=True)
#model.save("../../vec.txt")