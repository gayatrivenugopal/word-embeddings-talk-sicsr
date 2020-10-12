#Source: https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

from gensim.models import Word2Vec
# define training data
sentences = [['यह', 'है', 'मेरा', 'पहला', 'वाक्य'],
			['यह', 'है', 'मेरा', 'दूसरा', 'वाक्य'],
			['यह', 'है', 'एक', 'नया' 'वाक्य'],
			['एक','और', 'वाक्य'],
			['और', 'एक', 'अंतिम', 'वाक्य']]
# train model
model = Word2Vec(sentences, min_count = 1, sg = 1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['वाक्य'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model.most_similar(positive='वाक्य', topn=5))
