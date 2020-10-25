#Source: https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# define training data
sentences = [['यह', 'है', 'मेरा', 'पहला', 'वाक्य'],
			['यह', 'है', 'मेरा', 'दूसरा', 'वाक्य'],
			['यह', 'है', 'एक', 'नया', 'वाक्य'],
			['एक','और', 'वाक्य'],
			['और', 'एक', 'अंतिम', 'वाक्य']]
# train model
model = Word2Vec(sentences, min_count = 1, sg = 1)
# summarize the loaded model
#print(model)
# summarize vocabulary
#words = list(model.wv.vocab)
#print(words)
# access vector for one word
print(model['वाक्य'])
# save model
model.save('model.bin')
# load model
model = Word2Vec.load('model.bin')
print(model.most_similar(positive='वाक्य', topn=5))
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
plt.rcParams['font.sans-serif'] = ['Lohit Devanagari']
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
