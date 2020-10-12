#Source: https://towardsdatascience.com/word-embeddings-in-2020-review-with-code-examples-11eb39a1ee6d

# Download Google Word2Vec embeddings https://code.google.com/archive/p/word2vec/
!wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
!gunzip GoogleNews-vectors-negative300.bin
# Try Word2Vec with Gensim
import gensim
# Load pretrained vectors from Google
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
king = model['king']
# king - man + woman = queen
print(model.most_similar(positive=['woman', 'king'], negative=['man']))
