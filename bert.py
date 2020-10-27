#Source: https://colab.research.google.com/drive/1ZQvuAVwA3IjybezQOXnrXMGAnMyZRuPU#scrollTo=UeQNEFbUgMSf, https://medium.com/@dhartidhami/understanding-bert-word-embeddings-7dc4d2ea54ca

#from tokenizers import SentencePieceBPETokenizer as tokenizer
import torch
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine


text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = tokenizer.tokenize(marked_text)

# Print out the tokens.
print (tokenized_text)

#words that are not part of the ~30K vocabulary are represented as subwords and characters

#map the tokens to their vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

for tup in zip(tokenized_text, indexed_tokens):
    print(tup[0], tup[1])

#print(list(tokenizer.vocab.keys())[2005]) #for

#create sentence classes
segments_ids = [1] * len(tokenized_text) #mark each token in the sentence as 1 because we have only one sentence

#BERT requires inputs in the form of tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True) #the model returns all hidden-states.
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Run the text through BERT, and collect all of the hidden states produced from all the layers.
with torch.no_grad(): #deactivates gradient calculation, saves memory; gradient calculation is not needed since we are running a forward pass
    hidden_states = model(tokens_tensor, segments_tensors)[2]
    #hidden states has 4 dimensions: layer number (1 embeddings layer + 12 BERT layers), batch number (1 for 1 sentence), token number, hidden unit (768)

    print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
    layer_i = 0
    print ("Number of batches:", len(hidden_states[layer_i]))
    batch_i = 0
    print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
    token_i = 0
    print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

    #sum of the last 4 layers gives the best result in many cases
    # Stores the token vectors, with shape [len(tokens) x 768]
    token_vecs_sum = []
    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)
    # token_embeddings is a [13 x 1 x 22 x 768] tensor.
    # Remove dimension 1, the "batches" to make the dimension as 13 x 22 x 768
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    #print(token_embeddings.size())
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)

    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec) # [22 x 768]

    for i, token_str in enumerate(tokenized_text):
          print (i, token_str)

    '''
    print('First 5 vector values for each instance of "bank".')
    print('')
    print("bank vault   ", str(token_vecs_sum[6][:5]))
    print("bank robber  ", str(token_vecs_sum[10][:5]))
    print("river bank   ", str(token_vecs_sum[19][:5]))
    '''

    print(len(token_vecs_sum))
    # Calculate the cosine similarity between the word bank in "bank robber" vs "river bank" (different meanings).
    diff_bank = 1 - cosine(token_vecs_sum[10], token_vecs_sum[19])
    # Calculate the cosine similarity between the word bank in "bank robber" vs "bank vault" (same meaning).
    same_bank = 1 - cosine(token_vecs_sum[10], token_vecs_sum[6])
    print('Vector similarity for  *similar*  meanings:  %.2f' % same_bank)
    print('Vector similarity for *different* meanings:  %.2f' % diff_bank)

