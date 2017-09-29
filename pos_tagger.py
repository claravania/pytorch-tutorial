import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pdb

from torch.autograd import Variable

torch.manual_seed(1123)

def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	tensor = torch.LongTensor(idxs)
	return Variable(tensor)


training_data = [
	("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
	("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_ix = {}
char_to_ix = {}
for sent, tags in training_data:
	for word in sent:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)
		for char in word:
			if char not in char_to_ix:
				char_to_ix[char] = len(char_to_ix)
print(word_to_ix)
print(char_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

EMBEDDING_DIM = 8
CHAR_EMBEDDING_DIM = 12
HIDDEN_DIM = 15
CHAR_HIDDEN_DIM = 10


class LSTMTagger(nn.Module):

	def __init__(self, embedding_dim, char_embedding_dim, hidden_dim, char_hidden_dim, vocab_size, char_vocab_size, tagset_size):
		super(LSTMTagger, self).__init__()
		self.hidden_dim = hidden_dim
		self.char_hidden_dim = char_hidden_dim
		self.char_embeddings = nn.Embedding(char_vocab_size, char_embedding_dim)
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		
		self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)
		self.lstm = nn.LSTM(embedding_dim + char_hidden_dim, hidden_dim)

		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
		self.lstm_hidden = self.init_hidden()
		self.char_lstm_hidden = self.init_char_hidden()


	def init_hidden(self):
		return (Variable(torch.zeros(1, 1, self.hidden_dim)), Variable(torch.zeros(1, 1, self.hidden_dim)))


	def init_char_hidden(self):
		return (Variable(torch.zeros(1, 1, self.char_hidden_dim)), Variable(torch.zeros(1, 1, self.char_hidden_dim)))


	def forward(self, sentence, char_inputs):
		embeds = self.word_embeddings(sentence)

		# compute character LSTM representation
		# char_inputs is an array consists of character sequences of each word
		char_embeds = []
		for ch_seq in char_inputs:
			# clear char LSTM hidden states
			self.char_lstm_hidden = self.init_char_hidden()

			# map to character embeddings
			char_embed = self.char_embeddings(ch_seq)

			# run single LSTM over characters
			char_lstm_out, self.char_lstm_hidden = self.char_lstm(
				char_embed.view(len(ch_seq), 1, -1), self.char_lstm_hidden)

			# get the last hidden state
			last_hidden, _ = self.char_lstm_hidden
			char_embeds.append(last_hidden)

		char_embeds = torch.squeeze(torch.stack(char_embeds))

		# concat word embeddings and character embeddings
		embeds = torch.cat((embeds, char_embeds), 1)

		lstm_out, self.lstm_hidden = self.lstm(
			embeds.view(len(sentence), 1, -1), self.lstm_hidden)
		tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
		tag_scores = F.log_softmax(tag_space)
		return tag_scores


model = LSTMTagger(EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, CHAR_HIDDEN_DIM, len(word_to_ix), len(char_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


inputs = prepare_sequence(training_data[0][0], word_to_ix)
char_inputs = []
for w in training_data[0][0]:
	char_in = prepare_sequence(w, char_to_ix)
	char_inputs.append(char_in)
tag_scores = model(inputs, char_inputs)
print(tag_scores)

for epoch in range(300):
	for sentence, tags in training_data:
		# clear gradients
		model.zero_grad()
		# clear hidden states
		model.lstm_hidden = model.init_hidden()

		# prepare sentence input and target
		sentence_in = prepare_sequence(sentence, word_to_ix)
		targets = prepare_sequence(tags, tag_to_ix)

		# prepare character inputs
		char_inputs = []
		for w in sentence:
			char_in = prepare_sequence(w, char_to_ix)
			char_inputs.append(char_in)

		# run forward pass
		tag_scores = model(sentence_in, char_inputs)

		# compute loss
		loss = loss_function(tag_scores, targets)
		loss.backward()
		optimizer.step()


inputs = prepare_sequence(training_data[0][0], word_to_ix)
char_inputs = []
for w in training_data[0][0]:
	char_in = prepare_sequence(w, char_to_ix)
	char_inputs.append(char_in)
tag_scores = model(inputs, char_inputs)

print(tag_scores)

