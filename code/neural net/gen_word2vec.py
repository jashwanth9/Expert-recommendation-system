#usage: python w2vec_feats.py <path_to_training_data>
#training data has words separated by spaces on each line
import gensim
import numpy as np
import logging
import sys
import cPickle as pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def prep_data(file_name):
	f = open(file_name, 'r')
	data = f.readlines()
	f.close()
	all_sentences = []
	for line in data:
		line = line.replace("\r", "").replace("\n", "")
		all_sentences.append(line.split())
	return all_sentences

def avg_vecs(desc, model, dim, trained_words):
	nwords = 0
	#avg_feats = np.zeros((dims), dtype="float64")
	#if assume it to be zero then cosine similarity becomes nan in some cases
	avg_feats = np.random.randn((dims))
	for word in desc:
		if word in trained_words:
			nwords += 1
			avg_feats = np.add(avg_feats, model[word])
	if nwords == 0:
		return avg_feats
	return avg_feats / nwords

def gen_word2vec(data, dims):
	#min_count is the threshold for the words, a word has to be present min_count times
	model = gensim.models.Word2Vec(data, size=dims, min_count=3)
	#the for which we have embeddings
	trained_words = model.index2word
	return model, trained_words

def compute_similarity(train_feats, test_feats):
	num_train = train_feats.shape[0]
	num_test = test_feats.shape[0]
	cosine_sim = np.zeros((num_test, num_train))
	train_l2 = np.sqrt(np.sum(np.square(train_feats), axis=1))
	for i in range(num_test):
		cosine_sim[i,:] = np.sqrt(np.sum(np.square(train_feats - test_feats[i,:]), axis=1))
		cosine_sim[i,:] = np.sum(train_feats * test_feats[i, :], axis=1)
		test_l2 = np.sqrt(np.sum(np.square(test_feats), axis=1))
		denominator = train_l2 * test_l2
		cosine_sim[i,:] = cosine_sim[i,:] / denominator
	return cosine_sim

#dimentionality of the word vectors
dims = 50
sentences = prep_data(sys.argv[1])
model, trained_words = gen_word2vec(sentences, dims)
#save the model
model.save('w2v_model_words')

#average the word vectors
avg_word_vecs = []
for sentence in sentences:
	avg_word_vecs.append(avg_vecs(sentence, model, dims, trained_words));

avg_word_vecs = np.vstack(avg_word_vecs)
print avg_word_vecs.shape
pickle.dump(avg_word_vecs, open('question_word_wordvec100.p', 'wb'))
#load the model, Just FYI
#loaded_model = gensim.models.Word2Vec.load('w2v_model_words')
#cosine_sim = compute_similarity(avg_word_vecs, avg_word_vecs)
#pickle.dump(cosine_sim, open('question_word_wordvec_sim.p', 'wb')) 