# coding=utf-8

""" 1-nearest neighbor based probing experiments """

import os
import argparse
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import pickle
from nltk.corpus import wordnet as wn


def fix_seed():
	""" Enable reproducibility """
	np.random.seed(123)
	random.seed(123)
	os.environ['PYTHONHASHSEED'] = str(123)


def load_data_concat_new(input_file):
	"""Return 
			1. the concatenation of vectors in pairs, in [num_pairs, vec_size*2] 
			2. "is-a" label, in [num_pairs,]
			3. details, in [num_pairs,]
	"""
	vecs = []
	labels = []
	lemmas = []
	details = []
	with open(input_file) as f:
		for line in f:
			line = line.strip().split('\t')
			vec_concat = np.concatenate([np.array(eval(line[2])),np.array(eval(line[3]))], 0)
			vecs.append(vec_concat)
			labels.append(int(line[-2]))
			lemmas.append(line[:2])
			details.append(line[-1])
	return np.array(vecs), np.array(labels), lemmas, details




fix_seed()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-train_dir', required=True, help='Directory to a file containing pairs of train. contextualized or baseline vectors of words in context, and their is-a relation labels.')
parser.add_argument('-test_dir', required=True, help='Directory to a file containing pairs of test. contextualized or baseline vectors of words in context, and their is-a relation labels.')
parser.add_argument('--input_mode', required=True, choices=['concat'], help='How to form a pair of input vectors into a relation representation that will be input to classification programs.')
parser.add_argument('--probing_mode', required=True, choices=['full', 'fewshot'], help='Probing in the full or fewshot data setting.')

args = parser.parse_args()


# Load data
train_name = 'vecs_train.txt'
input_path = args.train_dir
if args.input_mode == 'concat':
	train_X, train_y, train_lemmas, train_details = load_data_concat_new(os.path.join(input_path, train_name))
else:
	print('This input mode is currently not supported.')

test_name = 'vecs_test.txt'
input_path = args.test_dir
if args.input_mode == 'concat':
	test_X, test_y, test_lemmas, test_details = load_data_concat_new(os.path.join(input_path, test_name))
else:
	print('This input mode is currently not supported.')



for num_k in [1]: #,5,10]:

	# Train and test k-nearest neighbor classifier
	classifier = KNeighborsClassifier(n_neighbors=num_k)
	classifier.fit(train_X, train_y)
	test_preds = classifier.predict(test_X)
	precision_mi, recall_mi, f1score_mi, _ = precision_recall_fscore_support(test_y, test_preds, average='micro')
	print('precision_mi, recall_mi, f1score_mi', precision_mi, recall_mi, f1score_mi)
	score_report = classification_report(test_y, test_preds)
	print(score_report)
	class_1_acc = len([_ for i in range(len(test_y)) if test_y[i]==1 and test_preds[i]==1])/len([_ for i in range(len(test_y)) if test_y[i]==1])
	class_0_acc = len([_ for i in range(len(test_y)) if test_y[i]==0 and test_preds[i]==0])/len([_ for i in range(len(test_y)) if test_y[i]==0])
	print('Acc class (1hop) isa ', class_1_acc)
	print('Acc class not isa ', class_0_acc)
	
	# Get each hop isa's acc/F1
	# print(test_details)
	test_y_by_hop = {}
	test_preds_by_hop = {}
	#for x in test_details:
	for i in range(len(test_details)):
		x = test_details[i]
		pos_tag = wn.lemma(test_lemmas[i][0][7:-2]).synset().pos()
		if x != '_' and pos_tag == 'n':	# ground-truth class 1, we focus on 'n' (noun) in our analysis
			x_hop = eval(x)['hops']
			if x_hop not in test_y_by_hop:
				test_y_by_hop[x_hop] = [1]
			else:
				test_y_by_hop[x_hop].append(1)
			if x_hop not in test_preds_by_hop:
				test_preds_by_hop[x_hop] = [test_preds[i]]
			else:
				test_preds_by_hop[x_hop].append(test_preds[i])


	acc_by_hop = {}
	for k in test_y_by_hop:
		acc_by_hop[k] = len([_ for i in range(len(test_y_by_hop[k])) if test_y_by_hop[k][i]==1 and test_preds_by_hop[k][i]==1])/len(test_y_by_hop[k])
	
	print('Acc class 1 by hop ', acc_by_hop)

	# For both classes, f1
	if args.probing_mode == 'full':
		class_0_y = test_y[1998:] # fullprobe
		assert 1 not in class_0_y
		class_0_preds = test_preds[1998:]
	else:
		class_0_y = test_y[480:] # noun
		assert 1 not in class_0_y
		class_0_preds = test_preds[480:]


	test_y_by_hop1 = {}
	test_preds_by_hop1 = {}
	for i in range(len(test_details)):
		x = test_details[i]
		pos_tag = wn.lemma(test_lemmas[i][0][7:-2]).synset().pos()
		if x != '_' and pos_tag == 'n': #'v': #'n': #'v'
			x_hop = eval(x)['hops']
			label_x = test_y[i]
			if x_hop not in test_y_by_hop1:
				test_y_by_hop1[x_hop] = [label_x]
			else:
				test_y_by_hop1[x_hop].append(label_x)
			if x_hop not in test_preds_by_hop1:
				test_preds_by_hop1[x_hop] = [test_preds[i]]
			else:
				test_preds_by_hop1[x_hop].append(test_preds[i])

	cummulative_start = 0
	for k,v in test_y_by_hop1.items():
		cur_len = len(v)
		test_y_by_hop1[k].extend([0]*cur_len)
		neg_pred_list = class_0_preds[cummulative_start:cummulative_start+cur_len]
		test_preds_by_hop1[k].extend(neg_pred_list)
		cummulative_start += cur_len

	f1_by_hop = {}
	for k in test_y_by_hop1:
		f1_by_hop[k] = f1_score(test_y_by_hop1[k], test_preds_by_hop1[k], average="micro")

	print('f1 by hop ', f1_by_hop)

	# Save results
	out_name = 'detailed_result_knn_wholeBert_multihop_' + args.input_mode + '_kOf' + str(num_k) +\
	'_' + args.train_dir.split('/')[0] + '_' + args.test_dir.split('/')[-1] + '.txt'
	with open(os.path.join(args.test_dir, out_name), 'w') as f:
		f.write('Micro_PRF\t{:.4f}\n'.format(f1score_mi))
		f.write('Acc class all-hop isa\t{:.4f}\n'.format(class_1_acc))
		f.write('Acc class non-isa\t{:.4f}\n'.format(class_0_acc))
		for i in sorted(list(acc_by_hop.keys())):
			f.write('Class {}-hop isa Acc\t{:.4f}\tnum_instances\t{}\n'.format(str(i), acc_by_hop[i], str(len(test_y_by_hop[i]))))
		for i in sorted(list(f1_by_hop.keys())):
			f.write('Overall {}-hop Micro_PRF\t{:.4f}\tnum_instances\t{}\n'.format(str(i), f1_by_hop[i], str(len(test_y_by_hop1[i]))))
		f.write('Test pred\tGround truth\tlemmas\tdetails\n')
		for i in range(len(test_details)):
			f.write('{}\t{}\t{}\t{}\n'.format(str(test_preds[i]), str(test_y[i]), test_lemmas[i], test_details[i]))









