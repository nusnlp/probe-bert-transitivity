"""Get statistics for the transitivity analysis"""

import argparse
import os
import numpy as np
import copy
from nltk.corpus import wordnet as wn


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-dir', required=True, help='Directory to test result files.')
parser.add_argument('--probing_mode', required=True, choices=['full'], help='Probing in the full or fewshot data setting.')




def get_overall_stats(probing_dir, probing_clf_name):
	for repeat in [0,1,2]:
		fname = 'detailed_result_knn_wholeBert_multihop_concat_kOf1_train-1998-{}_test-3996.txt'.format(repeat)
		filename = os.path.join(probing_dir, fname)
		a = open(filename).readlines()
		a = a[24:24+1998]

		test_pos_pairs = []
		test_pos_pairs_lemmas = []
		for x in a:
			x=x.strip().split('\t')
			label = x[1]
			pos_tag = wn.lemma(eval(x[2])[0][7:-2]).synset().pos()
			if x[-1] != '_' and pos_tag == 'n':
				test_pos_pairs_lemmas.append(eval(x[2]))
				test_pos_pairs.append(x)

		print(len(test_pos_pairs))
		assert len(test_pos_pairs) == 1998

		hop_ab_sum = 0
		hop_bc_sum = 0
		trans_relns = []
		trans_relns_inds = []
		for i in range(0,len(test_pos_pairs_lemmas)-2,3):
			j = i+1
			end_pair = [test_pos_pairs_lemmas[i][0], test_pos_pairs_lemmas[j][1]]
			end_pair_idx = i+2 #test_pos_pairs_lemmas.index(end_pair)
			# assert end_pair == test_pos_pairs_lemmas[end_pair_idx]

			ab_hop = eval(test_pos_pairs[i][-1])['hops']
			hop_ab_sum += ab_hop
			bc_hop = eval(test_pos_pairs[j][-1])['hops']
			hop_bc_sum += bc_hop
			print(i,j,end_pair_idx)
			print('hops: ', eval(test_pos_pairs[i][-1])['hops'], eval(test_pos_pairs[j][-1])['hops'], eval(test_pos_pairs[end_pair_idx][-1])['hops'])
			trans_relns.append([test_pos_pairs_lemmas[i], test_pos_pairs_lemmas[j], end_pair])
			trans_relns_inds.append([i, j, end_pair_idx])

		print('avg. ab hops ', hop_ab_sum/len(trans_relns_inds))
		print('avg. bc hops ', hop_bc_sum/len(trans_relns_inds))

		ct_a, ct_b, ct_c = 0,0,0
		ct_conditional = {'111':0, '110':0}
		for tt in trans_relns_inds:
			pred_a = int(eval(test_pos_pairs[tt[0]][0]))
			pred_b = int(eval(test_pos_pairs[tt[1]][0]))
			pred_c = int(eval(test_pos_pairs[tt[2]][0]))
			ct_a += pred_a
			ct_b += pred_b
			ct_c += pred_c
		
			if pred_a == 1 and pred_b == 1 and pred_c == 1:
				ct_conditional['111'] += 1
			if pred_a == 1 and pred_b == 1 and pred_c == 0:
				ct_conditional['110'] += 1

		len_cond = ct_conditional['111']+ct_conditional['110']
		cond_acc_ac = 0 if len_cond == 0 else ct_conditional['111']/len_cond

		len_trans_relns_inds = len(trans_relns_inds)
		# print('len_trans_relns_inds', len_trans_relns_inds)
		# print('acc. ab', ct_a/len_trans_relns_inds)
		# print('acc. bc', ct_b/len_trans_relns_inds)
		# print('acc. ac', ct_c/len_trans_relns_inds)
		# print('conditional prob ac on ab and bc', len_cond, cond_acc_ac)

		with open(os.path.join(probing_dir, 'transitivity_analysis_train-1998-{}_test-3996.txt'.format(repeat)), 'w') as f:
			f.write('Number of transitive relation triples\t{}'.format(str(len_trans_relns_inds)))
			f.write('Acc./F1 AB\t{}'.format(str(ct_a/len_trans_relns_inds)))
			f.write('Acc./F1 BC\t{}'.format(str(ct_b/len_trans_relns_inds)))
			f.write('Acc./F1 AC\t{}'.format(str(ct_c/len_trans_relns_inds)))
			f.write('Conditional prob. AC on AB and BC\t{}'.format(str(cond_acc_ac)))



if __name__ == '__main__':

	args = parser.parse_args()
	get_overall_stats(args.dir, 'knn')


