# Does BERT Know that the IS-A Relation Is Transitive?
-------------------------------------------

This repository contains the datasets, code, and scripts to conduct the analysis in [Lin and Ng (2022)](#reference).

## Reference
Ruixi Lin and Hwee Tou Ng (2022). 
[Does BERT Know that the IS-A Relation Is Transitive?](https://TBD).  Proceedings of the ACL 2022. 

Please cite: 
```
@inproceedings{lin2022isa,
    title = "Does {BERT} Know that the {IS}-A Relation Is Transitive?",
    author = "Lin, Ruixi  and Ng, Hwee Tou",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    year = "2022",
    url = "https://aclanthology.org/2022.acl-short.11",
    pages = "94--99",
}

```

**Table of contents**

[Prerequisites](#prerequisites)

[Example](#example)

[License](#license)


## Prerequisites

```
scikit-learn==0.24.1
nltk==3.5
```


## Example
1. Put the data folder under the project directory which contains the source code and dependencies. Run all probing experiments from the project directory and perform analysis using the following command. BERT embeddings for pairs in all datasets are pre-stored as vector files in the data folder for direct usages.

```
bash run_exp.sh
```

2. Results are saved in files, which can be found under
	./data/fullprobe/full-test/test-3996

3. The original datasets used in experiments are located under
	./data/raw_datasets/full

	Each line is an entry for a pair, consisting of context-1, context-2, word-1, word-2, sense-1, sense-2, details (the path that a IS-A pair is sampled from, offsets from leaf, number of hops of the pair, starting depth is not used), label. Label 1 denotes the IS-A relation, and 0 denotes not IS-A relation.

	An example data instance is as follows.
	we took a turn in the park	he enjoyed selling but he hated the travel	turn	travel	Lemma('turn.n.12.turn')	Lemma('travel.n.01.travel')	{'path': "[Synset('turn.n.12'), Synset('walk.n.04'), Synset('travel.n.01'), Synset('motion.n.06'), Synset('change.n.03'), Synset('action.n.01'), Synset('act.n.02'), Synset('event.n.01'), Synset('psychological_feature.n.01'), Synset('abstraction.n.06'), Synset('entity.n.01')]", 'offset': (0, 2), 'hops': 2, 'starting_depth_in_path': 0}	1


## License
The source code and models in this repository are licensed under the GNU General Public License v3.0 (see [LICENSE](LICENSE)). For commercial use of this code and models, separate commercial licensing is also available. Please contact Ruixi Lin (ruixi@u.nus.edu) and Prof. Hwee Tou Ng (nght@comp.nus.edu.sg).


