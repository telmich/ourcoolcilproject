.PHONY: help clean

##### Make this makefile awesome

help: # https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html <3
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-10s %s\n", $$1, $$2}'

##### Exercise targets

solution: cooc  ## Solve the exercise :-)
	@echo 'TODO :-)'
	@echo "Last finished step: $<"

vocab: data/vocab.pkl  ## [step 1, part 1] Construct the vocabulary: list of words appearing at least 5 times

cooc: data/cooc.pkl  ## [step 1, part 2] Compute the co-occurence matrix

glove: data/glove_vectors.pkl ## [step 1, part 3] Compute the GloVe word embeddings (see embeddings.py)



##### Targets that actually do stuff

# Reference for magic $whatever: https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html

data/vocab.txt: data/train_pos.txt data/train_neg.txt
	@echo 'Running ugly conglomerate of TA-provided scripts...'
	cat $+ | sed "s/ /\n/g" | grep -v '^\s*$$' | sort | uniq -c | sed "s/^\s\+//g" | sed "s/\s\+/ /g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > $@

data/vocab.pkl: data/vocab.txt
	./pickle_vocab.py $@ $+

data/cooc.pkl: data/vocab.pkl data/train_pos.txt data/train_neg.txt
	./cooc.py $@ $+

data/glove_vectors.pkl: data/cooc.pkl
	./embeddings.py $@ $+


##### Helper targets

clean:  ## Removes files listed in ./.gitignore
	rm -f $$(cat ./.gitignore | grep -v '^#')
