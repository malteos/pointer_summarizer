import collections
import json
import os
from typing import List

from nltk import word_tokenize
from tqdm.auto import tqdm

"""

- train/test split
- "The context C refers to the sentences surrounding the target citation text in A (src paper) and it is provided to distinguish different men-tions of B (trg paper) in different positions of A."

source_context = text_before_explicit_citation + text_after_explicit_citation  # TODO: Joined with white space?
target = target_abstract
gold reference  = explicit_citation

"The input context of citing paper is seen as a sequence of words"

"the input cited paper’s abstract is seen as a sequence of words"


- Compare with "Table 5: Comparison results on Explicit dataset ROUGE-1" -> F1



PTGEN has 256-dimensional hidden states and 128-dimensional
word embeddings. The vocabulary size is set to
50k.

beam size 4


tokenizer? original uses: Stanford CoreNLP edu.stanford.nlp.process.PTBTokenizer -> use stanza instead


pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz

"""

fp = '/Volumes/data/repo/explainable-document-similarity/data/xing2020/citation.json'
total_lines = 86052

VOCAB_SIZE = 50_000
vocab_counter = collections.Counter()


train_cits = []
test_cits = []


def tokenize(text) -> List[str]:
    # doc = nlp(text)
    # return [t.text for t in doc]

    return word_tokenize(text)


for i, line in tqdm(enumerate(open(fp)), total=total_lines):
    cit = json.loads(line)

    citing_context = (cit['text_before_explicit_citation'] + ' ' + cit['text_after_explicit_citation']).lower()
    cited_abstract = cit['tgt_abstract'].lower()
    citation_text = cit['explicit_citation'].lower()

    # tokenize
    citing_context_tokens = tokenize(citing_context)
    cited_abstract_tokens = tokenize(cited_abstract)
    citation_text_tokens = tokenize(citation_text)

    # update vocab
    vocab_counter.update(citing_context_tokens + cited_abstract_tokens + citation_text_tokens)

    # save to dict
    sample = dict(
        src_paper_id=cit['src_paper_id'],
        tgt_paper_id=cit['tgt_paper_id'],
        citing_context=citing_context,
        cited_abstract=cited_abstract,
        citation_text=citation_text,
        citing_context_tokens=citing_context_tokens,
        cited_abstract_tokens=cited_abstract_tokens,
        citation_text_tokens=citation_text_tokens,
    )

    if cit['train_or_test'] == 'train':
        train_cits.append(sample)
    else:
        test_cits.append(sample)


print(f'train: {len(train_cits):,}; test: {len(test_cits):,}; ({len(train_cits)/(len(train_cits)+len(test_cits))}%:)')

output_dir = './data/xing'

# train and test files
with open(os.path.join(output_dir, 'train.jsonl'), 'w') as f:
    for sample in train_cits:
        f.write(json.dumps(sample) + '\n')

with open(os.path.join(output_dir, 'test.jsonl'), 'w') as f:
    for sample in test_cits:
        f.write(json.dumps(sample) + '\n')

print("Writing vocab file...")
with open(os.path.join(output_dir, "vocab"), 'w') as writer:
    for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
print("Finished writing vocab file")
