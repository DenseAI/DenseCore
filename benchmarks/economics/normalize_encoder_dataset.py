#!/usr/bin/env python3
"""Normalize pinned official dataset Parquet to source-preserving requests.

Parquet conversion is an optional benchmark preparation dependency, e.g.
uv run --with pyarrow python benchmarks/economics/normalize_encoder_dataset.py ...
No model predictions or tokenizer receipts are invented here.
"""
import argparse
import hashlib
import json
from pathlib import Path


def normalize(rows, dataset, split):
    for index, row in enumerate(rows):
        if dataset == 'fancyzhx/ag_news':
            options = ['World', 'Sports', 'Business', 'Sci/Tech']
            label = row['label']
            if type(label) is not int or not 0 <= label < 4:
                raise ValueError('invalid AG News label')
            source_id = f'ag_news/{split}/{index}'
            yield {'id': source_id, 'source_id': source_id, 'gold_label': options[label],
                   'request': {'state': row['text'], 'questions': {'topic': {'type': 'choice',
                    'instructions': 'Classify the news article topic.', 'criteria': options}}}}
        elif dataset == 'LocalLLaMA/typed-decisions':
            questions = json.loads(row['questions']) if isinstance(row['questions'], str) else row['questions']
            gold = json.loads(row['gold']) if isinstance(row['gold'], str) else row['gold']
            # Preserve the upstream serialized state exactly. Retain only genuine
            # four-option choice questions; no conversion of score/noul semantics.
            source_id = f'typed-decisions/{split}/{row["id"]}'
            for name, question in questions.items():
                if question['type'] != 'choice' or len(question.get('criteria', {})) != 4:
                    continue
                yield {'id': source_id + '/' + name, 'source_id': source_id,
                       'gold_label': gold[name]['label'], 'source_gold_probabilities': gold[name]['probabilities'],
                       'request': {'state': row['state'], 'questions': {name: question}}}
        else:
            raise ValueError('unsupported dataset')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--dataset',choices=['fancyzhx/ag_news','LocalLLaMA/typed-decisions'],required=True)
    p.add_argument('--split',choices=['train','test'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import pyarrow.parquet as pq
    rows=list(normalize(pq.read_table(a.source).to_pylist(),a.dataset,a.split))
    a.output.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    print(json.dumps({'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
                      'normalized_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'examples':len(rows)}))

if __name__=='__main__': main()
