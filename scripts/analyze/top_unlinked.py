#!/opt/hf-env/bin/python

import json
from pathlib import Path
from collections import Counter

linked = Counter()
unlinked = Counter()

for f in Path('outputs/cancer-all-ft/el').glob('*.jsonl'):
    for line in open(f):
        for ent in json.loads(line).get('entities', []):
            if ent.get('entity', '').lower() == 'chemical':
                if ent.get('linking'):
                    linked[ent['text'].lower()] += 1
                else:
                    unlinked[ent['text'].lower()] += 1

print(f'Linked: {sum(linked.values()):,} ({len(linked)} unique)')
print(f'Unlinked: {sum(unlinked.values()):,} ({len(unlinked)} unique)')
print(f'\\nTop 250 unlinked:')
for term, count in unlinked.most_common(250):
    print(f'  {count:>6}  {term}')
