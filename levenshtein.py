from polyleven import levenshtein
from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np
import time

lev = 1

tokenizer = BertTokenizer.from_pretrained('/scratch/gpfs/wcc4/Greek/tokenizers/wordpiece-50k-vocab.txt')
num_tokens = len(tokenizer)

start = time.time()

tokens = []
for i in range(num_tokens):
  x = tokenizer.convert_ids_to_tokens(i)
  if x.startswith('##'): x = x[2:]
  tokens.append(x)

print(f"Done with token strings in {time.time()-start} seconds.")

lev_map = torch.empty((num_tokens, num_tokens), dtype=torch.int64)
for i in range(num_tokens):
  if i%(num_tokens//10) == 0: print("finished 10%", time.time()-start)
  for j in range(num_tokens):
    x = tokens[i]
    y = tokens[j]
    lev_map[i,j] = levenshtein(x, y, lev)

filter = (lev_map <= lev) & (lev_map > 0)
np.save('filter.npy', filter)

print("Saved Levenshtein filter in filter.npy")
