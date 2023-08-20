#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 10:38:17 2023

@author: Charlie Cowen-Breen

Building reports requires computation of the chance-confidence ratio [1] of every
word in the desired corpus. This script performs this computation. As the computation
is typically intensive, the script is designed to be parallelizable, and operates best
on computer clusters. That said, for users with a single CPU and no access to a GPU,
we also provide functionality (see "no-beam" flag below) which significantly accelerates
the procedure (at the cost, however, of the philological quality of the results).

RECOMMENDED SETTINGS
# If you do not have access to a GPU, run script with "no-beam" flag, which considerably decreases runtime, but also considerably decreases quality of the report generated.
# If you have access to a GPU, run script on default settings. Building a report considering 100 paragraphs takes approximately 35 hours in this case (see testing on various hardware below).
# If you have access to a computing cluster, we recommend parallelizing across multiple nodes, e.g. with each node consider 100 paragraphs each.
# Please note that out-of-memory errors may occur, unless you have 32GB RAM or more. If such errors occur, we suspect that implementation of numpy.memmap for the variable "lev_filter", instead of numpy.load, will signifcantly reduce RAM requirements.

HARDWARE CONFIGURATIONS
# In what follows, all tests were performed on either 
#  (1) A GPU-enabled computer: 2.8 GHz Intel Ice Lake with a single 1.41 GHz A100 GPU, or
#  (2) A GPU-disabled computer: 2.4 GHz Intel Broadwell.

TIME CONSIDERATIONS (with default settings)
# With completely default settings, this script takes 7 MINUTES to execute with hardware (1).
# With no-beam flag, this script takes 5 SECONDS to execute with hardware (1).
# With no-beam flag, this script takes 4 MINUTES and 26 SECONDS to execute with hardware (2).

MEMORY CONSIDERATIONS (with default settings)
# With completely default settings, this script should require 4GB RAM. If you increase the number of paragraphs considered, RAM requirements may increase.
# 100 paragraphs per report should ever not require more than 64GB RAM. When possible, we highly recommend implementing numpy.memmap to reduce RAM required, or if you have access to a cluster, parallelization across e.g. 32GB RAM nodes.

[1] Charlie Cowen-Breen, Creston Brooks, Johannes Haubold, Barbara Graziosi. 2023. Logion: Machine-Learning Based Detection and Correction of Textual Errors in Greek Philology. To appear in ACL 2023 Workshop (ALP).
"""

from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np
import time
from logion import *
import argparse
import os

""" Command line arguments """

parser = argparse.ArgumentParser(description="Build Logion Report")
parser.add_argument("-lev", "--lev", type=int, default=2, help="Tolerable Levenshtein distance (default: 2)")
parser.add_argument("-split_num", "--split_num", type=int, default=1, help="Number of model and corresponding data split to use. Useful in cases of large corpora necessitating many separate models. (default: 1)")
parser.add_argument("-start_at", "--start_at", type=int, default=0, help="Paragraph number at which to begin. Useful in cases of large corpora when parallelization is necessary. (default: 0)")
parser.add_argument("-num_pars", "--num_pars", type=int, default=1, help="Number of paragraphs to include in report (default: 1).")
parser.add_argument("-no-beam", "--no-beam", default=False, action="store_true", help="Pass this flag to prevent usage of beam search. This signficantly speeds up computation time, but also significantly decreases the quality of results. (default: false) [i.e. by default, beam search will be used.]")

args = parser.parse_args()

lev = args.lev
split_num = args.split_num
start_at = args.start_at
num_pars = args.num_pars
no_beam = args.no_beam

print("Beginning report build with parameters:")
print(f"Tolerable Levenshtein distance - {lev}")
print(f"Model/data split number - {split_num}")
print(f"Including in report - paragraphs {start_at+1} to {start_at+num_pars+1}")
print()
print("--------------------------------------------------------------------------")
print()

pars = range(start_at, start_at+num_pars)

""" Tokenizer hyper-parameters """

max_token_input = 512
start_token = 2
end_token = 3

""" Filepaths """

model_path = f'/scratch/gpfs/wcc4/Greek/models/combingpsellus/model{split_num}'
tokenizer_path = '/scratch/gpfs/wcc4/Greek/tokenizers/wordpiece-50k-vocab.txt' # By default, 50k tokenizer
levenshtein_path = f'/scratch/gpfs/wcc4/Greek/lev_maps/{lev}.npy'
data_path = '/scratch/gpfs/wcc4/Greek/data/combingpsellus/psellos_all_five.txt'
suggestion_separation_character = '*' # Used for separating suggestions for different words in output file. Spaces are not acceptable for this because some suggestions include insertions of new spaces into words.

""" Load device, tokenizer, model, and Levenshtein filter from filesystem """

start = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)
print("Using model path", model_path)
print("Using beam search." if not no_beam else "Not using beam search.")

model = BertForMaskedLM.from_pretrained(model_path).to(device)
model.eval() # We are not training BERT at all here, only evaluating it. Thus we call bert.eval().
model.to(device) # We need to bring the model back onto CPU after training in order to do inference.
sm = torch.nn.Softmax(dim=1) # In order to construct word probabilities, we will employ softmax.
torch.set_grad_enabled(False) # Since we are not training, we disable gradient calculation.

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
lev_filter = torch.tensor(np.load(levenshtein_path))
with open(data_path, 'r') as f:
  data = f.read().split('\n')
  data.pop(-1)
  f.close()

print(f'Loaded device, tokenizer, model, data, and Levenshtein filter successfully in time {time.time()-start}.')

""" Create output directory """

print()
output_dir = f"lev{lev}/{split_num}/{start_at}"
if not os.path.exists(output_dir):
  os.makedirs(output_dir)
  print(f"Output directory created at '{output_dir}'.")
else:
  print(f"Output directory already exists at '{output_dir}'.")
print()

""" Logion pipeline """

logion = Logion(model, tokenizer, lev_filter, device)

# Generate data split
data_split, _ = logion.get_fifth_and_rest(data, split_num) # List of paragraphs in appropriate data split

confidences_list = []
chances_list = []
ratios_list = []

chunk_list = []
suggestion_list = []

starttime = time.time()

for n in pars:
  if n >= len(data_split): break
  
  # Load nth paragraph and remove formatting
  chunk = logion.remove_formatting(data_split[n])

  # Skip paragraphs with lacunae
  if '[UNK]' in chunk: 
    print(f"Skipping paragraph {n+1} due to lacunae.")
    continue

  # Skip empty paragraphs
  if chunk == '' or chunk == '\n': 
    print(f"Skipping paragraph {n+1} due to empty paragraph.") 
    continue

  id_list = tokenizer(chunk, truncation=True, max_length=512)['input_ids']
  token_ids = torch.tensor(id_list).unsqueeze(0).to(device)

  # Skip paragraphs too long for model to process
  if token_ids.shape[1] > max_token_input : 
    print(f"Skipping paragraph {n+1} due to length:", token_ids.shape[1], "tokens long.")
    continue

  print(f"Beginning paragraph {n+1}.")
  
  # Compute chances ("scores") of all tokens
  score = logion.get_scores(token_ids)[1:-1]
 
  # Store tokens for refrence
  tokens = tokenizer.convert_ids_to_tokens(torch.tensor(id_list))

  # Define chances ("scores") of words by the minimum chance
  # of each of its composite tokens.
  word_score = []
  for ind in range(len(tokens[1:-1])):
    if tokens[1:-1][ind].startswith('##'): # token continues word
      if score[ind] < word_score[-1]:
        word_score[-1] = score[ind]
    else: # token is not a suffix, so it begins a new word
      word_score.append(score[ind])

  # Construct map between token id's and words
  tokens_by_words = []
  for i in range(len(tokens[1:-1])):
    if not tokens[1:-1][i].startswith('##'):
      tokens_by_words.append([token_ids[0,1:-1][i].item()])
    else:
      tokens_by_words[-1] = tokens_by_words[-1] + [token_ids[0,1:-1][i].item()]

  confidences_list.append([])
  chances_list.append([])
  ratios_list.append([])

  chunk_list.append([])
  suggestion_list.append([])

  # Iterate through all words in paragraph, computing confidences individually
  for word_ind in range(len(word_score)):

    # Separate tokens into those before, during, and after the relevent word
    pre_tokens = sum(tokens_by_words[:word_ind], [])
    transmitted_tokens = tokens_by_words[word_ind]
    post_tokens = sum(tokens_by_words[word_ind+1:], [])

    # Compute text corresponding to token lists defined above, useful for Levenshtein distance.
    pre_text =  logion.display_sentence(tokenizer.convert_ids_to_tokens(pre_tokens))
    transmitted_text = logion.display_sentence(tokenizer.convert_ids_to_tokens(transmitted_tokens))
    post_text = logion.display_sentence(tokenizer.convert_ids_to_tokens(post_tokens))

    # Construct input tensor containing start token, pre-word tokens, a mask token for the word, post-word tokens, and end token.
    masked_token_ids = torch.unsqueeze(torch.tensor([start_token] + pre_tokens + [tokenizer.mask_token_id] + post_tokens + [end_token]), 0).to(device)
    suggestions = logion.fill_the_blank_given_transmitted(pre_text, post_text, transmitted_text, transmitted_tokens, masked_token_ids, max_masks=5, depth=20, breadth=20, max_lev=lev, min_lev=lev, no_beam=no_beam)

    max_confidence = suggestions[0][1]/word_score[word_ind]    
    suggested_word = suggestions[0][0]

    chances_list[-1].append(word_score[word_ind])
    confidences_list[-1].append(suggestions[0][1])
    ratios_list[-1].append(suggestions[0][1]/word_score[word_ind])

    chunk_list[-1].append(transmitted_text)
    suggestion_list[-1].append(suggested_word)


print(f"Scores computed in {time.time() - starttime} seconds, and saved in {output_dir}.")

with open(output_dir + '/' + 'wordconfidences.txt', 'w') as out:
  for confidences in confidences_list:
    out.write('[' + ','.join([str(confidence) for confidence in confidences]) + ']' + '\n')

with open(output_dir + '/' + 'wordchances.txt', 'w') as out:
  for chances in chances_list:
    out.write('[' + ','.join([str(chance) for chance in chances]) + ']' + '\n')

with open(output_dir + '/' + 'wordratios.txt', 'w') as out:
  for ratios in ratios_list:
    out.write('[' + ','.join([str(ratio) for ratio in ratios]) + ']' + '\n')

with open(output_dir + '/' + 'transmittions.txt', 'w') as out:
  for transmittions in chunk_list:
    out.write(' '.join(transmittions) + '\n')

with open(output_dir + '/' + 'suggestions.txt', 'w') as out:
  for suggestions in suggestion_list:
    out.write(suggestion_separation_character.join(suggestions) + '\n')
