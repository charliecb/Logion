from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np
import time
from polyleven import levenshtein
from logion import *


""" Model hyper-parameters """

split_num = 1
pars = range(0, 100)
lev = 1

""" Tokenizer hyper-parameters """

max_token_input = 512
start_token = 2
end_token = 3

""" Filepaths """

model_path = f'/scratch/gpfs/wcc4/Greek/models/combingpsellus/model{split_num}'
tokenizer_path = '/scratch/gpfs/wcc4/Greek/tokenizers/wordpiece-50k-vocab.txt' # By default, 50k tokenizer
levenshtein_path = 'filter.npy'
data_path = '/scratch/gpfs/wcc4/Greek/data/combingpsellus/psellos_all_five.txt'

""" Load device, tokenizer, model, and Levenshtein filter from filesystem """

start = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)
print("Using model path", model_path)

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


""" Logion pipeline """

logion = Logion(model, tokenizer, lev_filter, device)

# Generate data split
data_split, _ = logion.get_fifth_and_rest(data, split_num) # List of paragraphs in appropriate data split

confidences_list = []
chances_list = []
ratios_list = []

chunk_list = []
suggestion_list = []

for n in pars:
  
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

  starttime = time.time()
  
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

    # Separate tokens into those before, during, and after the relevent word
    pre_tokens = sum(tokens_by_words[:word_ind], [])
    transmitted_tokens = tokens_by_words[word_ind]
    post_tokens = sum(tokens_by_words[word_ind+1:], [])

    # Compute text corresponding to token lists defined above, useful for Levenshtein distance.
    pre_text =  logion.display_sentence(tokenizer.convert_ids_to_tokens(pre_tokens))
    transmitted_text = logion.display_sentence(tokenizer.convert_ids_to_tokens(transmitted_tokens))
    post_text = logion.display_sentence(tokenizer.convert_ids_to_tokens(post_tokens))

    # Construct input tensor containing start token, pre-word tokens, a mask token for the word, post-word tokens, and end token.
    masked_token_ids = torch.unsqueeze(torch.tensor([start_token] + pre_tokens + [tokenizer.mask_token_id] + post_tokens + [end_token]), 0).to(device)
    suggestions = logion.fill_the_blank_given_transmitted(pre_text, post_text, transmitted_text, transmitted_tokens, masked_token_ids, max_masks=5, depth=20, breadth=20, max_lev=lev, min_lev=lev)

    max_confidence = suggestions[0][1]/word_score[word_ind]    
    suggested_word = suggestions[0][0]

    chances_list[-1].append(word_score[word_ind])
    confidences_list[-1].append(suggestions[0][1])
    ratios_list[-1].append(suggestions[0][1]/word_score[word_ind])

    chunk_list[-1].append(transmitted_text)
    suggestion_list[-1].append(suggested_word)


print(f"Scores computed in {time.time() - starttime} seconds, and saved in out.txt.")

with open('wordconfidences.txt', 'w') as out:
  for confidences in confidences_list:
    out.write('[' + ','.join([str(confidence) for confidence in confidences]) + ']' + '\n')

with open('wordchances.txt', 'w') as out:
  for chances in chances_list:
    out.write('[' + ','.join([str(chance) for chance in chances]) + ']' + '\n')

with open('wordratios.txt', 'w') as out:
  for ratios in ratios_list:
    out.write('[' + ','.join([str(ratio) for ratio in ratios]) + ']' + '\n')

with open('transmittions.txt', 'w') as out:
  for transmittions in chunk_list:
    out.write(' '.join(transmittions) + '\n')

with open('suggestions.txt', 'w') as out:
  for suggestions in suggestion_list:
    out.write('*'.join(suggestions) + '\n')
