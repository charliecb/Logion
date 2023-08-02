from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np
import time
from polyleven import levenshtein


""" Report building functionality """

class Logion:

  def __init__(self, Model, Tokenizer, Levenshtein_Filter, Device):
    self.Model = Model
    self.Tokenizer = Tokenizer
    self.lev_filter = Levenshtein_Filter
    self.device = Device
    self.sm = torch.nn.Softmax(dim=1) # In order to construct word probabilities, we will employ softmax.
    torch.set_grad_enabled(False) # Since we are not training, we disable gradient calculation.
    
  def get_scores(self, token_ids, relevant_token_ids = None): # If no relevant_token_ids are specified, all tokens are considered relevant.
    scores = []
    num_tokens = token_ids.shape[1]
    for index_of_id_to_mask in range(num_tokens):
      start = time.time()

      # Remember the actual token
      underlying_token_id = token_ids[0, index_of_id_to_mask].item()

      # If either relevant_token_ids aren't specified, or if the underlying token id is among the relevant ones, mask it.
      if (not relevant_token_ids) or (underlying_token_id in relevant_token_ids):
        # Now mask it
        token_ids[0, index_of_id_to_mask] = self.Tokenizer.mask_token_id

        # Now pass through model

        # Pass input through BERT and get probabilities for which words lie under mask:
        logits = self.Model(token_ids).logits
        #print(logits.shape)
        mask_logits = logits[:,index_of_id_to_mask,:]
        #print(mask_logits.shape)
        probabilities = self.sm(mask_logits).flatten()

        scores.append(probabilities[underlying_token_id].item())

        # Set token back to its original
        token_ids[0, index_of_id_to_mask] = underlying_token_id

      else: # Then there must be some relevant_token_ids, but this isn't one of them
        scores.append(-1) # The trivial gray case

      #print(f"Done with token number {index_of_id_to_mask+1} in {time.time() - start} seconds.")
    return scores

  def get_probabilities(self, masked_text_ids):
    mask_positions = (masked_text_ids.squeeze() == self.Tokenizer.mask_token_id).nonzero().flatten().tolist() # Taking note of the positions of the masks in the text
    logits = self.Model(masked_text_ids).logits.squeeze(0)
    mask_logits = logits[mask_positions]
    probabilities = self.sm(mask_logits) # Convert logits to probabilities and print them:
    return probabilities

  # Get top 5 suggestions for each masked position:
  def argkmax(self, array, k, dim=0, prefix=None): # Return indices of the 1st through kth largest values of an array
    if not prefix:
      indices = []
      for i in range(1,k+1):
        indices.append(torch.kthvalue(-array, i, dim=dim).indices.cpu().numpy().tolist())
    else:
      indices = []
      i = 1
      while len(indices) < k:
        index = torch.kthvalue(-array, i, dim=dim).indices.cpu().numpy().tolist()
        if self.Tokenizer.convert_ids_to_tokens(index)[0].startswith(prefix): indices.append(index)
        i += 1
      
    return torch.tensor(indices)

  def display_sentence(self, toks):
    s = ''
    first_tok = True
    for tok in toks:
      if tok.startswith('##'): tok = tok[2:]  # remove suffix hashtags
      elif tok in ['´',',','.',';']: pass
      elif first_tok: first_tok = False
      else: tok = " " + tok

      s += tok
    return s

  def remove_formatting(self, text):
    return self.display_sentence(self.Tokenizer.convert_ids_to_tokens(self.Tokenizer.encode(text)[1:-1]))

  def get_token_index_containing_character_index(self, tokens, char_index):
    ret = 0
    while len(self.display_sentence(tokens[:ret+1])) < 1 + char_index:
      ret += 1
    return ret

  def checktokens(self, s):
    return self.Tokenizer.convert_ids_to_tokens(self.Tokenizer.encode(s))[1:-1]

  # Get top k suggestions for each masked position:
  def argkmaxbeam(self, array, k, prefix='', dim=0): # Return indices of the 1st through kth largest values of an array, given prefix
    indices = []
    new_prefixes = []
    added = 0
    ind = 1
    while added < k:
      if ind > len(array[0]):
        break
      val = torch.kthvalue(-array, ind, dim=dim).indices.cpu().numpy().tolist()
      if prefix != '':
        cur_tok = self.Tokenizer.convert_ids_to_tokens(val[0]).replace('##', '')
        trunc_prefix = prefix[:min(len(prefix), len(cur_tok))]
        if not cur_tok.startswith(trunc_prefix):
          ind += 1
          continue
      else:
        cur_tok = ''
      indices.append(val)
      if len(cur_tok) >= len(prefix):
        new_prefixes.append('')
      else:
        new_prefixes.append(prefix[len(cur_tok):])
      ind += 1
      added += 1
    return torch.tensor(indices), new_prefixes

  # gets n predictions / probabilities for a single masked token , by default, the first masked token
  def get_n_preds(self, token_ids, n, prefix, masked_ind, fill_inds, cur_prob=1):
    mask_positions = (token_ids.squeeze() == self.Tokenizer.mask_token_id).nonzero().flatten().tolist()
    for i in range(len(fill_inds)):
      token_ids.squeeze()[mask_positions[i]] = fill_inds[i]

    #print(len(mask_positions), len(fill_inds))
    model_id = min(len(mask_positions) - len(fill_inds) - 1, 4)
    #print(model_id)
    logits = self.Model(token_ids).logits.squeeze(0)
    mask_logits = logits[[[masked_ind]]]
    probabilities = self.sm(mask_logits)
    arg1, prefixes = self.argkmaxbeam(probabilities, n, prefix, dim=1)
    suggestion_ids = arg1.squeeze().tolist()
    n_probs = probabilities.squeeze()[suggestion_ids]
    n_probs = torch.mul(n_probs, cur_prob).tolist()
    new_fill_inds = [fill_inds + [i] for i in suggestion_ids]
    return tuple(zip(new_fill_inds, n_probs, prefixes)) 

  def beam_search(self, token_ids, beam_size, prefix='', breadth=100):
    mask_positions = (token_ids.detach().clone().squeeze() == self.Tokenizer.mask_token_id).nonzero().flatten().tolist()
    #print(len(mask_positions))
    num_masked = len(mask_positions)
    cur_preds = self.get_n_preds(token_ids.detach().clone(), beam_size, prefix, mask_positions[0], [])
    #for c in range(len(cur_preds)):
      #print(tokenizer.convert_ids_to_tokens(cur_preds[c][0][0]))

    for i in range(num_masked - 1):
      #print(i)
      candidates = []
      for j in range(len(cur_preds)):
        candidates += self.get_n_preds(token_ids.detach().clone(), breadth, cur_preds[j][2], mask_positions[i + 1], cur_preds[j][0], cur_preds[j][1])
      candidates.sort(key=lambda k:k[1],reverse=True)
      if i != num_masked - 2:
        cur_preds = candidates[:beam_size]
      else:
        cur_preds = candidates[:breadth]
    return cur_preds


  def suggest_filtered(self, tokens, ground_token_id, filter):
    probs = self.get_probabilities(tokens).cpu().squeeze()
    #print("exact probs")
    #print(sorted([(i, probs[i]) for i in range(len(probs))], key=lambda x: x[1])[::-1][:100])
    filtered_probs = probs * filter[ground_token_id]
    #print(filtered_probs[tokenizer.convert_tokens_to_ids('τη')])
    suggestion = self.argkmax(filtered_probs, 1)
    #print(tokenizer.convert_ids_to_tokens(argkmax(filtered_probs, 10)), [probs[k] for k in argkmax(filtered_probs,10)])
    return suggestion, probs[suggestion]


  def fill_the_blank_given_transmitted(self, pre_text, post_text, transmitted_text, transmitted_tokens, tokens, max_masks=3, depth=20, breadth=20, max_lev=1, min_lev=0):
    filtered_suggestions = {'?': 0.0}
    for num_masks in range(1,max_masks+1):
      if num_masks == 1 and len(transmitted_tokens) == 1:
        #print(tokens)
        sug, prob = self.suggest_filtered(tokens, transmitted_tokens[0], self.lev_filter)
        #print(sug)
        word = self.display_word(self.Tokenizer.convert_ids_to_tokens(sug))
        #print(word)
        filtered_suggestions[word] = prob
        continue

        #print("Num masks and len transmitted is one")
      text = pre_text + f"".join([f"{self.Tokenizer.mask_token}"]*num_masks) + post_text
      #print(text)
      tokens = self.Tokenizer.encode(text, return_tensors='pt').to(self.device)
      #print(tokens)
      sugs = self.beam_search(tokens, depth if num_masks>1 or len(transmitted_tokens)>1 else len(self.Tokenizer), breadth=breadth if num_masks>1 or len(transmitted_tokens)>1 else len(self.Tokenizer))
      #print([sugs[k] for k in range)
      #print([(s[0], s[1]) for s in sugs[:100]])

      for suggestion, probability, _ in sugs:
        #print(suggestion)
        converted = self.Tokenizer.convert_ids_to_tokens(suggestion)
        word = self.display_word(converted)
        #print(probability)
        #print(suggestion[0], transmitted_tokens[0])
        d = levenshtein(word, transmitted_text, max_lev) #lev_dist(display_word(converted), transmitted_text)
        if d > max_lev or d < min_lev: continue
        #print(f"{probability:.1%} - {display_word(converted)} - {transmitted_text}")
        if (word not in filtered_suggestions) or (word in filtered_suggestions and filtered_suggestions[word] < probability):
          filtered_suggestions[word] = probability
          break
          #print(display_word(converted))
        #print(filtered_suggestions)
      
    sorted_filtered_suggestions = sorted(filtered_suggestions.items(), key = lambda x: x[1])[::-1]
    #print(sorted_filtered_suggestions[:1])
    return sorted_filtered_suggestions[:1]


  def display_word(self, toks):
    s = ''
    first_tok = True
    for tok in toks:
      is_suffix = tok.startswith('##')
      if is_suffix: tok = '' + tok[2:]  # remove suffix hashtags
      elif not first_tok: s += ' '
      
      s += tok
      
      first_tok = False
    return s

  def get_fifth_and_rest(self, lst, n):
      # Calculate the length of one fifth of the list
      one_fifth = len(lst) // 5

      # Calculate the remainder
      remainder = len(lst) % 5

      # Calculate start and end indices for the nth fifth
      start = (n - 1) * one_fifth + min(n - 1, remainder)
      if n <= remainder:
          end = start + one_fifth + 1
      else:
          end = start + one_fifth

      # Get the nth fifth of the list
      nth_fifth = lst[start:end]

      # Get the rest of the list
      rest = lst[:start] + lst[end:]

      return nth_fifth, rest
