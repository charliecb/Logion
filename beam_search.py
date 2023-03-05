from transformers import BertTokenizer, BertForMaskedLM
import torch

# Filepath to corpus, in (Greek) unicode text, separated by lines into <=512 token paragraphs.
data_filepath = "path_to_corpus"
model_filepaths = [
    "/models/base",
    "/models/expert2span",
    "/models/expert3span",
    "/models/expert4span",
    "/models/expert5span"
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")

# Download Greek tokenizer from Lefever et al., which is borrowed from
#  the modern Greek tokenizer from Koutsikakis et al. (https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1)
tokenizer = BertTokenizer.from_pretrained('pranaydeeps/Ancient-Greek-BERT')

# We trained several models, each of which specialize in infilling a
#  different number of missing tokens (for more than 5 missing tokens,
#  we default to the model specializing in 5 missing tokens, as such
#  occurences are rare).
models = [
    BertForMaskedLM.from_pretrained(filepath).to(device)
    for filepath in model_filepaths
]

for model in models: model.eval()

sm = torch.nn.Softmax(dim=1) # In order to construct word probabilities, we will employ softmax.
torch.set_grad_enabled(False) # Since we are not training, we disable gradient calculation.

# Get top k suggestions for each masked position:
def argkmax(array, k, prefix='', dim=0): # Return indices of the 1st through kth largest values of an array, given prefix
  indices = []
  new_prefixes = []
  added = 0
  ind = 1
  while added < k:
    if ind > len(array[0]):
      break
    val = torch.kthvalue(-array, ind, dim=dim).indices.cpu().numpy().tolist()
    if prefix != '':
      cur_tok = tokenizer.convert_ids_to_tokens(val[0]).replace('##', '')
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
def get_n_preds(token_ids, n, prefix, masked_ind, fill_inds, cur_prob=1):
  mask_positions = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().flatten().tolist()
  for i in range(len(fill_inds)):
    token_ids.squeeze()[mask_positions[i]] = fill_inds[i]

  #print(len(mask_positions), len(fill_inds))
  model_id = min(len(mask_positions) - len(fill_inds) - 1, 4)
  #print(model_id)
  model = models[model_id]
  logits = model(token_ids).logits.squeeze(0)
  mask_logits = logits[[[masked_ind]]]
  probabilities = sm(mask_logits)
  arg1, prefixes = argkmax(probabilities, n, prefix, dim=1)
  suggestion_ids = arg1.squeeze().tolist()
  n_probs = probabilities.squeeze()[suggestion_ids]
  n_probs = torch.mul(n_probs, cur_prob).tolist()
  new_fill_inds = [fill_inds + [i] for i in suggestion_ids]
  return tuple(zip(new_fill_inds, n_probs, prefixes)) 

def beam_search(token_ids, beam_size, prefix='', breadth=100):
  mask_positions = (token_ids.detach().clone().squeeze() == tokenizer.mask_token_id).nonzero().flatten().tolist()
  #print(len(mask_positions))
  num_masked = len(mask_positions)
  cur_preds = get_n_preds(token_ids.detach().clone(), beam_size, prefix, mask_positions[0], [])
  #for c in range(len(cur_preds)):
    #print(tokenizer.convert_ids_to_tokens(cur_preds[c][0][0]))

  for i in range(num_masked - 1):
    #print(i)
    candidates = []
    for j in range(len(cur_preds)):
      candidates += get_n_preds(token_ids.detach().clone(), breadth, cur_preds[j][2], mask_positions[i + 1], cur_preds[j][0], cur_preds[j][1])
    candidates.sort(key=lambda k:k[1],reverse=True)
    if i != num_masked - 2:
      cur_preds = candidates[:beam_size]
    else:
      cur_preds = candidates[:breadth]
  return cur_preds

# Get top 5 suggestions for each masked position:
def argkmax_right(array, k, suffix='', dim=0): # Return indices of the 1st through kth largest values of an array
  indices = []
  new_suffixes = []
  added = 0
  ind = 1
  while added < k:
    if ind > len(array[0]):
      break
    val = torch.kthvalue(-array, ind, dim=dim).indices.cpu().numpy().tolist()
    if suffix != '':
      cur_tok = tokenizer.convert_ids_to_tokens(val[0]).replace('##', '')
      trunc_suffix = suffix[len(suffix) - min(len(suffix), len(cur_tok)):]
      if not cur_tok.endswith(trunc_suffix):
        ind += 1
        continue
    else:
      cur_tok = ''
    indices.append(val)
    if len(cur_tok) >= len(suffix):
      new_suffixes.append('')
    else:
      new_suffixes.append(suffix[:len(suffix) - len(cur_tok)])
    ind += 1
    added += 1
  return torch.tensor(indices), new_suffixes

# gets n predictions / probabilities for a single masked token , by default, the first masked token
def get_n_preds_right(token_ids, n, suffix, masked_ind, fill_inds, cur_prob=1):
  mask_positions = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().flatten().tolist()
  # fill in the current guessed tokens
  for i in range(len(fill_inds)):
    token_ids.squeeze()[mask_positions[len(mask_positions) - i - 1]] = fill_inds[i]
  #print(len(mask_positions), len(fill_inds))
  model_id = min(len(mask_positions) - len(fill_inds) - 1, 4)
  #print(model_id)
  model = models[model_id]
  logits = model(token_ids).logits.squeeze(0)
  mask_logits = logits[[[masked_ind]]]
  probabilities = sm(mask_logits)
  arg1, suffixes = argkmax_right(probabilities, n, suffix, dim=1)
  suggestion_ids = arg1.squeeze().tolist()
  n_probs = probabilities.squeeze()[suggestion_ids]
  n_probs = torch.mul(n_probs, cur_prob).tolist()
  new_fill_inds = [fill_inds + [i] for i in suggestion_ids]
  return tuple(zip(new_fill_inds, n_probs, suffixes)) 

def beam_search_right(token_ids, beam_size, suffix='', breadth=100):
  mask_positions = (token_ids.detach().clone().squeeze() == tokenizer.mask_token_id).nonzero().flatten().tolist()
  num_masked = len(mask_positions)
  cur_preds = get_n_preds_right(token_ids.detach().clone(), beam_size, suffix, mask_positions[-1], [])
  #for c in range(len(cur_preds)):
    #print(tokenizer.convert_ids_to_tokens(cur_preds[c][0][0]))
  for i in range(num_masked - 1, 0, -1):
    #print('here: ' + str(i))
    candidates = []
    for j in range(len(cur_preds)):
      candidates += get_n_preds_right(token_ids.detach().clone(), breadth, cur_preds[j][2], mask_positions[i - 1], cur_preds[j][0], cur_preds[j][1])
    candidates.sort(key=lambda k:k[1],reverse=True)
    if i != 1:
      cur_preds = candidates[:beam_size]
    else:
      cur_preds = candidates[:breadth]
  for tokens, probability, _ in cur_preds:
    tokens.reverse()
  return cur_preds

def display_word(toks):
  s = ''
  first_tok = True
  for tok in toks:
    is_suffix = tok.startswith('##')
    if is_suffix: tok = tok[2:]  # remove suffix hashtags
    elif not first_tok: s += ' '
    
    s += tok
    
    first_tok = False
  return s


# Text should be provided with "{tokenizer.mask_token}" in place
#  of the blanks desired to be infilled.
text = f"""Καὶ ἡ τοῦ Μηδικίου δωρεὰ ἀπὸ τοῦ ἐμοῦ προσώπου εἰς τὸ ἐκείνου μεταπλασθήτω, ὡς ἂν γνῷ καὶ οὗτος, οἷα τὰ παρὰ τῆς σῆς δεσποτικῆς χειρὸς ἐπιφορτιζόμενα ἡμῖν ἄχθη, ὡς εὐπετῆ καὶ κοῦφα, καὶ ἐπιρρωννύντα μᾶλλον ἢ δαπανῶντα τοῦ ἀχθοφοροῦντος τὴν δύναμιν.    Ὁρᾷς ὅπως κατατολμῶ σου, θειότατε βασιλεῦ, καὶ οὔτε σου τὸν ὑπερφαῆ κύκλον δέδοικα, οὔτε σου τὸ μέγεθος τῆς ψυχῆς πέφρικα, ἀλλὰ πολλάκις σοι δημηγόρος ἐφέστηκα;    εἰ γὰρ καὶ καταπλήττεις τῷ ἀπαραμίλλῳ κάλλει τῶν ἀρετῶν, ἀλλ’ ὡς θεὸς εὐμενὴς καὶ ἵλεως ἕστηκας·    καί σού τις, δειλιῶν τὸ τῆς λαμπηδόνος ὑπερφυές, καὶ ἀτεχνῶς μύων τὰ ὄμματα, θαρρεῖ πως τὴν ἐπιείκειαν.    Αὕτη γοῦν κἀμὲ πολλάκις δημηγόρον ποιεῖ·    καὶ ὑποχωροῦντα ἐφέλκεται·    καὶ ὑποστελλόμενον ἠρέμα ἐπάγεται.    Δύο γοῦν ἐπὶ σοὶ ἀπέραντα κατανενόηκα πέρατα ὕψος καὶ βάθος·    τὸ μὲν φρονήσεως, τὸ δὲ ταπεινώσεως.    Ἀλλ’ ὅτε μὲν εἰς τὸ ὕψος ἀνανεύσω τὴν κεφαλήν, ἰλιγγιῶ καὶ σκοτοδινιῶ, καὶ οὐκ ἔχω πῶς ἂν ἐνατενίσω σου τῷ ἀπείρῳ φωτί.    Ὅταν {tokenizer.mask_token}{tokenizer.mask_token}{tokenizer.mask_token} βάθος τῆς σῆς μετριοφροσύνης ἐγκύψω ὥσπερ εἰς ἀχανὲς πέλαγος, μικροῦ δεῖν ἐξίσταμαι τῶν φρενῶν καὶ οὐκ ἔχω, πῶς ἂν ἐμαυτὸν ἐπιστηρίξω, ἵνα σου θεάσωμαι τὰ ἀθέατα.    Ὢ ἀρρήτου συγκράσεως Ὢ εὐμελοῦς τῶν ἐναντίων μίξεως Ἐξήτασαι μετὰ ἀγγέλων ταῖς τῶν ἀρετῶν ἀστραπαῖς, καὶ τεθέασαι μετὰ ἀνθρώπων τῷ ἀλύπῳ τῶν σῶν ἠθῶν καὶ ἡμέρῳ χρήματι.    Ἕστηκας ὥσπερ ἐν κέντρῳ τοῖς ἀνακτορικοῖς σημείοις, καὶ τὸν πάντα κύκλον περιοδεύεις τῆς οἰκουμένης.    Ἵστασαι τοῖς Ἄραψιν ἀντιπρόσωπος·    πρὸς τὴν Περσικὴν ἠγώνισαι δύναμιν·    ἀναστέλλεις τὸ βάρβαρον θράσος. """

tokens = tokenizer.encode(text, return_tensors='pt').to(device)
sugs = beam_search_right(tokens, 20)

for suggestion, probability, _ in sugs:
  converted = tokenizer.convert_ids_to_tokens(suggestion)
  print(f"{probability:.1%} - {display_word(converted)}")
