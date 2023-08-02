#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:22:07 2023

@author: charlie
"""


from functools import lru_cache
import numpy as np
import time

# Disregard suggestions to swap between two of the following characters (uninteresting philologically)
punctuation = ['·', '.', ',', ';', '?', '(', ')', ':', '!', '‚', '’', '—', '«', '»']
lev = 1
suggestion_separation_character = '*'

""" Levenshtein tools """

# Purpose is only to confirm the required Levenshtein distance between suggestions and transmittions
def lev_dist(a, b):
    '''
    This function will calculate the levenshtein distance between two input
    strings a and b

    params:
        a (String) : The first string you want to compare
        b (String) : The second string you want to compare

    returns:
        This function will return the distnace between string a and b.

    example:
        a = 'stamp'
        b = 'stomp'
        lev_dist(a,b)
        >> 1.0
    '''

    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):

        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),      # insert character
            min_dist(s1 + 1, s2),      # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)

""" Graphical tools """

def colorcode(token, score, bold=False): # Here score = 0 is green, and score = 1 is red, and -1 is trivially gray
    ret = ''
    if bold: ret += '<b>'

    if score < 0.5 and score >= 0:
        i = int(score*256*2)
        code1 = hex(i)[2:].upper()
        if i < 16: code1 = "0" + code1
        ret += f'<font color="#{code1}FF00">{token.upper() if bold else token}</font>'
    elif score >= 0.5:
        j = int((1-score)*256*2)
        code2 = hex(j)[2:].upper()
        if j < 16: code2 = "0" + code2
        ret += f'<font color="#FF{code2}00">{token.upper() if bold else token}</font>'
    else: # score < 0
        ret += f'<font color="#000000">{token.upper() if bold else token}</font>' #858585

    if bold: ret += '</b>'
    return ret

def display_colorcoded(tokens, scores, greenness, intensifyred = False, specialtokindex = None, normalize=True):
  # In this section, we visualize the reconstructed sentence by replacing the mask_tokens with underscores, with as many underscores as characters are masked.
  first_tok = True
  s = ""

  nontrivialscores =  list(filter(lambda s: s != -1, scores)) # filter for non-trivial scores
  orderedscores = sorted(
     nontrivialscores
  )[::-1] # reverse order
  total = len(nontrivialscores)

  tokindex = 0
  for tok, score in zip(tokens, scores):  # ignore [CLS] and [SEP]
      # Determining color
      if score == -1: color = -1 # trivial gray color
      else: # determine red/green color

        index = orderedscores.index(score)
        if normalize:
          color = (index/total)**greenness if not intensifyred else 1 - (1- (index/total))**greenness
        else:
          color = (1-score)**greenness if not intensifyred else 1 - ((score))**greenness

      is_suffix = tok.startswith('##')
      if is_suffix: tok = tok[2:]  # remove suffix hashtags

      is_punctuation = tok in ['.', ',', ';']
      if (not first_tok) and (not is_suffix) and (not is_punctuation): s += ' '

      s += colorcode(tok, color, specialtokindex and tokindex in specialtokindex)

      if first_tok: first_tok = False

      tokindex += 1

  return s

def display_charcoded(text, indices):
  ret = text
  for index in indices[::-1]:
    ret = ret[:index]+'<b>'+ret[index].upper()+'</b>'+ret[index+1:]
  #display (Markdown (ret))
  return ret

def getmax(arrays, bound):
  runningmax = -1
  runningargmax = None
  for n, array in enumerate(arrays):
    for k, item in enumerate(array):
      if item < bound and item > runningmax:
        runningmax = item
        runningargmax = (n, k)
  return runningargmax

def sort_indices(arrays, trunc=100, bound=np.inf):
  sorted_indices = []
  argmax = getmax(arrays, bound)
  #print(argmax)
  while argmax:
    if len(sorted_indices) > trunc: break
    sorted_indices.append(argmax)
    bound = arrays[argmax[0]][argmax[1]]
    argmax = getmax(arrays, bound)
  return sorted_indices

def report_flag(transmitted, suggested, ratio, chances, confidences, index=None):
  ret = ""

  flag = np.argmax(ratio) if not index else index

  possibilities = [s for s in suggested.split(suggestion_separation_character)[flag:flag+1+len(suggested.split(suggestion_separation_character))-len(transmitted.split())] if s != '?' and lev_dist(s, transmitted.split()[flag]) == lev]
  suggestion_line = f"<p>{transmitted.split()[flag]} &emsp; ⟶ &emsp; {' or '.join(possibilities)}</p>"
  chance_format = f"{chances[flag]:.2E}"
  stat_line = f"<p>{chance_format.split('E')[0]} &nbsp &times &nbsp 10<sup>{chance_format.split('E')[1]}</sup> &emsp; ⟶ &emsp; {confidences[flag]:.1%}</p>"

  s = f"""<table width="100%" style="background-color: transparent;">
  <tr>
    <td> </td>
    <td> <center>{suggestion_line}</center></td>
    <td >{stat_line}</td>
  </tr>
  </table>"""
  ret += s
  #display (Markdown(s))

  ret += display_colorcoded(transmitted.split(), [1-r for r in ratio], 0.3, intensifyred=True)

  if len(possibilities) > 1:
    ret += f"<p>Caution! Indexing issue may have occured in this paragraph. Please consider each of the suggestions listed above individually.</p>"
    display (Markdown(f"<p>Caution! Indexing issue may have occured in this paragraph. Please consider each of the suggestions listed above individually.</p>"))

  return ret

def print_report(transmittions, suggestions, ratios):
  # auto-asserts that lists have same length
  for t, s, r in zip(transmittions, suggestions, ratios):
    report_flag(t, s, r)
    print()

def ordered_report(transmittions, suggestions, ratios, chances, confidences, trunc=100, bound=np.inf):
  count = 0
  ret = ""
  for index in sort_indices(ratios, trunc=1.5*trunc, bound=bound):
    ratio = ratios[index[0]]
    suggested = suggestions[index[0]]
    transmitted=transmittions[index[0]]
    flag = np.argmax(ratio) if not index[1] else index[1]
    possibilities = [s for s in suggested.split(suggestion_separation_character)[flag:flag+1+len(suggested.split(suggestion_separation_character))-len(transmitted.split())] if s != '?' and lev_dist(s, transmitted.split()[flag]) == lev]
    if len(possibilities) > 1: print(len(suggested.split(suggestion_separation_character)), len(transmitted.split(' ')))
    if possibilities == []: continue
    if transmitted.split()[index[1]] in punctuation:
      pas = True
      for p in possibilities:
        if p not in punctuation: pas = False
      if pas: continue
    if count > trunc-1: break
    count += 1
    ret += f"<h3>Flag {count}</h3>"
    #display (Markdown(f"<h3>Flag {count}</h3>"))

    formattedCCR = f"{ratios[index[0]][index[1]]:.0f}" if ratios[index[0]][index[1]] > 10 else f"{ratios[index[0]][index[1]]:.2f}"
    ret += f"<p>The {index[1]+1}th word of this paragraph (#{index[0]+1}) is suspicious.</p>"
    #display (Markdown(f"<p>The {index[1]+1}th word of this paragraph (#{index[0]+1}) is suspicious.</p>"))
    ret += report_flag(transmittions[index[0]], suggestions[index[0]], ratios[index[0]], chances[index[0]], confidences[index[0]], index[1])
    ret += f"<p>ρ = {formattedCCR}</p>"
    #display (Markdown(f"<p>ρ = {formattedCCR}</p>"))
    ret += '<break/>'
  return ret

def handle_tensor(s):
  assert s.startswith('tensor')
  return float(s.split('[')[1].split(']')[0])

start_time = time.time()

folder_name = "output"
allfiles = []
doctypes = ['suggestions.txt', 'transmittions.txt', 'wordratios.txt', 'wordchances.txt', 'wordconfidences.txt']
for foldernumber in range(1,9):
  allfiles.append({
      'num': foldernumber
  })
  for doctype in doctypes:
    with open(folder_name + '/' + str(foldernumber) + '/' + doctype, 'r') as f:
      allfiles[foldernumber - 1][doctype] = f.read().splitlines()
    if doctype in ['wordchances.txt', 'wordconfidences.txt', 'wordratios.txt']:
      allfiles[foldernumber - 1][doctype] = [[(float(s) if not s.startswith('tensor') else handle_tensor(s)) for s in l[1:-1].split(',')] for l in allfiles[foldernumber - 1][doctype]]
      #ignore open and closing brackets, etc
    #remove redundnacies
    allfiles[foldernumber - 1][doctype] = [allfiles[foldernumber - 1][doctype][n] for n in range(len(allfiles[foldernumber - 1][doctype]))]

print("Sample file content: \n")
for k in range(5):
  print(allfiles[0]['transmittions.txt'][k])
  print(allfiles[0]['suggestions.txt'][k])
  print(allfiles[0]['wordratios.txt'][k])
  print()

for n, files in enumerate(allfiles):
  for key in files:
    pass #print(len(files[key]) if type(files[key]) == list else "")
  for k in range(len(files['transmittions.txt'])):
    t = files['transmittions.txt'][k]
    s = files['suggestions.txt'][k]
    r = files['wordratios.txt'][k]
    #c = files['wordratios.txt'][k]
    #co = files['wordratios.txt'][k]
    c = files['wordchances.txt'][k]
    co = files['wordconfidences.txt'][k]

    try:
      l = zip(t, s, r, c, co)
    except:
      print('Indexing error occured')
  print(len(files['transmittions.txt']), "paragraphs reported in directory", n+1)

alltransmittions = sum([f['transmittions.txt'] for f in allfiles], [])
allsuggestions = sum([f['suggestions.txt'] for f in allfiles], [])
allwordratios = sum([f['wordratios.txt'] for f in allfiles], [])
allwordchances = sum([f['wordchances.txt'] for f in allfiles], [])
allwordconfidences = sum([f['wordconfidences.txt'] for f in allfiles], [])

uniquetransmittions = []
uniquesuggestions = []
uniquewordratios = []
uniquewordchances = []
uniquewordconfidences = []
for t, s, r, c, co in zip(alltransmittions, allsuggestions, allwordratios, allwordchances, allwordconfidences):
  if t not in uniquetransmittions and s not in uniquesuggestions and r not in uniquewordratios and c not in uniquewordchances and co not in uniquewordconfidences:
    uniquetransmittions.append(t)
    uniquesuggestions.append(s)
    uniquewordratios.append(r)
    uniquewordchances.append(c)
    uniquewordconfidences.append(co)

output = ""
Lectio = """<center><h1><font color='#858585'>Lectio Facilior: ρ ≫ 1</font></h1></center>"""
#display (Markdown(Lectio))
output += Lectio
output += ordered_report(uniquetransmittions, uniquesuggestions, uniquewordratios, uniquewordchances, uniquewordconfidences, trunc=200, bound=np.inf) #1.00)

with open('header.txt', 'r') as f:
  header = f.read()
  f.close()

with open('footer.txt', 'r') as f:
  footer = f.read()
  f.close()

with open('report.htm', 'w') as f:
  f.write(header + output + footer)
  f.close()
print()
print(f"Graphical report successfully generated and stored in report.htm in {time.time()-start_time} seconds.")