#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:17:53 2023

@author: Charlie Cowen-Breen

This short script reads in flags from the filsystem which 
have been collected by "build_report.py", and generates a
graphical representation of the report. The output is an 
HTML file which contains suggested flags ranked by 
chance-confidence ratio [1], together with visually
color-coded representations of the paragraphs which contain
them. Standard web browsers can then convert HTML to PDF
for accessibility.
"""

import os
import time
#import numpy as np
from polyleven import levenshtein

# Number of flags to include in report
num_flags = 100

# Directory in which to search for flags
base_directory = '.'

# "None" if Levenshtein distance should not be strictly enforced in report. If lev takes an integer other than None, the report is constrained to only give flags with Levenshtein distance lev.
lev = None
suggestion_separation_character = '*'
punctuation = ['·', '.', ',', ';', '?', '(', ')', ':', '!', '‚', '’', '—', '«', '»']


# These five text files must be present in a given directory for it to be able to generate a report
necessaryFiles = ['suggestions.txt', 'transmittions.txt', 'wordconfidences.txt', 'wordchances.txt', 'wordratios.txt']

def isSubset(subset, superset):
  for a in subset:
    if not a in superset:
      return False
  return True

# Warmup task: get all subdirectories containing the required txt files
def getDirectoryList(path):
    directoryList = []

    #return nothing if path is a file
    if os.path.isfile(path):
        return []

    #add dir to directorylist if it contains .txt files
    if isSubset(necessaryFiles, os.listdir(path)):
        directoryList.append(path)

    for d in os.listdir(path):
        new_path = os.path.join(path, d)
        if os.path.isdir(new_path):
            directoryList += getDirectoryList(new_path)

    return sorted(directoryList)

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

def handle_tensor(s):
  assert s.startswith('tensor')
  return float(s.split('[')[1].split(']')[0])

def find_nth(haystack, needle, n):
  start = haystack.find(needle)
  while start >= 0 and n > 1:
      start = haystack.find(needle, start+len(needle))
      n -= 1
  return start

def report_flag(transmitted, suggested, ratio, chances, confidences, index=None):
  ret = ""

  flag = index

  possibilities = [s for s in suggested.split(suggestion_separation_character)[flag:flag+1+len(suggested.split(suggestion_separation_character))-len(transmitted.split())] if s != '?' and (True if (not lev) else levenshtein(s, transmitted.split()[flag]) == lev)]
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
    #display (Markdown(f"<p>Caution! Indexing issue may have occured in this paragraph. Please consider each of the suggestions listed above individually.</p>"))

  return ret

def make_report(ordered_flags, trunc):
  count = 0
  ret = ""
  suggestions = allsuggestions
  transmittions = alltransmittions
  ratios = allwordratios
  chances = allwordchances
  confidences = allwordconfidences

  for flag in ordered_flags:
    index = [flag[0], flag[1]]
    if not flag[1]: pass #print(flag)
    ratio = ratios[index[0]]
    suggested = suggestions[index[0]]
    transmitted=transmittions[index[0]]
    flag = index[1]
    possibilities = [s for s in suggested.split(suggestion_separation_character)[flag:flag+1+len(suggested.split(suggestion_separation_character))-len(transmitted.split())] if s != '?' and (True if (not lev) else levenshtein(s, transmitted.split()[flag]) == lev)]
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

start_time = time.time()

allfiles = []
for n, folder_path in enumerate(getDirectoryList(base_directory)):
  allfiles.append({
      'path': folder_path,
      'num': n
  })
  for doctype in necessaryFiles:
    with open(folder_path + '/' + doctype, 'r') as f:
      allfiles[n][doctype] = f.read().splitlines()
    if doctype in ['wordchances.txt', 'wordconfidences.txt', 'wordratios.txt']:
      # In this case, the doc contains numbers
      allfiles[n][doctype] = [[(float(s) if not s.startswith('tensor') else handle_tensor(s)) for s in l[1:-1].split(',')] for l in allfiles[n][doctype]]
      # ignore open and closing brackets, etc
    # remove redundnacies
    allfiles[n][doctype] = [allfiles[n][doctype][k] for k in range(len(allfiles[n][doctype]))]

print("Sample file content: \n")
for k in range(min(5, len(allfiles))):
  print(allfiles[0]['transmittions.txt'][k][:find_nth(allfiles[0]['transmittions.txt'][k], ' ', 10)], '...')
  print(allfiles[0]['suggestions.txt'][k][:find_nth(allfiles[0]['suggestions.txt'][k], suggestion_separation_character, 10)], '...')
  print(allfiles[0]['wordratios.txt'][k][:10], '...')
  print()

for n, files in enumerate(allfiles):
  for k in range(len(files['transmittions.txt'])):
    t = files['transmittions.txt'][k]
    s = files['suggestions.txt'][k]
    r = files['wordratios.txt'][k]
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

# Before the have been sorted by CCR, flags consist of all triples (paragraph_number, word_number, CCR).
unsorted_flags = [[(par_index, word_index, allwordratios[par_index][word_index]) for word_index in range(len(allwordratios[par_index]))]
                   for par_index in range(len(allwordratios))]
all_unsorted_flags = sum(unsorted_flags, [])
sorted_flags = sorted(all_unsorted_flags, key=lambda x: x[2])[::-1] # Sort all flags by CCR in descending order

output = ""
Lectio = """<center><h1><font color='#858585'>Lectio Facilior: ρ ≫ 1</font></h1></center>"""
#display (Markdown(Lectio))
output += Lectio
output += make_report(sorted_flags, num_flags)

with open('header.txt', 'r') as f:
  header = f.read()
  f.close()

with open('footer.txt', 'r') as f:
  footer = f.read()
  f.close()

with open(f'{base_directory}/finalreport.htm', 'w') as f:
  f.write(header + output + footer)
  f.close()
print()
print(f"Graphical report successfully generated and stored in finalreport.htm in {time.time()-start_time} seconds.")