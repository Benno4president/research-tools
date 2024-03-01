# -*- coding: utf-8 -*-
"""Documents with text to word list

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/jacomyma/mapping-controversies/blob/main/notebooks/Documents_with_text_to_word_list.ipynb

# 🍕 Documents with text to word list

**Input:** a list of documents with their text content (CSV).

**Outputs:**
* a list of words with scores (CSV)
* a list of document-word pairs (CSV)

This scripts extracts so-called [named entities](https://en.wikipedia.org/wiki/Named-entity_recognition): words or groups of words that are person names, organizations, locations...

The **description of the types** is displayed in the SCRIPT section of the notebook (execute it first).

## How to use

1. Put your input file in the same folder as the notebook
1. Edit the settings if needed
1. Run all the cells
1. Take ALL the output files from the notebook folder

# SETTINGS
"""

# Input file
input_file = "documents.csv"

# Which column contains the text?
text_column = "Text"

# Use named entities
# (if set to False, it just extracts words)
# (don't extract all words if you have many documents!)
use_named_entities = True

# Use high quality named entities recognition model (slower, takes space)
high_quality_model = False

# Output files
output_file_words = "words.csv"
output_file_pairs = "words-and-documents.csv"

"""# SCRIPT

### Install and import libraries
This notebook draws on existing code.
You can ignore the output.
"""

# Install (if needed)
!pip install pandas
!pip install spacy

# Import
import csv
import pandas as pd
import spacy
if use_named_entities and high_quality_model:
  spacy.cli.download("en_core_web_lg")

print("Done.")

"""### Read the input file"""

doc_df = pd.read_csv(input_file, quotechar='"', encoding='utf8', doublequote=True, quoting=csv.QUOTE_NONNUMERIC, dtype=object, on_bad_lines='skip')
print("Preview of the document list:")
doc_df

"""### Extract named entities
We use spacy. More fun stuff to do [there](https://www.analyticsvidhya.com/blog/2021/06/nlp-application-named-entity-recognition-ner-in-python-with-spacy/).
"""

if use_named_entities:
  # Named entities recognition engine
  if high_quality_model:
    NER = spacy.load("en_core_web_lg")
  else:
    NER = spacy.load("en_core_web_sm")

  doc_index = {}
  count=1
  print("Extracting named entities from "+str(len(doc_df.index))+" documents. This might take a while...")
  for index, row in doc_df.iterrows():
    text = row[text_column]
    if count % 10 == 0:
      print("Named entities extracted from "+str(count)+" documents out of "+str(len(doc_df.index))+". Continuing...")
    count += 1
    entities = {}
    try:
      nertxt = NER(text)
      for ne in nertxt.ents:
        entsign = ne.text + '-' + ne.label_
        if entsign not in entities:
          entities[entsign] = {'NE-text': ne.text, 'NE-type':ne.label_, 'NE-count':0}
        entities[entsign]['NE-count'] += 1
    except:
      print("An exception occurred (a document could not be analyzed)")
    doc_index[index] = entities
else:
  # Tokenizer (extract words)
  from spacy.lang.en import English
  spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
  NLP = English()
  doc_index = {}
  count=1
  print("Extracting words from "+str(len(doc_df.index))+" documents. This might take a while...")
  for index, row in doc_df.iterrows():
    text = row[text_column]
    if count % 10 == 0:
      print("Text extracted from "+str(count)+" documents out of "+str(len(doc_df.index))+". Continuing...")
    count += 1

    tokens = NLP(text)
    words = {}
    for token in tokens:
      if not token.is_stop and token.is_alpha:
        t = token.text
        if t not in words:
          words[t] = {'W-text': t, 'W-count':0}
        words[t]['W-count'] += 1
    doc_index[index] = words

print("Done.")

"""### Explanation of the named entity types
The library does not explain the types in its documentation, because it depends on the models. But the explanation is actually embedded in the library itself. The code below outputs the meaning of the types.
"""

if use_named_entities:
  types = set()
  for index, row in doc_df.iterrows():
    for entsign in doc_index[index]:
      ne = doc_index[index][entsign]
      types.add(ne['NE-type'])
  print('Explanation of the types:')
  for t in types:
    print(' - '+t+': '+spacy.explain(t))
else:
  print('Done. (this is only relevant to named entities)')

"""### Aggregate words into dataframe"""

doc_notxt_df = doc_df.drop(columns=[text_column])
word_index = {}
for index, row in doc_notxt_df.iterrows():
  if use_named_entities:
    for entsign in doc_index[index]:
      ne = doc_index[index][entsign]
      if entsign not in word_index:
        word_index[entsign] = {'text': ne['NE-text'], 'type': ne['NE-type'], 'count-occurences-total':0, 'count-documents':0}
      word_index[entsign]['count-occurences-total'] += ne['NE-count']
      word_index[entsign]['count-documents'] += 1

  else:
    for wsign in doc_index[index]:
      w = doc_index[index][wsign]
      if wsign not in word_index:
        word_index[wsign] = {'text': w['W-text'], 'count-occurences-total':0, 'count-documents':0}
      word_index[wsign]['count-occurences-total'] += w['W-count']
      word_index[wsign]['count-documents'] += 1

word_df = pd.DataFrame(word_index.values())
print("Done.")
print("Preview of the words list:")
word_df

"""### Aggregate pairs into a dataframe"""

pair_list = []
for index, row in doc_notxt_df.iterrows():
  for entsign in doc_index[index]:
    if word_index[entsign]['count-documents']>1:
      new_row = {**row, **doc_index[index][entsign]}
      pair_list.append(new_row)

pair_df = pd.DataFrame(pair_list)
print("Done.")
print("Preview of the pairs list:")
pair_df

"""### Save as CSV"""

try:
  pair_df.to_csv(output_file_pairs, index = False, encoding='utf-8')
except IOError:
  print("/!\ Error while writing the pairs output file")

try:
  word_df.to_csv(output_file_words, index = False, encoding='utf-8')
except IOError:
  print("/!\ Error while writing the words output file")
print("Done.")