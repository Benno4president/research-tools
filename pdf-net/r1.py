import argparse
import csv
import pandas as pd
import spacy
import fitz
import re
"""### Extract named entities
We use spacy. More fun stuff to do [there](https://www.analyticsvidhya.com/blog/2021/06/nlp-application-named-entity-recognition-ner-in-python-with-spacy/).
"""
def words(doc_df:pd.DataFrame, use_named_entities=True, high_quality_model=False):
    if use_named_entities:
      # Named entities recognition engine
      if high_quality_model:
        NER = spacy.load("en_core_web_lg")
      else:
        NER = spacy.load("en_core_web_sm")

      doc_index = {}
      count=1
      print(f"Extracting named entities from {len(doc_df)} documents. This might take a while...")
      for index, row in doc_df.iterrows():
        text = row['text']
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
      #spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
      NLP = English()
      doc_index = {}
      count=1
      print("Extracting words from "+str(len(doc_df.index))+" documents. This might take a while...")
      for index, row in doc_df.iterrows():
        text = row['text']
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

    doc_notxt_df = doc_df.drop(columns=['text'])
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

    """### Aggregate pairs into a dataframe"""

    pair_list = []
    for index, row in doc_notxt_df.iterrows():
      for entsign in doc_index[index]:
        if word_index[entsign]['count-documents']>1:
          new_row = {**row, **doc_index[index][entsign]}
          pair_list.append(new_row)

    pair_df = pd.DataFrame(pair_list)

    """### Save as CSV"""
    output_file_words = "words.csv"
    output_file_pairs = "words-and-documents.csv"
    try:
      pair_df.to_csv(output_file_pairs, index = False, encoding='utf-8')
    except IOError:
      print("/!\ Error while writing the pairs output file")

    try:
      word_df.to_csv(output_file_words, index = False, encoding='utf-8')
    except IOError:
      print("/!\ Error while writing the words output file")
    print("Done.")


def get_pdf_text(pdf_path):
    pdf_text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            block = page.get_text("blocks")[1:]
            for x0, y0, x1, y1, text, block_no, block_type in block:
                if block_type == 0:
                    text = text.replace("\n", " ")
                    text = re.sub(r"([a-zA-Z])- ?([a-zA-Z])", r"\1\2", text)
                    pdf_text += f"{text}\n"
    return pdf_text.encode(errors='replace').decode()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='+', type=argparse.FileType())
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # Use named entities
    # (if set to False, it just extracts words)
    # (don't extract all words if you have many documents!)
    use_named_entities = True
    # Use high quality named entities recognition model (slower, takes space)
    high_quality_model = False
    if use_named_entities and high_quality_model:
        spacy.cli.download("en_core_web_lg")

    
    pdf_text = [get_pdf_text(x) for x in args.file]
    doc_df = pd.DataFrame(pdf_text, columns=['text'])
    words(doc_df, use_named_entities, high_quality_model)