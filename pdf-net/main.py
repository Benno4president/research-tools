"""
Copyright Kristian Bengtson 2024

DISCLAIMER:
    All this code is trash, i will fix it oneday..


"""
import os
import re
import sys
import csv
import fitz
import string
import argparse
import networkx as nx
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from spacy.lang.en import English
import nltk

nltk.download('averaged_perceptron_tagger')



path = lambda *x: os.path.abspath(os.path.join(os.path.dirname(__file__), *x))
pdf_dir = './docs'

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

"""
nltk tag legend
https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
OBSERVATION: 
    'NNS' only allows for selecting top k top represent the group. fx systems, drones, data, results, applications
    'NN' does too. fx drone, UAV, network, control, Online, search. but also pp, m, vol, et, al
    'NNP' is trash. Australia, 8, January
"""

def is_noun(word:str):
    tagged_sentence = nltk.tag.pos_tag([word])
    #edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
    #return ' '.join(edited_sentence)
    w, tag = tagged_sentence[0]
    ok = (tag in ['NN','NNS']) 
    return ok



"""
https://github.com/jacomyma/mapping-controversies/blob/main/notebooks/Documents_with_text_to_word_list.ipynb

https://github.com/jacomyma/mapping-controversies/blob/main/notebooks/Words_and_documents_with_text_to_network.ipynb
"""


def doc_2_wordlist(df: pd.DataFrame, text_key: str):
    doc_df = df.copy()
    doc_index = {}
    # Tokenizer (extract words)
    NLP = English()
    count=1
    print("Extracting words from "+str(len(doc_df.index))+" documents. This might take a while...")
    for index, row in doc_df.iterrows():
        text = row[text_key]
        if count % 10 == 0:
            print("Text extracted from "+str(count)+" documents out of "+str(len(doc_df.index))+". Continuing...")
        count += 1

        tokens = NLP(text)
        words = {}
        for token in tokens:
            if not token.is_stop and token.is_alpha:
                t = token.text.lower()
                if t not in words:
                    words[t] = {'W-text': t, 'W-count':0}
                words[t]['W-count'] += 1
        doc_index[index] = words


    doc_notxt_df = doc_df.drop(columns=[text_key])
    word_index = {}
    for index, row in doc_notxt_df.iterrows():
        for wsign in doc_index[index]:
            w = doc_index[index][wsign]
            if wsign not in word_index:
              word_index[wsign] = {'text': w['W-text'], 'count-occurences-total':0, 'count-documents':0}
            word_index[wsign]['count-occurences-total'] += w['W-count']
            word_index[wsign]['count-documents'] += 1

    word_df = pd.DataFrame(word_index.values())
    return word_df


def text_2_network_graph(text_df:pd.DataFrame, id_key:str, text_key:str, word_df:pd.DataFrame, word_key:str):
    # Delete documents that contain none of the words?
    discard_unrelated_documents = True

    # Get a set of the words
    words = set()
    for index, row in word_df.iterrows():
      words.add(row[word_key])

    # Init data for output
    network_doc_set = set()
    network_word_set = set()
    network_edge_list = []

    # Search words in documents
    for index, row in text_df.iterrows():
        print(index, end='\r')
        if type(row[text_key]) == str:
            text = row[text_key].lower()
            count_per_word = {}
            flag = False
            for word in words:
                count = text.count(word.lower())
                count_per_word[word] = count
                if count > 0:
                    flag = True

            if flag or not discard_unrelated_documents:
                doc_id = row[id_key]
                network_doc_set.add(doc_id)
                for word in words:
                    count = count_per_word[word]
                    if count > 0:
                        network_word_set.add(word)
                        network_edge_list.append((doc_id,word,{"count":count}))

    # Build the nodes
    nodes = []
    text_df_no_text = text_df.drop(columns=[text_key]) 
    for index, row in text_df_no_text.iterrows():
      if row[id_key] in network_doc_set:
        nodes.append((row[id_key], {**row, 'title':row['title'], 'label':row[id_key], 'type':'document'}))

    for index, row in word_df.iterrows():
      if row[word_key] in network_word_set:
        nodes.append((row[word_key], {**row, 'label':row[word_key], 'type':'term'}))

    # Build edges
    edges = network_edge_list

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    print("Done building network graph")
    return G


RANDOM_BS_I_DONT_LIKE_AARRHH = list(string.ascii_lowercase) + [
    'pp', 'proc', 'vol', 'et', 'al',
    'ieee','time','use','uses','results',
    'number', 'distance','figure',
    'exp','authors','set','wiley','fig',
    'value','syst','university','doi','eq',
    'σ', 'μ', 'π',# make real fix
    'ij','rl','oa','nj',
    'sep', 
]
def clean_words(wo:pd.DataFrame):
    wo = wo[~wo['text'].isin(RANDOM_BS_I_DONT_LIKE_AARRHH)]
    wo = wo[wo['text'].apply(is_noun)]
    #wo = wo[wo['count-occurences-total'] > 5]
    #wo = wo[wo['count-documents'] > 5] # MAKE THIS DYNAMIC TODO TODO TODO TODO
    return wo

def word_cloud(word_count:dict[str,int]):
    # lower max_font_size, change the maximum number of word and lighten the background:
    wordcloud = WordCloud(width=1920, height=1080,max_font_size=150, max_words=250, background_color="white").generate_from_frequencies(word_count)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', const='', type=argparse.FileType())
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    if args.file: 
        files = [args.file]
        paths = [args.file]
    else:
        # hahah this is fucked, fix if folder feature is needed.
        files = [x for x in os.listdir(path(pdf_dir)) if x.endswith('.pdf')]
        paths = [os.path.join(pdf_dir,x) for x in files]

    pdf_texts = [get_pdf_text(x) for x in paths]

    doc_df = pd.DataFrame(pdf_texts, columns=['txt'])
    doc_df['title'] = files
    doc_df['temp_index'] = range(0,len(doc_df))
    
    #print("Preview of the document list:")
    #print(doc_df)
    #doc_df.to_csv('./document_df.csv', index=False, encoding='utf-8', escapechar='\\')
    #exit()
    words = doc_2_wordlist(doc_df, 'txt')
    #print(words)
    words = clean_words(words)

    topK_words = words.sort_values('count-occurences-total',ascending=False).head(50)
    print(list(topK_words['text']))

    as_dict = dict(topK_words[['text','count-occurences-total']].values)
    word_cloud(as_dict)

    #graph = text_2_network_graph(text_df=doc_df, id_key='temp_index', text_key='txt', word_df=words, word_key='text')

    #output_file_network = "./document-network.gexf"
    #nx.write_gexf(graph, output_file_network)

    # https://colab.research.google.com/github/jacomyma/mapping-controversies/blob/main/notebooks/Words_and_documents_with_text_to_document_list_with_words.ipynb
    # https://colab.research.google.com/github/jacomyma/mapping-controversies/blob/main/notebooks/Documents_with_text_to_word_list.ipynb
    ## Frequency of word set in pdf
    # find clusters
    # validate cluster ???
    # find cluster with highest frequency of word set
    # select papers
