import argparse
import csv
import pandas as pd
import spacy
import re
import networkx as nx
from pprint import pprint
import utils

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
            # all numbers
            if ne.label_ == 'CARDINAL': 
               continue
            # all not YYYY, this might be dumb bc fx "17 1998" is ded. fix one day.
            if ne.label_ == 'DATE' and len(ne.text) != 4: 
               continue
            entsign = ne.text + '-' + ne.label_
            if entsign not in entities:
              entities[entsign] = {'NE-text': ne.text.lower(), 'NE-type':ne.label_, 'NE-count':0}
            entities[entsign]['NE-count'] += 1
        except:
          print("An exception occurred (a document could not be analyzed)")
        doc_index[index] = entities
    else:
      # Tokenizer (extract words)
      from spacy.lang.en import English
      leftovers = ['ieee','fig','figure','exp','use','authors','exp','set'] # sci pdfs
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
          if not token.is_stop and token.is_alpha and token.text.lower() not in leftovers:
            t = token.text.lower()
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
    return pair_df, word_df


def network_graph(text_df:pd.DataFrame, id_key:str, text_key:str, word_df:pd.DataFrame, word_key:str):
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
    seen = set()
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
                words_in_doc = set()
                last_word = ''
                #pprint(count_per_word)

                for tt in words:
                  if count_per_word[tt] > 0:
                    network_edge_list.append((doc_id,tt,{"count":count}))
                
                for word in words:
                    count = count_per_word[word]
                    if count > 0:
                        if not doc_id+word in seen: 
                          network_word_set.add(word)
                          words_in_doc.add(word)
                          seen.add(doc_id+word)
                          # word to word sequencial
                          if last_word:
                            network_edge_list.append((last_word,word,{"source": doc_id, "count":count_per_word[word]}))
                          last_word = word
                          #for w in words_in_doc:
                          #  network_edge_list.append((word,w,{"source": doc_id, "count":count_per_word[word]}))
                          
                          #network_edge_list.append((doc_id,word,{"first_encounter": doc_id,"count":count}))
                #for w in words_in_doc:
                #  for w2 in words_in_doc:
                #    if not (w == w2 or (w+w2 in seen)):
                #      # still adds both ways
                #      seen.add(w+w2)
                #      seen.add(w2+w)
                #      network_edge_list.append((w,w2,{"source": doc_id, "count":count_per_word[word]}))


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
    print(f"Done building network graph w/ nodes: {len(G.nodes)}, edges: {len(G.edges)}")
    return G


from typing import List, Dict, Set
import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community import modularity

def calculate_modularity_clusters(G: nx.Graph, weight_param: str = 'weight') -> Dict[str, List[str]]:
    """
    Calculate modularity-based clusters in the graph.

    Args:
        G (nx.Graph): Input graph with words and documents as nodes,
                     and word appearances as weighted edges.
        weight_param (str): String to access edge weight in the graph.

    Returns:
        List[Set[str]]: List of sets of communities.
    """
    communities_generator = community.greedy_modularity_communities(G, weight=weight_param)
    return [set(n) for n in communities_generator]

def _calculate_document_distance_from_clusters(G: nx.Graph, clusters: Dict[str, List[str]], weight_param: str = 'weight'
                                              ) -> Dict[str, Dict[str, float]]:
    """
    Calculate the distance of each document from each cluster using modularity score.

    Args:
        G (nx.Graph): Input graph with words and documents as nodes,
                     and word appearances as weighted edges.
        clusters (Dict[str, List[str]]): Dictionary of clusters where each key
                                          is the cluster ID (as string) and the
                                          value is a list of nodes in that cluster.
        weight_param (str): String to access edge weight in the graph.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping each document node to a
                                      dictionary of cluster IDs (as keys) and
                                      corresponding modularity distances (as values).
    """
    # Compute modularity score for the entire graph
    modularity_score = modularity(G, [set(nodes) for nodes in clusters.values()], weight=weight_param)
    
    # Calculate distances of each document from each cluster
    doc_distances = {}
    
    #for cluster_id, cluster_nodes in clusters.items():
    #  l = {}
    #  for doc in [node for node in G.nodes() if G.nodes[node]['type']=='document']:


##########
    #inverse_cluster_by_first = {(b[0], a) for a,b in clusters.items()}
    for doc in [node for node in G.nodes() if G.nodes[node]['type']=='document']:
        
        doc_distances[doc] = {}
        for cluster_id, cluster_nodes in clusters.items():
            # Create a subgraph containing the document node and all nodes in the cluster
            subgraph_nodes = set(cluster_nodes + [doc])
            subgraph = G.subgraph(subgraph_nodes)
            # Compute modularity score for the subgraph
            #print(cluster_nodes)
            #subgraph_nodes.remove(doc)
            #print(set(subgraph_nodes) == set(cluster_nodes))
            #print([nodes for nodes in clusters.values() if nodes == cluster_nodes] )
            #print(set(clusters[cluster_id]))
            #exit()
            subgraph_modularity = modularity(subgraph, [G.nodes[doc]], weight=weight_param)
            #print(subgraph_nodes)
            
            # Calculate modularity-based distance
            distance = modularity_score - subgraph_modularity
            
            # Store the modularity score in the edge (document, cluster)
            doc_distances[doc][cluster_id] = distance
    
    return doc_distances


def calculate_document_distance_from_clusters(G: nx.Graph, clusters: List[Set[str]], weight_param: str = 'weight') -> Dict[str, Dict[str, float]]:
    """
    Calculate the distance of each document node from each cluster of words.

    Parameters:
        G (nx.Graph): The input graph where nodes represent words and documents.
        clusters (List[Set[str]]): List of sets of words representing clusters.
        weight_param (str): The edge attribute to be used as weights in the graph (default is 'weight').

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where keys are document nodes
            and values are dictionaries mapping cluster names to distance values.
    """
    # Dictionary to store distances from documents to clusters
    doc_distances = {}

    # Iterate over each document node
    document_nodes = [node for node in G.nodes() if G.nodes[node]['type']=='document']
    
    for doc in document_nodes:
        doc_distances[doc] = {}

        # Get the set of words in the document
        doc_words = set(G[doc])

        # Calculate distances to each cluster
        for cluster_idx, cluster_words in enumerate(clusters):
            cluster_name = f"Cluster_{cluster_idx + 1}"
            
            # Compute Jaccard similarity as distance metric
            if len(doc_words) == 0 and len(cluster_words) == 0:
                distance = 0.0
            else:
                distance = 1.0 - len(doc_words.intersection(cluster_words)) / len(doc_words.union(cluster_words))

            # Store the distance in the dictionary
            doc_distances[doc][cluster_name] = distance

    return doc_distances

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdfs', nargs='+', type=argparse.FileType())
    parser.add_argument('-o','--out', type=str, help='output as gexf')
    args = parser.parse_args()
    return args

def main():    
    """
    TODO:
      Words are too plentyfull and weird
        [OK] maybe remove cardinals
        [OK] clean the text

      
      [OK] document node 1 and term node "1" is merged in gephi import
      [?] Maybe make each pdf be a node while using blocks or sentences instead of entire pdf for word2word
        - update: used sentences.. it should be word mapped to next word in sentence maybeee.
      [] Use https://www.nltk.org/api/nltk.stem.porter.html for suffix stripping
      [] add weight parameter to .add_edge(.., weight=variable) 
      [] Modular framework could have potential
      [] graphpage.network_graphs https://medium.com/@amuhryanto/analyzing-trade-networks-using-networkx-and-plotly-unveiling-patterns-and-insights-e36d5d242e58
      [] add variable for amount of clusters
    """
    args = parse_args()
    # Use named entities
    # (if set to False, it just extracts words)
    # (don't extract all words if you have many documents!)
    use_named_entities = False
    # Use high quality named entities recognition model (slower, takes space)
    high_quality_model = False
    #if use_named_entities and high_quality_model:
    #    spacy.cli.download("en_core_web_lg")  

    print('collecting pdfs')
    pdf_text = [utils.get_pdf_text(x) for x in args.pdfs]
    

    print('cleaning raw text')
    titles_col = []
    new_text = []
    for f, t in zip(args.pdfs, pdf_text):
      s = t.split('\n') # maybe a pdf block comes out with \n every time? 
      s = [utils.clean_text(x) for x in s]
      s = [x for x in s if len(x.split(' ')) > 2]
      new_text += s
      titles_col += [f.name]*len(s)
    pdf_text = new_text
    #pdf_text = [x for x in pdf_text if len(x.split(' ')) > 3]
    
    doc_df = pd.DataFrame(pdf_text, columns=['text'])
    #doc_df['title'] = list(map(lambda x: x.name, args.file))
    doc_df['title'] = titles_col
    print(doc_df)
    
    
    
    print('making word lists')
    pairs_df, words_df = words(doc_df, use_named_entities, high_quality_model)
    print(len(words_df))

    # contraints
    words_df = words_df[words_df['count-documents'] > 1]
    words_df = words_df[words_df['text'].apply(lambda x: len(x)>2)]
    words_df['cod'] = words_df['count-occurences-total']
    words_df = words_df[words_df['cod'] > words_df['cod'].max()*0.08]
    #print(words_df.sort_values('cod',ascending=False))
    print(len(words_df))
    #print(words_df)

    print('making graph')
    G = network_graph(text_df=doc_df, id_key='title', text_key='text', word_df=words_df, word_key='text')

    print('finding clusters')
    clusters = calculate_modularity_clusters(G, weight_param='weight')
    #print("Clusters:", clusters)
    
    doc_distances = calculate_document_distance_from_clusters(G, clusters, weight_param='count-occurences-total')
    print("Document Distances:")
    pprint(doc_distances)

    if args.out:
      output_file_network = f"{args.out}.gexf"
      print('writing graph')
      nx.write_gexf(G, output_file_network)
    else:
      return G, doc_distances, clusters
    print('Done')

if __name__ == '__main__':
  main()
