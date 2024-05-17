import streamlit as st
import pandas as pd
import networkx as nx   
from r1 import main as _get_cluster_dist
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def sidebar():
    selections = {
        'out':None,
        }
    
    return selections

@st.cache_data
def get_cluster_dist():
    return _get_cluster_dist()

def main():
    """
    [] https://github.com/ferru97/PyPaperBot?tab=readme-ov-file
    """
    margins_css = """
    <style>
        .main > div {
            padding-left: 1rem;
            padding-right: 3rem;
        }
    </style>
    """

    st.set_page_config(page_title=f"Explorer",layout='wide')
    st.markdown(margins_css, unsafe_allow_html=True)

    c1,c2 = st.columns((2,1))

    G, doc_distances, clusters = get_cluster_dist()
    doc_distances = pd.DataFrame.from_dict(doc_distances).T
    
    # calc portion
    doc_clusters = []
    for col in doc_distances.columns:
        mod = 0.95
        for _ in range(3):
            print(f'### {col} has mod {mod}')
            portion = doc_distances[doc_distances[col] < doc_distances[col].mean()*mod]
            if len(portion) > 4 and len(portion) < len(doc_distances) // 3:
                doc_clusters.append(portion)
                break
            mod += 0.015 if not len(portion) > 4 else -0.01

    with c1:
        for doc_c in doc_clusters:
            st.dataframe(doc_c)
    with c2:
        st.write(clusters)

    #communities = nx.community.greedy_modularity_communities(G)
#
    ## Compute positions for the node clusters as if they were themselves nodes in a
    ## supergraph using a larger scale factor
    #supergraph = nx.cycle_graph(len(communities))
    #superpos = nx.spring_layout(G, scale=50, seed=429)
#
    ## Use the "supernode" positions as the center of each node cluster
    #centers = list(superpos.values())
    #pos = {}
    #for center, comm in zip(centers, communities):
    #    pos.update(nx.spring_layout(nx.subgraph(G, comm), center=center, seed=1430))
#
    ## Nodes colored by cluster
    #for nodes, clr in zip(communities, ("tab:blue", "tab:orange", "tab:green")):
    #    nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes, node_color=clr, node_size=100)
    #nx.draw_networkx_edges(G, pos=pos)
    #with c1:
        

    #with c2:


#def too_much_effort():




if __name__ == "__main__":
    main()    
