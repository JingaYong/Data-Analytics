import pandas as pd
import numpy as np
import spacy as sp
import networkx as nx
from spacy import displacy
import matplotlib.pyplot as plt
import scipy as sc
from pyvis.network import Network
import warnings
import os

warnings.filterwarnings('ignore')

# Reads the contents of the book into a doc
def book_text(book):
    ner = sp.load('en_core_web_sm')
    # root = 'D:\\Users\\Me\\Programming\\My_Python\\DataAnalytics\\Projects\\Witcher Project\\Data'
    book_t = open(os.path.join('Data',book)).read()
    book_doc = ner(book_t)
    return book_doc

# Get the entities of the books
def get_entities(book_doc):
    se_df = []
    for sent in book_doc.sents:
        entity_list = [ent.text for ent in sent.ents]
        se_df.append({'sentence':sent,'entities':entity_list})
        
    se_df = pd.DataFrame(se_df)
    return se_df

# Remove entities which are not characters from the books
def filter_entities(chars_df,ent_list):
     return [ent for ent in ent_list 
            if ent in list(char_df.character) 
            or ent in list(char_df.character_firstname)]
    

def create_relations(df,w_size):
    window_size = w_size
    rels = []

    for i in range(df.shape[0]):
        end = min(i+5,df.shape[0])
        char_list = sum((df.loc[i:end].char_entities),[])
        
        # Remove Duplicates in char_list
        char_unique = [char_list[i] for i in range(len(char_list))
                       if (i==0) or char_list[i]!=char_list[i-1]]
        # print(char_unique)
        if len(char_unique)>1:
            for idx,i in enumerate(char_unique[:-1]):
                    b = char_unique[idx+1]
                    rels.append({'source':i,'target':b})
                    
    rels_df = pd.DataFrame(rels)
    return rels_df

def organise_entity_df(rels_df):
    rels_df = pd.DataFrame(np.sort(rels_df.values,axis=1),columns=rels_df.columns)
    rels_df['weights']=1
    rels_df = rels_df.groupby(['source','target'],sort=False,as_index=False).sum()
    return rels_df

    
    
def pyvis_plot(graph,book_name):
    net= Network(notebook=True,width='1000px',height='1000px',
             bgcolor='#222222',font_color='white')

    # customize node size
    node_degree = dict(graph.degree)
    nx.set_node_attributes(graph,node_degree,'size')

    net.from_nx(graph)
    net.show(book_name,'.html')
