# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 18:13:11 2020

@author: USER
"""

import gensim
import pandas as pd
import os
import pickle
import re
#import pyLDAvis.gensim
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
#from wordcloud import WordCloud
import  matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
#from iteration_utilities import deepflatten  ## list flatten 
import random
from collections import Counter
import time
from datetime import datetime 
import matplotlib.font_manager
import networkx as nx
import matplotlib.font_manager as fm
import math
from sklearn.manifold import TSNE
from gensim.models import CoherenceModel
from soynlp.tokenizer import LTokenizer
from soynlp.noun import LRNounExtractor_v2

matplotlib.font_manager.findSystemFonts(fontpaths  = None , fontext = 'ttf') # 활용가능한 폰트 찾기 

def make_dtm(text, max_features = 1000000, min_df = 1, stopwords = None):
    while True:
        cv =  CountVectorizer(max_features= max_features, 
                              min_df = min_df, 
                              stop_words = stopwords,
                              tokenizer = lambda x: x.split(' '))
        tdm = cv.fit_transform(text)
        words = cv.get_feature_names()
                
        try:
            tdm_mat = tdm.toarray()
        except MemoryError:
            min_df += 10
            print("\rmin_freq:", min_freq, end = " ")
        else:
            break

    tdm_df = pd.DataFrame(tdm_mat)
    tdm_df.columns= np.array(words)
    return tdm_df

def make_wordcloud(dtm_mat, width=1000, height=500, max_words = 500):
    dtm_mat_sum = dtm_mat.sum(axis = 0)
    wc_dic = {}
    for i in range(len(dtm_mat_sum)):
        wc_dic[dtm_mat_sum.index[i]] = int(dtm_mat_sum.iloc[i])
    wordcloud = WordCloud(font_path='/Library/Fonts/NanumSquareRoundEB.ttf',
                          width = width, height = height,
                          background_color = "white",
                          max_words=max_words).generate_from_frequencies(wc_dic)    
    return wordcloud

def analy_asso(dtm_mat, support, confidence):
    asso_data = dtm_mat.apply(lambda x: np.where(x ==0, False, True), axis = 0)        
    while True:
        try:
            asso_gen = apriori(asso_data, min_support=support, use_colnames=True,
                                   max_len = 2, low_memory = True)
        except MemoryError:    
            support += 0.0001
            print("\rsupport:",support, end = " ")
        else:
            break
            
    asso_cut_confidence = association_rules(asso_gen, metric="confidence", min_threshold= confidence )
    return(asso_cut_confidence)

def make_network(asso_rules, font_size = 10, eps = .1, figure = True):
    sample_g = nx.Graph()
    for i in range(len(asso_rules)):
        sample_g.add_edge(" ".join(list(asso_rules["antecedents"][i])),
                              " ".join(list(asso_rules["consequents"][i])),
                              weight = asso_rules["support"].iloc[i])
            
    pgr_cen = nx.pagerank(sample_g) ## 페이지링크 알고리즘을 해당노드의 연결성(엣지 웨이트)
    sorted_pgr = sorted(pgr_cen.items(), key = (lambda x:x[1]), reverse = True)
        
    graphic_g = nx.Graph()

    for i in range(len(sorted_pgr)):
        graphic_g.add_node(sorted_pgr[i][0])
                            
    for j in range(len(asso_rules)):
        graphic_g.add_weighted_edges_from(
                    [ (" ".join(list(asso_rules["antecedents"][j])),
                       " ".join(list(asso_rules["consequents"][j])),
                       asso_rules["support"].iloc[j] * 100 )])
        
    weight_list = ["떡볶이"]
    node_weight = []
    for key, values in pgr_cen.items():
        if key in weight_list:
                node_weight.append(math.log(values * 50000 * 10000)* 500)
        else:
            node_weight.append(math.log(values * 50000)* 500)
    
    if figure:
        nx.draw(graphic_g, 
                node_color = 'white',
                nodelist = pgr_cen.keys(),
                node_size = node_weight,
                edge_color= "#2090e6", 
                pos = nx.spring_layout(graphic_g,k = eps), 
                font_size = font_size,
                font_family = 'AppleGothic',
                with_labels = True,
                font_weight = 'bold',
                alpha = 1,
                font_color = "black",
                edgecolors = "#2090e6",
                linewidths = 2)
    
    else:
        return sorted_pgr


### topic modeling -> LDA 

def compute_coherence_values(text, limit, start=2, step=3):
    
    text = [words.split(" ") for words in text]  
    dictionary = corpora.Dictionary(text)
    corpus = [dictionary.doc2bow(text) for text in text]
    coherence_values = []
    model_list = []
    perplexity_values = []
    for num_topics in range(start, limit, step):
        print("Topics: {}".format(num_topics))
        model = gensim.models.ldamodel.LdaModel(corpus, 
                                               id2word=dictionary, 
                                               num_topics = num_topics, 
                                               update_every = 1,
                                               #chunksize = round(len(corpus)/20),
                                               passes=10)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=text, dictionary=dictionary, coherence='c_v')        
        perplexitymodel = model.log_perplexity(corpus)
        
        coherence_values.append(coherencemodel.get_coherence())
        perplexity_values.append(perplexitymodel)
        
    return model_list, coherence_values, perplexity_values

def run_lda(text, num_topics):
    """
    text: 문장
    num_topics: 도출할 토픽의 수
    num_words: 토픽당 출력할 단어의 수(시각화용)
    """
    text = [words.split(" ") for words in text]  
    dictionary = corpora.Dictionary(text)
    corpus = [dictionary.doc2bow(text) for text in text]

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, 
                                               id2word=dictionary, 
                                               num_topics = num_topics, 
                                               update_every = 1,
                                               #chunksize = round(len(corpus)/20),
                                               passes=10)
    return corpus, ldamodel

def find_topic_terms(lda_model, topicn, topn = 20):
    term_dic = {}
    for idx, term in lda_model.id2word.iteritems():
        term_dic[idx]= term
        
    topic_terms = lda_model.get_topic_terms(topicn, topn = topn)
    all_terms = []
    all_prob = []
    for term_prob in topic_terms:
        term = term_dic[term_prob[0]]
        prob = term_prob[1]
        all_terms.append(term)
        all_prob.append(prob)
        
    return(pd.DataFrame({"{}_term".format(topicn) : all_terms,
                "{}_prob".format(topicn) : all_prob}))
     
  
def find_doc_topic(model, corpus):
    '''
    model : 토픽 모델
    corpus : 모델 생성시 활용된 corpus, doc2bow 함수를 통해 성성된 결과물임
    '''
    
    all_assigned_topic = []
    all_assigned_prob = []
    for cor in corpus :
        topic_list = model.get_document_topics(cor, minimum_probability = 0)
        sorted_topic_list = sorted(topic_list, key=lambda x:x[1], reverse=True)
        all_assigned_topic.append(sorted_topic_list[0][0])
        all_assigned_prob.append(sorted_topic_list[0][1])
        
    return pd.DataFrame({"topic" : all_assigned_topic, "prob" : all_assigned_prob})


if __name__ == "_main_":
    
    
  mac_path = "/Users/seongwoo/OneDrive/2020_하반기/키워드도출_북구"
  main_path = mac_path
  
  
  main_dir = 'F:/(업무)D-Help Desk/(2109_예산담당관)주민참여예산사업 분석'
  apply = pd.read_csv(os.path.join(main_dir, 'data', 'apply_from.csv'), encoding = 'cp949')
  
  
  ### 시작 #####
  
  subs = apply["사업명"] 
  ### 단어빈도 및 네트워크 분석 ###
  word_freq = []
  word_pgr = []
  for year , grpd in apply.groupby("연도"):
    doc_cnt = len(grpd)
    print(year)
    
    dtm_mat = make_dtm(grpd["title_cont"], stopwords = ["마리", "광역","작년"])

    plt.figure( figsize = (20, 20) )
    plt.imshow(make_wordcloud(dtm_mat, width = 700, height = 1000))
    plt.axis("off")
    plt.savefig(main_path + "/results/{}/{}_wordcloud.png".format(f_name ,year), bbox_inches='tight')
    plt.show()
    plt.close()

    word_divide_doc = pd.DataFrame(pd.Series(dtm_mat.sum(axis = 0), name = "{}_freq".format(year)))
    word_divide_doc['{}_idf'.format(year)] = pd.Series(dtm_mat.sum(axis = 0), name = "{}_freq".format(year)) / doc_cnt
    word_freq.append(word_divide_doc)

    asso_rules = analy_asso(dtm_mat,support= 0.05, confidence = 0.1)
    plt.figure( figsize = (40, 20) )
    make_network(asso_rules, font_size = 20,eps = 1, figure = True)
    plt.axis("off")
    plt.savefig(main_path + "/results/{}/{}_wordnetwork.png".format(f_name, year), bbox_inches = "tight")
    plt.close()

    pgr_list = {}
    for word, pgr in make_network(asso_rules, font_size = 25,eps = 1, figure = False):
        pgr_list[word] = pgr
    word_pgr.append(pd.Series(pgr_list, name = "{}_pgr".format(year)))

  word_freq2 = pd.concat(word_freq, axis = 1).fillna(0)
  word_freq2["2019_idf_inc"] = round(((word_freq2["2019_idf"] - word_freq2["2018_idf"]) / word_freq2["2018_idf"]).fillna(0) * 100, 1)
  word_freq2["2020_idf_inc"] = round(((word_freq2["2020_idf"] - word_freq2["2019_idf"]) / word_freq2["2019_idf"]).fillna(0) * 100, 1)

  word_pgr2 = pd.concat(word_pgr, axis = 1).fillna(0)
  word_pgr2["2019_pgr_inc"] = round(((word_pgr2["2019_pgr"] - word_pgr2["2018_pgr"]) / word_pgr2["2018_pgr"]).fillna(0) * 100, 1)
  word_pgr2["2020_pgr_inc"] = round(((word_pgr2["2020_pgr"] - word_pgr2["2019_pgr"]) / word_pgr2["2019_pgr"]).fillna(0) * 100, 1)

  word_freq2.to_csv(main_path + "/results/{}/anual_keywords.csv".format(f_name), encoding = "cp949")
  word_pgr2.to_csv(main_path + "/results/{}/pgr_keywords.csv".format(f_name), encoding = "cp949")







