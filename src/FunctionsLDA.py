# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:53:54 2021

@author: nelso
"""

from src.libraries import *

def TreatOverview(df):
    overviewm=df["overview"]
    #Regex  \p{M}
    
    overviewm=overviewm.str.lower()
    overviewm = overviewm.replace(r'\p{M}'," ")
    overviewm = overviewm.str.replace('[^\w\s]',' ')
    overviewm = overviewm.str.replace('\s+', ' ', regex=True)
    overviewm = overviewm.str.replace(r'[^\w\s]'," ")
    overviewm = overviewm.str.replace(r'\S*\d\S*', ' ')
    overviewm = overviewm.str.replace(r'\W*\b\w{18,}\b', ' ')
    overviewm = overviewm.str.replace(r'W*\b\w{1,2}\b', ' ')
    overviewm = overviewm.str.replace(r' +', ' ')
    
    return overviewm

def perf_LDA(text, dico, corpus, i, coherence_val):
    """ Number of topics
    """  
    print("Number of topics:", i, "\n")

    modele_lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dico,
                                           num_topics=i,
                                           random_state=100,
                                           #update_every=1,
                                           #chunksize=10,
                                           #passes=10,
                                           alpha='auto')

    coherence_model_lda = CoherenceModel(model=modele_lda, texts=text, dictionary=dico, coherence='c_v')
    res_cv              = coherence_model_lda.get_coherence()
    coherence_val.append(res_cv)
    print("Coherence c_v:", res_cv, "\n")

    return(coherence_val)


#Permet d'attribuer un topic à chaque film, ainsi que la probabilité d'appartenir à ce topic et les keywords qui représentent le topic
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.DataFrame(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


def recommandationsystem(df_dominant_topic,movie):
    Recommand = df_dominant_topic.copy()
    Interval  = [(Recommand.loc[Recommand['Titre'] == movie, 'Probabilité_Topic'].iloc[0])*(1-abs(0.05)),\
       (Recommand.loc[Recommand['Titre'] == movie, 'Probabilité_Topic'].iloc[0])*(1+abs(0.05))]
    Recommand["Probabilité_Topic"]=Recommand[(Recommand["Topic_Dominant"]==Recommand.loc[Recommand['Titre'] == movie, \
                    'Topic_Dominant'].iloc[0]) & (Recommand["Genre"]==Recommand.loc[Recommand['Titre'] == movie, \
                    'Genre'].iloc[0])]['Probabilité_Topic'].between(Interval[0], Interval[1], inclusive=False)
    
    Recommand = Recommand.dropna(subset=['Probabilité_Topic'])
    Recommand[Recommand["Probabilité_Topic"]==True]
    print(Recommand["Titre"].sample(n=5))


'''
Partie du code qui a permis de trouver les listes de stopwords

import sklearn
from sklearn.cluster import KMeans

model = FastText.load_fasttext_format('C:/Users/nelso/OneDrive/Bureau/cc.en.300.bin')

overviewm=pd.DataFrame(overviewm)
overviewm['overview'] = overviewm.overview.apply(lambda x: ' '.join([str(i) for i in x]))


all_text = ' '.join([text for text in overviewm["overview"]])
token_text = word_tokenize(all_text)
token_text= Counter(token_text)
token_processed = {key:val for key, val in token_text.items() if val > 5}


# On vectorise les mots du dictionnaires token_processed
def get_vect_rep(token_processed):

    """ Getting vector representation for each word of the corpus
        Input:
            -- all_words
            
        Output:

            -- dict_vect: dict, containing the mapping between the words and their corresponding vector representation
    """

    # Init empty dictionary
    dict_vect = dict()
    no_rep = []

    for word in list(token_processed.keys()):
        try:
            dict_vect[word] = model[word]

        except KeyError:
            no_rep.append(word)

    return dict_vect, no_rep

dict_vect, no_rep = get_vect_rep(token_processed)
print(no_rep)

#['SWDB', 'Otík']

ssd = []
for k in range(1,16):
    print(k)
    km = KMeans(n_clusters=k)
    km = km.fit(list(dict_vect.values()))
    ssd.append(km.inertia_)

plt.plot(range(1,16), ssd, 'x-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#La méthode du coude monte un certain applainissement à partir de 6/7 clusters
best_k=6

# Nombre de clusters
kmeans = KMeans(n_clusters=best_k)
kmeans = kmeans.fit(list(dict_vect.values()))

# Les labels des clusters
labels = kmeans.predict(list(dict_vect.values()))

# Valeurs centroïdes
centroids = kmeans.cluster_centers_

# Dictionnaire Labels - Mots associés
tmp = list(zip(list(dict_vect.keys()), labels))
labeled_words = dict(tmp)


all_cat = [[] for i in range(best_k)]

for (word, label) in labeled_words.items():
    all_cat[label].append(word)

for i in range(len(all_cat)):
    print("Label:", str(i))
    sublist = all_cat[i]
    print("Nb of words:", len(sublist))
    print("50 words of this category:")
    print(sublist[:50])
    print("\n")
    
import pickle
Names=list(all_cat[0]) + list(all_cat[1]) + list(all_cat[2])
pickle.dump(Names, open( "C:/Users/nelso/OneDrive/Bureau/StopwordsNames.pkl", "wb" ))

Verbs = pickle.load(open( "C:/Users/nelso/OneDrive/Bureau/StopwordsVerbs.pkl", "rb" ))
'''
