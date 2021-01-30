# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 17:41:27 2021

@author: nelso
"""

from src.libraries import *
from src.FunctionsLDA import *
stop = stopwords.words('english')
Path="C:/Users/nelso/OneDrive/Bureau/movie-recommender-system/"

df=pd.read_csv(Path+"datasets/movies_metadata.csv")
df=df.drop(["belongs_to_collection","homepage","tagline"], axis=1)
df=df.dropna()


overviewm=TreatOverview(df)

#On charge les listes de stopwords et on les enlève
overviewm=overviewm.apply(lambda x: word_tokenize(x))
overviewm=overviewm.apply(lambda x: [item for item in x if item not in stop])

Verbs = pickle.load(open( Path+ "Stopwords/StopwordsVerbs.pkl", "rb" ))
Names = pickle.load(open( Path+ "Stopwords/StopwordsNames.pkl", "rb" ))

overviewm=overviewm.apply(lambda x: [item for item in x if item not in [x.lower() for x in Verbs]
])
overviewm=overviewm.apply(lambda x: [item for item in x if item not in [x.lower() for x in Names]
])



#Topic modeling
overview = overviewm.tolist()

#LDA
# Création de dictionnaire
id2word = corpora.Dictionary(overview)
# Filtre fréquence
id2word.filter_extremes(no_below=10)
#Création de "bag of words"
corpus = [id2word.doc2bow(text) for text in overview]


#Find the optimal number of clusters
seq = range(1,21) # Choose the scale of topics you want
coherence_val=[]
for i in seq:
    perf_LDA(overview, id2word, corpus, i, coherence_val)
    
# Plot the coherence u_mass
plt.figure(figsize=(16,8))
plt.plot(coherence_val)
plt.title("Evolution of coherence (c_v) against the number of topics choosen in LDA")
plt.xlabel("Number of topics")
plt.xticks(np.arange(20), [i for i in range(1,21)])
plt.ylabel("coherence")
plt.show()

#Optimal Number of topic : 8 (sinon 16/17)
modele_lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=8,
                                           random_state=100,
                                           #update_every=1,
                                           #chunksize=10,
                                           #passes=10,
                                           alpha='auto')

model_topics = modele_lda.show_topics(8, formatted=False)

modele_lda.show_topics(8)

#Viz with wordclouds
Couleurs=['#F5F5DC','#000000','#FFEBCD','#0000FF','#A52A2A','#7FFF00','#D2691E', '#FFD700','#008000', '#ADFF2F',
          '#FFC0CB','#87CEEB','#6A5ACD','#708090','#FFFAFA','#00FF7F','#EE82EE','#FFFF00']

cloud = WordCloud(background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: Couleurs[i],
                  prefer_horizontal=1.0)

fig, axes = plt.subplots(3, 3, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    if i < 8:
        fig.add_subplot(ax)
        cloud.generate_from_frequencies(dict(model_topics[i][1]))
        plt.title("Topic Nb %d" %i)
        plt.gca().imshow(cloud)
        plt.gca().axis('off')
    else :
        pass
    
fig.delaxes(axes[-1,2])

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


#Extract movie genre in a JSON column
df["Index"]=df["genres"].str.find("\'name\': ").astype(int)
df["Index2"]=df["genres"].str.find("}").astype(int)
df["genre1"]=df["genres"].apply(lambda x : x[x.find("\'name\': ")+9:x.find("}")-1])

df["overview_clean"]=overviewm
df["overview_clean"]=df["overview_clean"].apply(lambda x : " ".join(x))


#Topic modeling clusters' probability extraction
LDADf=df[["original_title","genre1","overview_clean"]]

df_topic_sents_keywords = format_topics_sentences(modele_lda, corpus, LDADf)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Indice', 'Topic_Dominant', 'Probabilité_Topic', 'Keywords', 'Titre', "Genre","Overview"]

data_crosstab = pd.crosstab(df_dominant_topic['Topic_Dominant'],
                            df_dominant_topic['Genre'],
                               margins = False).apply(lambda r: round(r/r.sum(),2), axis=1)

maxValueIndex = data_crosstab.idxmax() 
  
#Highest cluster probability by movie genre
print("Topic ayant la probabilité la plus élevée pour chaque genre : ") 
print(maxValueIndex)


#Recommendation function
recommandationsystem(df_dominant_topic,"Toy Story")
