# Movies Recommender System
Recommender systems are one of the most successful and widespread application of machine learning technologies in business. You can find large scale recommender systems in retail, video on demand, or music streaming.

# Algorithms implemented and evaluated
  * Content based filtering
  * Collaborative Filtering
    * Memory based collaborative filtering
      * User-Item Filtering
      * Item-Item Filtering
    * Model based collaborative filtering
      * Single Value Decomposition(SVD)
      * SVD++
  * Hybrid Model
      * Content Based + SVD 
  
  ## Files contained in the project
  * movie_recommendation_system.ipynb : python notebook code file
  * movie_recommendation_system.html : html version of python notebook
  * movies.csv : movies data from MovieLens dataset
  * ratings.csv : rating given by user to movie from MovieLens dataset 

# Improvment propositions and alternative movie recommendation system

  ## Libraries

  All libraries we used are in a requirements.txt file. Can be installed like this : 
  
  `pip install -R -u requirements.txt`

  Containing : _Pandas_, _Numpy_, _Seaborn_, _NLTK_, _Gensim_, _Wordcloud_ and others...

 ##  New directory tree :

   Here, we reorganized the initial project + we add our new features. It's cleaner and easier to read.

   ```
   |--- README.md
   |--- Research_paper.pdf
   |--- requirements.txt
   |--- LICENSE
   |--- datasets
   |     +-- movies.csv
   |     +-- ratings.csv
   |     +-- movies_metadata.csv
   |--- Stopwords
   |     +-- StopwordsNames.pkl
   |     +-- StopwordsVerbs.pkl
   |--- scripts
   |     +-- linux_propo.py
   |     +-- movie_recommendation_system.html
   |     +-- movie_recommendation_system.ipynb
   |--- src
   |     +-- libraries.py
   |     +-- FunctionsLDA.py
   ```
  
  ## Our recommendation movie system :

  * Topic modeling : Using LDA models (comparison with "coherence" metric to find the optimal number of clusters)
  * Analyse highest cluster's probability by movie
  * Creation of a movie recommendation function based on : Highest probability, genre and IMDb rating quantile

  ## Sample of our recommendation movie system :

  Let's say you just saw "Toy Story" and are looking for a similar movie to watch. Here is what's our system will purpose to you : 

  `recommandationsystem(df_dominant_topic,"Toy Story")`

  https://github.com/Sonson6/movie-recommender-system/blob/master/recommendation%20system.png
