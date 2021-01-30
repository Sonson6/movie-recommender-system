# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:52:53 2021

@author: nelso
"""

#modules Python
import pandas as pd
import numpy as np
import unicodedata
import sys
from datetime import datetime
import re

#Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import seaborn as sns
import ipywidgets as widgets


#NLP Python
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = nltk.stem.WordNetLemmatizer()
wordnet_lemmatizer = WordNetLemmatizer()

import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.models.wrappers import FastText

import collections
from collections import Counter


from nltk.corpus import stopwords