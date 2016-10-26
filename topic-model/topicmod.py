import json

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.stem.snowball import SnowballStemmer

n_samples = 4
n_features = 5
n_topics = 4
n_top_words = 3

stemmer = SnowballStemmer("english", ignore_stopwords=True)
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    d = (stemmer.stem(w) for w in analyzer(doc))
    return d

with open('toy.json') as df:    
    data = json.load(df)

corpus = []
for e in data:
    corpus.append(e["doc"])

# NMF
# tfidf_vectorizer = TfidfVectorizer(max_df=1, min_df=0,
#                                    max_features=n_features,
#                                    stop_words='english',
#                                    analyzer = stemmed_words)
# tfidf = tfidf_vectorizer.fit_transform(corpus)

# nmf = NMF(n_components=n_topics, random_state=1,
#           alpha=.1, l1_ratio=.5).fit(tfidf)

# tfidf_feature_names = tfidf_vectorizer.get_feature_names()
# print(tfidf_feature_names)

# for topic_idx, topic in enumerate(nmf.components_):
#     print("Topic #%d:" % topic_idx)
#     print(" ".join([tfidf_feature_names[i]
#                     for i in topic.argsort()[:-n_top_words - 1:-1]]))

# LDA
tf_vectorizer = CountVectorizer(max_features=n_features,
                                stop_words='english'
                                )
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()
print tf_feature_names

lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=10,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

model = lda.fit(tf)

for topic_idx, topic in enumerate(model.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([tf_feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))

print("----------- Topic classification --------------")
for doc in corpus:
    print(model.transform(tf_vectorizer.transform((doc, ))))
print("-----------------------------------------------")