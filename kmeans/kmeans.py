from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_iris

cards = pd.read_csv('Credit Card Customer Data.csv')
pd.set_option('display.max_columns', None)
le = LabelEncoder()
for col in cards.columns:
    cards[col] = le.fit_transform((cards[col]))

print(cards.head())

sse = []
for k in range(1,11):
    km = KMeans(n_clusters=k, random_state=2)
    km.fit(cards)
    sse.append(km.inertia_)

sns.set_style("whitegrid")
g = sns.lineplot(x=range(1,11), y=sse)
g.set(xlabel="Num of cluters (k)", ylabel="Sum of squared error", title="Elbow method")
plt.show()

kmeans = KMeans(n_clusters=3,random_state=2)
kmeans.fit(cards)

print("Cluter centers:",kmeans.cluster_centers_)