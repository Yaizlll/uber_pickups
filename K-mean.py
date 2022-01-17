#import des librairies l'environnement
import streamlit as st
import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

st.title('Implémentation du clustering des fleurs d’Iris avec l’algorithme K-Means, Python et Scikit Learn')

#chargement de base de données iris
iris = datasets.load_iris()

#affichage des données, vous permet de mieux comprendre le jeu de données (optionnel)
#print(iris)
#print(iris.data)
#print(iris.feature_names)
#print(iris.target)
#print(iris.target_names)

#Stocker les données en tant que DataFrame Pandas
x=pd.DataFrame(iris.data)
# définir les noms de colonnes
x.columns=['Sepal_Length','Sepal_width','Petal_Length','Petal_width']

st.dataframe(x)

y=pd.DataFrame(iris.target)
y.columns=['Targets']


#Cluster K-means
nb_clust = st.slider("How many clusters?",min_value=2,max_value=10,value=2)
model=KMeans(n_clusters=nb_clust)
#adapter le modèle de données
model.fit(x)

st.write("My predicted lables")
st.write(model.labels_)

colormap = np.array((['Red','green','blue','Red','green','blue','Red','green','blue','Red','green','blue']))

fig,ax = plt.subplots()
plt.scatter(x.Petal_Length,x.Petal_width,c=colormap[y.Targets],s=40)
plt.scatter(x.Petal_Length,x.Petal_width,c=colormap[model.labels_],s=40)
st.pyplot(fig)