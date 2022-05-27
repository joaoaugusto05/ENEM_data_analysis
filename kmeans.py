import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kmeans_lib import *
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Leio meu dataframe
df = pd.read_csv('MICRODADOS_ENEM_2019.csv', usecols = ['NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_CN', 'NU_NOTA_MT'], sep = ';', encoding = 'ISO-8859-1')


df = df[df['NU_NOTA_CH'].notna()]
df = df[df['NU_NOTA_CN'].notna()]
df = df[df['NU_NOTA_MT'].notna()]
X  = np.array(df)
'''''
k = 8


#Faço tratamento de dados retirando a coluna que nao quero

kmeans = KMeans(n_clusters= k, random_state= 0)
kmeans.fit(X)



model = Clustering(k)
model.fit(X, False)

print(f1_score(confusion_matrix(kmeans.labels_, model.labels_), k))

colors = {0:'red', 1:'blue', 2:'green', 3: 'yellow', 4: 'pink', 5: 'orange', 6: 'purple'}
plt.scatter(X[:,0],X[:,1], c=model.labels_, cmap='rainbow')
plt.show()


plt.scatter(X[:,0],X[:,1], c=model.labels_, cmap='rainbow')
plt.show()

'''''
total_errors = []
f1 = []
n = 8
plt.clf()
for i in range(n):
    f1_parcial = 0
    for j in range(5):
        kmeans = KMeans(n_clusters= i + 1, random_state= 0)
        kmeans.fit(X)

        model = Clustering(k = i +1)
        model.fit(X, False)
        f1_parcial += f1_score(confusion_matrix(kmeans.labels_, model.labels_), i + 1)
    f1.append(f1_parcial/5)
    #it_error = model.fit(X, True) 
    #total_errors.append(it_error)

plt.plot(f1)
plt.show()

plt.clf()

plt.xticks(np.arange(len(total_errors)), np.arange(1, len(total_errors)+1))
plt.plot(total_errors)
plt.show()
#Printo meu resultado
