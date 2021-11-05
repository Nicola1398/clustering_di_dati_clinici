import pandas as pd #for data management
import numpy as np #for data management
import pylab as pl
import seaborn as sns #for data visualization and specifically for pairplot()
import matplotlib.pyplot as plt #for data visualization
from sklearn.preprocessing import StandardScaler  #to transform the dataset
from scipy import stats

#Kmeans
from sklearn.cluster import KMeans #to instantiate, train and use model
from sklearn import metrics #for Model Evaluation


#Target (data di decesso - data di ricovero)

#import data from an Excel file(.xlsx) with pandas
raw_dataset_target = pd.read_excel("TH in inpts for ANNE - Target.xlsx", engine='openpyxl', sheet_name="Foglio1", usecols='C,D:AR', header=1)
print(raw_dataset_target)
print("Dati caricati correttamente\n")

#features + target
df_columns = pd.read_excel("TH in inpts for ANNE - Target.xlsx", engine='openpyxl', sheet_name="Foglio1", usecols='C,D:AR', header=None, nrows=1, skiprows=[0])
columns = (np.asarray(df_columns)).flatten() #array
print(columns)
size_columns = columns.shape[0]
print(size_columns) #dim array

#solo features
df_features = pd.read_excel("TH in inpts for ANNE - Target.xlsx", engine='openpyxl', sheet_name="Foglio1", usecols='C,D:AQ', header=None, nrows=1, skiprows=[0])
features = (np.asarray(df_features)).flatten() #array
print(features)
size_features = features.shape[0]
print(size_features) #dim array

#Data Exploration

#matrice di correlazione e heatmap
print(raw_dataset_target.corr())

plt.figure(figsize = (15,6))
sns.heatmap( raw_dataset_target.corr(), annot=True)
plt.title("Heatmap")
plt.savefig("immagini/heatmap_target.pdf")
#plt.show()
plt.close()

#scatterplot
fig, (ax1, ax2, ax3) = plt.subplots( 1, 3, figsize = (16,5) )

sns.scatterplot(x="Total cholesterol", y="LDL", data = raw_dataset_target, ax = ax1, color = "#2ecc71", ).set_title("Altamente Correlati\nCorr: 0.91", fontsize = 14)
sns.scatterplot(x="Primary Dilated Cardiomyopathy", y="CAD", data = raw_dataset_target, ax = ax2, color = "#34495e").set_title("Inversamente Correlati\nCorr: -0.40", fontsize = 14)
sns.scatterplot(x="Smoke History of smoke", y="HR", data = raw_dataset_target, ax = ax3, color = "#e74c3c").set_title("Scarsamente Correlati\nCorr: 0.0032", fontsize = 14)

plt.savefig("immagini/correlazioni_target.pdf")
#plt.show()
plt.close()

#boxplot
#plt.figure(figsize = (15,4))
sns.boxplot(data = raw_dataset_target, orient="v", whis=10)
plt.title("Boxplot")
plt.savefig("immagini/boxplot_target.pdf")
#plt.show()
plt.close()

#controllo se ci sono dei valori mancanti nel dataset
print("Valori mancanti nel dataset:\n")
print(raw_dataset_target.isna().sum())

#data standardization (rendere confrontabili variabili di ordini di grandezza diversi attraverso un processo di standardizzazione)
scaler = StandardScaler()
scaled_array = scaler.fit_transform(raw_dataset_target)
#StandardScaler() return a numpy.ndarray (a matrix) that it can be inserted inside a dataframe for better management:
scaled_dataframe = pd.DataFrame( scaled_array, columns = raw_dataset_target.columns )

sns.boxplot(data = scaled_dataframe, orient = "v", whis=10)
plt.title("Boxplot data standardization")
plt.savefig("immagini/boxstd_target.pdf")
#plt.show()
plt.close()
print(scaled_dataframe.describe())
print(scaled_dataframe)
print("Tutte le variabili hanno media molto vicina allo 0 e deviazione standard vicina a 1\n")
print("La standardizzazione ha funzionato e adesso tutte le variabili risultano confrontabili")

#separazione delle caratteristiche
only_features = scaled_dataframe[features]
print(only_features)
print(only_features.shape)
print(only_features.shape[1])

#separazione del target
target = scaled_dataframe["Time of life"]
print(target)

#Kmeans
#define the model
kmeans_model = KMeans(n_clusters = 2)
#predict the labels of clusters
label = kmeans_model.fit_predict(only_features)  #senza target
print("Etichette dei cluster")
print(label)

#add the Cluster column to the dataset
raw_dataset_target["Cluster for Kmeans"] = kmeans_model.labels_
scaled_dataframe["Cluster for Kmeans"] = kmeans_model.labels_
#only_features["Cluster for Kmeans"] = label
print(raw_dataset_target)
print(scaled_dataframe)
#print(only_features)

#ora separo i dati in base all'etichetta assegnata
cluster0_filter = scaled_dataframe["Cluster for Kmeans"]==0
print(scaled_dataframe[cluster0_filter])

cluster1_filter = scaled_dataframe["Cluster for Kmeans"]==1
print(scaled_dataframe[cluster1_filter])


#T-test e boxplot tra cluster0 e cluster1 per rilevare le feature più significative

alpha = 0.05
for i in range (0,size_columns,1):
    #Feature, cluster0 e cluster1
    t_value,p_value=stats.ttest_ind(scaled_dataframe[cluster0_filter][columns[i]],scaled_dataframe[cluster1_filter][columns[i]], equal_var=False)
    print(columns[i])
    print("---------------------------------------------------------------------")
    print('Test statistic: %.6f' %(t_value))
    print('p-value: %f' %(p_value))

    if p_value<=alpha:
        print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse\n")

    else:

        print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali\n")

    filter0 = raw_dataset_target[cluster0_filter][columns[i]]
    filter1 = raw_dataset_target[cluster1_filter][columns[i]]
    median0 = raw_dataset_target[cluster0_filter][columns[i]].median()
    median1 = raw_dataset_target[cluster1_filter][columns[i]].median()
    shape0 = raw_dataset_target[cluster0_filter][columns[i]].shape
    shape1 = raw_dataset_target[cluster1_filter][columns[i]].shape

    '''print(raw_dataset[cluster0_filter][features[i]])
    print(raw_dataset[cluster1_filter]["Total cholesterol"])'''

    df=[filter0,filter1]
    sns.boxplot(data=df, orient="v", whis=10, showmeans=True)
    plt.title("Boxplot %s (Kmeans)\nP-value: %.6f" %(columns[i], p_value))
    #plt.xlabel("Cluster")
    plt.xticks(np.arange(2), ('cluster 0\nN.patients: %d\nMedian: %.2f' %(shape0[0], median0), 'cluster 1\nN.patients: %d\nMedian: %.2f' %(shape1[0], median1)))
    plt.savefig("immagini/Kmeans2_target/%s.pdf" %(columns[i]))
    #plt.show()'''
    plt.close()


#search for the best K with silhoutte index
k_to_test = range(2, 30, 1)  # [2,3,4, ..., 29]
silhouette_scores = {}

#higher silhouette index, better clustering
for k in k_to_test:
    model_kmeans_k = KMeans(n_clusters=k)
    model_kmeans_k.fit(scaled_dataframe.drop("Cluster for Kmeans", axis=1))
    labels_k = model_kmeans_k.labels_
    score_k = metrics.silhouette_score(scaled_dataframe.drop("Cluster for Kmeans", axis=1), labels_k)
    silhouette_scores[k] = score_k
    print("Tested Kmeans with k = %d\tSS: %5.4f" % (k, score_k))

print("Trovato il miglior K")

#plot shows us the best K
#plt.figure(figsize = (16,5))
plt.plot(silhouette_scores.values())
plt.xticks(range(0,28,1), silhouette_scores.keys())
plt.title("Silhouette Metric")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.axvline(0, color = "r")
plt.savefig("immagini/Kmeans2_target/SilhoutteIndex.pdf")
#plt.show()
plt.close()






