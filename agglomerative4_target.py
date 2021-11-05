import pandas as pd #for data management
import numpy as np #for data management
import pylab as pl
import seaborn as sns #for data visualization and specifically for pairplot()
import matplotlib.pyplot as plt #for data visualization
from sklearn.preprocessing import StandardScaler  #to transform the dataset
from scipy import stats

#Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

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


#Agglomerative clustering
#uso il dendrogramma per stabilire il numero ottimale di cluster
dendrogram = sch.dendrogram(sch.linkage(only_features, method='ward'))
plt.title("Dendrogramma")
plt.savefig("immagini/Agglomerative4_target/Dendrogramma.pdf")
#plt.show()
plt.close()

model = AgglomerativeClustering(n_clusters=4)
model.fit(only_features) #senza target
label = model.labels_
print(label)

#add the Cluster column to the dataset for Agglomerative Clustering
raw_dataset_target["Cluster for Agglomerative"] = label
scaled_dataframe["Cluster for Agglomerative"] = label
print(raw_dataset_target)
print(scaled_dataframe)

#separo i dati in base all'etichetta assegnata
#ora separo i dati in base all'etichetta assegnata
cluster0_filter = scaled_dataframe["Cluster for Agglomerative"]==0
print(scaled_dataframe[cluster0_filter])

cluster1_filter = scaled_dataframe["Cluster for Agglomerative"]==1
print(scaled_dataframe[cluster1_filter])

cluster2_filter = scaled_dataframe["Cluster for Agglomerative"]==2
print(scaled_dataframe[cluster2_filter])

cluster3_filter = scaled_dataframe["Cluster for Agglomerative"]==3
print(scaled_dataframe[cluster3_filter])


# Kruskal-Wallis test e boxplot tra cluster0, cluster1, cluster2 e cluster3 per rilevare le feature più significative

#Features, cluster0, cluster1, cluster2 e cluster3
alpha = 0.05
for i in range (0,size_columns,1):
    filter0 = raw_dataset_target[cluster0_filter][columns[i]]
    filter1 = raw_dataset_target[cluster1_filter][columns[i]]
    filter2 = raw_dataset_target[cluster2_filter][columns[i]]
    filter3 = raw_dataset_target[cluster3_filter][columns[i]]

    median0 = raw_dataset_target[cluster0_filter][columns[i]].median()
    median1 = raw_dataset_target[cluster1_filter][columns[i]].median()
    median2 = raw_dataset_target[cluster2_filter][columns[i]].median()
    median3 = raw_dataset_target[cluster3_filter][columns[i]].median()

    shape0 = raw_dataset_target[cluster0_filter][columns[i]].shape
    shape1 = raw_dataset_target[cluster1_filter][columns[i]].shape
    shape2 = raw_dataset_target[cluster2_filter][columns[i]].shape
    shape3 = raw_dataset_target[cluster3_filter][columns[i]].shape

    #Kruskal-Wallis test (versione non parametrica di ANOVA)
    stat, pvalue = stats.kruskal(filter0, filter1, filter2, filter3)
    print("Kruskal-Wallis test for %s (versione non parametrica di ANOVA)" %(columns[i]))
    print("--------------------------------------------------------------------")
    print("p-value: %.6f" %(pvalue))

    if pvalue <= alpha:
        print("Poiché p-value(=%f)" % pvalue, "<=", "alpha(=%.2f)" % alpha, "ci sono delle differenze tra le feature\n")

    else:

        print("Poichè p-value(=%f)" % pvalue, ">", "alpha(=%.2f)" % alpha, "le feature sono uguali\n")


    df=[filter0,filter1, filter2, filter3]
    sns.boxplot(data=df, orient="v", whis=10, showmeans=True)
    plt.title("Boxplot %s (Agglomerative)\nP-value: %.6f" % (columns[i], pvalue))
    #plt.xlabel("Cluster")
    plt.xticks(np.arange(4), ('cluster 0\npatients: %d\nmed: %.1f' %(shape0[0], median0), 'cluster 1\npatients: %d\nmed: %.1f' %(shape1[0], median1), 'cluster 2\npatients: %d\nmed: %.1f' %(shape2[0], median2), 'cluster 3\npatients: %d\nmed: %.1f' %(shape3[0], median3)))
    plt.savefig("immagini/Agglomerative4_target/%s.pdf" %(columns[i]))
    #plt.show()
    plt.close()