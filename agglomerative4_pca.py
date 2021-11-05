import pandas as pd #for data management
import numpy as np #for data management
import seaborn as sns #for data visualization and specifically for pairplot()
import matplotlib.pyplot as plt #for data visualization
from sklearn.preprocessing import StandardScaler  #to transform the dataset
from scipy import stats
from sklearn.decomposition import PCA

#Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

#import data from an Excel file(.xlsx) with pandas
raw_dataset = pd.read_excel("TH in inpts for ANNE.xlsx", engine='openpyxl', sheet_name="Foglio1", usecols='C,D:AQ', header=1)
print(raw_dataset)
print("Dati caricati correttamente\n")

#Data Exploration

#matrice di correlazione e heatmap
print(raw_dataset.corr())
#plt.figure(figsize=( 6,3)  , dpi=200)
plt.figure(figsize = (15,6))
sns.heatmap( raw_dataset.corr(), annot=True)
plt.title("Heatmap")
plt.savefig("immagini/heatmap.pdf")
#plt.show()
plt.close()

#scatterplot
fig, (ax1, ax2, ax3) = plt.subplots( 1, 3, figsize = (16,5) )

sns.scatterplot(x="Total cholesterol", y="LDL", data = raw_dataset, ax = ax1, color = "#2ecc71", ).set_title("Altamente Correlati\nCorr: 0.85", fontsize = 14)
sns.scatterplot(x="HDL", y="Triglycerides", data = raw_dataset, ax = ax2, color = "#34495e").set_title("Inversamente Correlati\nCorr: -0.25", fontsize = 14)
sns.scatterplot(x="Height", y="History of dyslipidaemia", data = raw_dataset, ax = ax3, color = "#e74c3c").set_title("Scarsamente Correlati\nCorr: -0.0052", fontsize = 14)

plt.savefig("immagini/correlazioni.pdf")
#plt.show()
plt.close()


#boxplot
#plt.figure(figsize = (15,4))
sns.boxplot(data = raw_dataset, orient = "v", whis=10)
plt.title("Boxplot")
plt.savefig("immagini/boxplot.pdf")
#plt.show()
plt.close()

#controllo se ci sono dei valori mancanti nel dataset
print("Valori mancanti nel dataset:\n")
print(raw_dataset.isna().sum())

#data standardization (rendere confrontabili variabili di ordini di grandezza diversi attraverso un processo di standardizzazione)
scaler = StandardScaler()
scaled_array = scaler.fit_transform(raw_dataset)
#StandardScaler() return a numpy.ndarray (a matrix) that it can be inserted inside a dataframe for better management:
scaled_dataframe = pd.DataFrame( scaled_array, columns = raw_dataset.columns )

sns.boxplot(data = scaled_dataframe, orient = "v", whis=10)
plt.title("Boxplot data standardization")
plt.savefig("immagini/boxstd.pdf")
#plt.show()
plt.close()
print(scaled_dataframe.describe())
print("Tutte le variabili hanno media molto vicina allo 0 e deviazione standard vicina a 1\n")
print("La standardizzazione ha funzionato e adesso tutte le variabili risultano confrontabili\n")

#PCA (Principal Component Analysis)

pca = PCA(n_components=4)
principalComponents = pca.fit_transform(scaled_dataframe)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4'])
print(principalDf)
size_pca = principalDf.shape[1] #colonne


#varianza spiegata
print("Varianza spiegata")
print("---------------------------------------")
print(pca.explained_variance_ratio_)

#scatter plot

plt.scatter(x=principalDf["principal component 1"], y=principalDf["principal component 2"], c="r")
plt.title("2 component PCA")
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.savefig("immagini/Agglomerative4_pca/2pca.pdf")
plt.close()

#Agglomerative clustering
#uso il dendrogramma per stabilire il numero ottimale di cluster
dendrogram = sch.dendrogram(sch.linkage(principalDf, method='ward'))
plt.title("Dendrogramma")
plt.savefig("immagini/Agglomerative4_pca/Dendrogramma.pdf")
#plt.show()
plt.close()

model = AgglomerativeClustering(n_clusters=4)
model.fit(principalDf)
label = model.labels_
print(label)

#add the Cluster column to the dataset for Agglomerative Clustering
principalDf["Cluster for Agglomerative"] = label
print(principalDf)

#separo i dati in base all'etichetta assegnata
#ora separo i dati in base all'etichetta assegnata
cluster0_filter = principalDf["Cluster for Agglomerative"]==0
print(principalDf[cluster0_filter])

cluster1_filter = principalDf["Cluster for Agglomerative"]==1
print(principalDf[cluster1_filter])

cluster2_filter = principalDf["Cluster for Agglomerative"]==2
print(principalDf[cluster2_filter])

cluster3_filter = principalDf["Cluster for Agglomerative"]==3
print(principalDf[cluster3_filter])

# Kruskal-Wallis test e boxplot tra cluster0, cluster1, cluster2 e cluster3 per rilevare le feature più significative

#Features, cluster0, cluster1, cluster2 e cluster3
alpha = 0.05
for i in range (0,size_pca,1):
    filter0 = principalDf[cluster0_filter][principalDf.columns[i]]
    filter1 = principalDf[cluster1_filter][principalDf.columns[i]]
    filter2 = principalDf[cluster2_filter][principalDf.columns[i]]
    filter3 = principalDf[cluster3_filter][principalDf.columns[i]]

    median0 = principalDf[cluster0_filter][principalDf.columns[i]].median()
    median1 = principalDf[cluster1_filter][principalDf.columns[i]].median()
    median2 = principalDf[cluster2_filter][principalDf.columns[i]].median()
    median3 = principalDf[cluster3_filter][principalDf.columns[i]].median()

    shape0 = principalDf[cluster0_filter][principalDf.columns[i]].shape
    shape1 = principalDf[cluster1_filter][principalDf.columns[i]].shape
    shape2 = principalDf[cluster2_filter][principalDf.columns[i]].shape
    shape3 = principalDf[cluster3_filter][principalDf.columns[i]].shape

    #Kruskal-Wallis test (versione non parametrica di ANOVA)
    stat, pvalue = stats.kruskal(filter0, filter1, filter2, filter3)
    print("Kruskal-Wallis test for %s (versione non parametrica di ANOVA)" %(principalDf.columns[i]))
    print("--------------------------------------------------------------------")
    print("p-value: %.6f" %(pvalue))

    if pvalue <= alpha:
        print("Poiché p-value(=%f)" % pvalue, "<=", "alpha(=%.2f)" % alpha, "ci sono delle differenze tra le feature\n")

    else:

        print("Poichè p-value(=%f)" % pvalue, ">", "alpha(=%.2f)" % alpha, "le feature sono uguali\n")


    df=[filter0,filter1, filter2, filter3]
    sns.boxplot(data=df, orient="v", whis=10, showmeans=True)
    plt.title("Boxplot %s (Agglomerative)\nP-value: %.6f" % (principalDf.columns[i], pvalue))
    #plt.xlabel("Cluster")
    plt.xticks(np.arange(4), ('cluster 0\npatients: %d\nmed: %.1f' %(shape0[0], median0), 'cluster 1\npatients: %d\nmed: %.1f' %(shape1[0], median1), 'cluster 2\npatients: %d\nmed: %.1f' %(shape2[0], median2), 'cluster 3\npatients: %d\nmed: %.1f' %(shape3[0], median3)))
    plt.savefig("immagini/Agglomerative4_pca/%s.pdf" %(principalDf.columns[i]))
    #plt.show()
    plt.close()