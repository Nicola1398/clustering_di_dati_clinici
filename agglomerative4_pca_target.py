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
print(scaled_dataframe)
print(scaled_dataframe.describe())
print("Tutte le variabili hanno media molto vicina allo 0 e deviazione standard vicina a 1\n")
print("La standardizzazione ha funzionato e adesso tutte le variabili risultano confrontabili")

#separazione delle caratteristiche
#only_features = scaled_dataframe.loc[:, features].values
only_features = scaled_dataframe[features]
print(only_features)

#separazione del target
#target = scaled_dataframe.loc[:,["Time of life"]].values
target = scaled_dataframe["Time of life"]
print(target)

#PCA (Principal Component Analysis)

pca = PCA(n_components=4)
principalComponents = pca.fit_transform(only_features)
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
plt.savefig("immagini/Agglomerative4_pca_target/2pca.pdf")
plt.close()

#Agglomerative clustering
#uso il dendrogramma per stabilire il numero ottimale di cluster
dendrogram = sch.dendrogram(sch.linkage(principalDf, method='ward'))
plt.title("Dendrogramma")
plt.savefig("immagini/Agglomerative4_pca_target/Dendrogramma.pdf")
#plt.show()
plt.close()

model = AgglomerativeClustering(n_clusters=4)
model.fit(principalDf)
label = model.labels_
print(label)

#concatenazione del dataframe finale (features + target)
finalDf = pd.concat([principalDf, target], axis = 1)
print(finalDf)
size_finalDF = finalDf.shape[1] #colonne

#add the Cluster column to the dataset for Agglomerative Clustering
finalDf["Cluster for Agglomerative"] = label
print(finalDf)

#separo i dati in base all'etichetta assegnata
#ora separo i dati in base all'etichetta assegnata
cluster0_filter = finalDf["Cluster for Agglomerative"]==0
print(finalDf[cluster0_filter])

cluster1_filter = finalDf["Cluster for Agglomerative"]==1
print(finalDf[cluster1_filter])

cluster2_filter = finalDf["Cluster for Agglomerative"]==2
print(finalDf[cluster2_filter])

cluster3_filter = finalDf["Cluster for Agglomerative"]==3
print(finalDf[cluster3_filter])

# Kruskal-Wallis test e boxplot tra cluster0, cluster1, cluster2 e cluster3 per rilevare le feature più significative

#Features, cluster0, cluster1, cluster2 e cluster3
alpha = 0.05
for i in range (0,size_finalDF-1,1):
    filter0 = finalDf[cluster0_filter][finalDf.columns[i]]
    filter1 = finalDf[cluster1_filter][finalDf.columns[i]]
    filter2 = finalDf[cluster2_filter][finalDf.columns[i]]
    filter3 = finalDf[cluster3_filter][finalDf.columns[i]]

    median0 = finalDf[cluster0_filter][finalDf.columns[i]].median()
    median1 = finalDf[cluster1_filter][finalDf.columns[i]].median()
    median2 = finalDf[cluster2_filter][finalDf.columns[i]].median()
    median3 = finalDf[cluster3_filter][finalDf.columns[i]].median()

    shape0 = finalDf[cluster0_filter][finalDf.columns[i]].shape
    shape1 = finalDf[cluster1_filter][finalDf.columns[i]].shape
    shape2 = finalDf[cluster2_filter][finalDf.columns[i]].shape
    shape3 = finalDf[cluster3_filter][finalDf.columns[i]].shape

    #Kruskal-Wallis test (versione non parametrica di ANOVA)
    stat, pvalue = stats.kruskal(filter0, filter1, filter2, filter3)
    print("Kruskal-Wallis test for %s (versione non parametrica di ANOVA)" %(finalDf.columns[i]))
    print("--------------------------------------------------------------------")
    print("p-value: %.6f" %(pvalue))

    if pvalue <= alpha:
        print("Poiché p-value(=%f)" % pvalue, "<=", "alpha(=%.2f)" % alpha, "ci sono delle differenze tra le feature\n")

    else:

        print("Poichè p-value(=%f)" % pvalue, ">", "alpha(=%.2f)" % alpha, "le feature sono uguali\n")


    df=[filter0,filter1, filter2, filter3]
    sns.boxplot(data=df, orient="v", whis=10, showmeans=True)
    plt.title("Boxplot %s (Agglomerative)\nP-value: %.6f" % (finalDf.columns[i], pvalue))
    #plt.xlabel("Cluster")
    plt.xticks(np.arange(4), ('cluster 0\npatients: %d\nmed: %.1f' %(shape0[0], median0), 'cluster 1\npatients: %d\nmed: %.1f' %(shape1[0], median1), 'cluster 2\npatients: %d\nmed: %.1f' %(shape2[0], median2), 'cluster 3\npatients: %d\nmed: %.1f' %(shape3[0], median3)))
    plt.savefig("immagini/Agglomerative4_pca_target/%s.pdf" %(finalDf.columns[i]))
    #plt.show()
    plt.close()

#Time of life, cluster0, cluster1, cluster2 e cluster3
filter0 = raw_dataset_target[cluster0_filter]["Time of life"]
filter1 = raw_dataset_target[cluster1_filter]["Time of life"]
filter2 = raw_dataset_target[cluster2_filter]["Time of life"]
filter3 = raw_dataset_target[cluster3_filter]["Time of life"]

median0 = raw_dataset_target[cluster0_filter]["Time of life"].median()
median1 = raw_dataset_target[cluster1_filter]["Time of life"].median()
median2 = raw_dataset_target[cluster2_filter]["Time of life"].median()
median3 = raw_dataset_target[cluster3_filter]["Time of life"].median()

shape0 = raw_dataset_target[cluster0_filter]["Time of life"].shape
shape1 = raw_dataset_target[cluster1_filter]["Time of life"].shape
shape2 = raw_dataset_target[cluster2_filter]["Time of life"].shape
shape3 = raw_dataset_target[cluster3_filter]["Time of life"].shape

#Kruskal-Wallis test (versione non parametrica di ANOVA)
stat, pvalue = stats.kruskal(filter0, filter1, filter2, filter3)
print("Kruskal-Wallis test for Time of life (versione non parametrica di ANOVA)")
print("--------------------------------------------------------------------")
print("p-value: %.6f" %(pvalue))

if pvalue <= alpha:
    print("Poiché p-value(=%f)" % pvalue, "<=", "alpha(=%.2f)" % alpha, "ci sono delle differenze tra le feature\n")

else:

    print("Poichè p-value(=%f)" % pvalue, ">", "alpha(=%.2f)" % alpha, "le feature sono uguali\n")


df=[filter0,filter1, filter2, filter3]
sns.boxplot(data=df, orient="v", whis=10, showmeans=True)
plt.title("Boxplot Time of life (Agglomerative)\nP-value: %.6f" % (pvalue))
#plt.xlabel("Cluster")
plt.xticks(np.arange(4), ('cluster 0\npatients: %d\nmed: %.1f' %(shape0[0], median0), 'cluster 1\npatients: %d\nmed: %.1f' %(shape1[0], median1), 'cluster 2\npatients: %d\nmed: %.1f' %(shape2[0], median2), 'cluster 3\npatients: %d\nmed: %.1f' %(shape3[0], median3)))
plt.savefig("immagini/Agglomerative4_pca_target/Time of life.pdf")
#plt.show()
plt.close()

#T-test e boxplot tra cluster0 e cluster2 per rilevare le feature più significative
print("Analizzo le feature sui pazienti dei due cluster più diversi\n")
alpha = 0.05
for i in range (0,size_columns,1):
    #Feature, cluster0 e cluster2
    t_value,p_value=stats.ttest_ind(scaled_dataframe[cluster0_filter][columns[i]],scaled_dataframe[cluster2_filter][columns[i]], equal_var=False)
    print(columns[i])
    print("---------------------------------------------------------------------")
    print('Test statistic: %.6f' %(t_value))
    print('p-value: %f' %(p_value))

    if p_value<=alpha:
        print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse\n")

    else:

        print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali\n")

    filter0 = raw_dataset_target[cluster0_filter][columns[i]]
    filter2 = raw_dataset_target[cluster2_filter][columns[i]]
    median0 = raw_dataset_target[cluster0_filter][columns[i]].median()
    median2 = raw_dataset_target[cluster2_filter][columns[i]].median()
    shape0 = raw_dataset_target[cluster0_filter][columns[i]].shape
    shape2 = raw_dataset_target[cluster2_filter][columns[i]].shape

    df=[filter0,filter2]
    sns.boxplot(data=df, orient="v", whis=10, showmeans=True)
    plt.title("Boxplot %s (Agglomerative)\nP-value: %.6f" %(columns[i], p_value))
    #plt.xlabel("Cluster")
    plt.xticks(np.arange(2), ('cluster 0\nN.patients: %d\nMedian: %.2f' %(shape0[0], median0), 'cluster 2\nN.patients: %d\nMedian: %.2f' %(shape2[0], median2)))
    plt.savefig("immagini/Agglomerative4_pca_target/T-test/%s.pdf" %(columns[i]))
    #plt.show()'''
    plt.close()