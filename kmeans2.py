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


#import data from an Excel file(.xlsx) with pandas
raw_dataset = pd.read_excel("TH in inpts for ANNE.xlsx", engine='openpyxl', sheet_name="Foglio1", usecols='C,D:AQ', header=1)
print(raw_dataset)
print("Dati caricati correttamente\n")

df_features = pd.read_excel("TH in inpts for ANNE.xlsx", engine='openpyxl', sheet_name="Foglio1", usecols='C,D:AQ', header=None, nrows=1, skiprows=[0])
features = (np.asarray(df_features)).flatten() #array
print(features)
size_features = features.shape[0]
print(size_features) #dim array
#print(features[0])
#size_features = len(features.columns)

#Data Exploration

#matrice di correlazione e heatmap
print(raw_dataset.corr())

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
sns.boxplot(data = raw_dataset, orient="v", whis=10)
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
print("La standardizzazione ha funzionato e adesso tutte le variabili risultano confrontabili")


#Kmeans
#define the model
kmeans_model = KMeans(n_clusters = 2)
#predict the labels of clusters
label = kmeans_model.fit_predict(scaled_dataframe) #o scaled_array
print("Etichette dei cluster")
print(label)

#add the Cluster column to the dataset
raw_dataset["Cluster for Kmeans"] = kmeans_model.labels_
scaled_dataframe["Cluster for Kmeans"] = kmeans_model.labels_
print(raw_dataset)

#ora separo i dati in base all'etichetta assegnata
cluster0_filter = scaled_dataframe["Cluster for Kmeans"]==0
print(scaled_dataframe[cluster0_filter])

cluster1_filter = scaled_dataframe["Cluster for Kmeans"]==1
print(scaled_dataframe[cluster1_filter])

#T-test e boxplot tra cluster0 e cluster1 per rilevare le feature pi?? significative

alpha = 0.05
for i in range (0,size_features,1):
    #Feature, cluster0 e cluster1
    t_value,p_value=stats.ttest_ind(scaled_dataframe[cluster0_filter][features[i]],
                                    scaled_dataframe[cluster1_filter][features[i]], equal_var=False)
    print(features[i])
    print("---------------------------------------------------------------------")
    print('Test statistic: %.6f' %(t_value))
    print('p-value: %f' %(p_value))

    if p_value<=alpha:
        print("Poich?? p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse\n")

    else:

        print("Poich?? p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali\n")

    filter0 = raw_dataset[cluster0_filter][features[i]]
    filter1 = raw_dataset[cluster1_filter][features[i]]
    median0 = raw_dataset[cluster0_filter][features[i]].median()
    median1 = raw_dataset[cluster1_filter][features[i]].median()
    shape0 = raw_dataset[cluster0_filter][features[i]].shape
    shape1 = raw_dataset[cluster1_filter][features[i]].shape

    '''print(raw_dataset[cluster0_filter][features[i]])
    print(raw_dataset[cluster1_filter]["Total cholesterol"])'''

    df=[filter0,filter1]
    sns.boxplot(data=df, orient="v", whis=10, showmeans=True)
    plt.title("Boxplot %s (Kmeans)\nP-value: %.6f" %(features[i], p_value))
    #plt.xlabel("Cluster")
    plt.xticks(np.arange(2), ('cluster 0\nN.patients: %d\nMedian: %.2f' %(shape0[0], median0), 'cluster 1\nN.patients: %d\nMedian: %.2f' %(shape1[0], median1)))
    plt.savefig("immagini/Kmeans2/%s.pdf" %(features[i]))
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
plt.savefig("immagini/Kmeans2/SilhoutteIndex.pdf")
#plt.show()
plt.close()




