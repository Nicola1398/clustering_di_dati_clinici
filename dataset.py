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

#DBSCAN
from sklearn.cluster import DBSCAN #to instantiate and fit the model
from sklearn.metrics import pairwise_distances #for Model evaluation
from sklearn.neighbors import NearestNeighbors #for Hyperparameter Tuning

from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

#Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

#import data from an Excel file(.xlsx) with pandas
raw_dataset = pd.read_excel("TH in inpts for ANNE.xlsx", engine='openpyxl', sheet_name="Foglio1", usecols='C,D:AQ', header=1)
print(raw_dataset)
print("Dati caricati correttamente\n")

'''print(raw_dataset.keys())
print(raw_dataset.head())
print("prova")'''

#Data Exploration

#matrice di correlazione e heatmap
print(raw_dataset.corr())

plt.figure(figsize = (15,6))
sns.heatmap( raw_dataset.corr(), annot=True)
plt.title("Heatmap")
#plt.savefig("immagini/heatmap.pdf")
plt.show()

#scatterplot
fig, (ax1, ax2, ax3) = plt.subplots( 1, 3, figsize = (16,5) )

sns.scatterplot(x="Total cholesterol", y="LDL", data = raw_dataset, ax = ax1, color = "#2ecc71", ).set_title("Altamente Correlati\nCorr: 0.85", fontsize = 14)
sns.scatterplot(x="HDL", y="Triglycerides", data = raw_dataset, ax = ax2, color = "#34495e").set_title("Inversamente Correlati\nCorr: -0.25", fontsize = 14)
sns.scatterplot(x="Height", y="History of dyslipidaemia", data = raw_dataset, ax = ax3, color = "#e74c3c").set_title("Scarsamente Correlati\nCorr: -0.0052", fontsize = 14)

#plt.savefig("immagini/correlazioni.pdf")
plt.show()

#pairplot
'''sns.pairplot(raw_dataset, palette = "Accent")
plt.show()'''

#boxplot
#plt.figure(figsize = (15,4))
sns.boxplot(data = raw_dataset, orient = "v")
plt.title("Boxplot")
#plt.savefig("immagini/boxplot.pdf")
plt.show()

#controllo se ci sono dei valori mancanti nel dataset
print("Valori mancanti nel dataset:\n")
print(raw_dataset.isna().sum())

#data standardization (rendere confrontabili variabili di ordini di grandezza diversi attraverso un processo di standardizzazione)
scaler = StandardScaler()
scaled_array = scaler.fit_transform(raw_dataset)
#StandardScaler() return a numpy.ndarray (a matrix) that it can be inserted inside a dataframe for better management:
scaled_dataframe = pd.DataFrame( scaled_array, columns = raw_dataset.columns )

sns.boxplot(data = scaled_dataframe, orient = "v")
plt.title("Boxplot data standardization")
#plt.savefig("immagini/boxstd.pdf")
plt.show()
print(scaled_dataframe.describe())
print("Tutte le variabili hanno media molto vicina allo 0 e deviazione standard vicina a 1\n")
print("La standardizzazione ha funzionato e adesso tutte le variabili risultano confrontabili")


#Kmeans
#define the model
kmeans_model = KMeans(n_clusters = 2)
#predict the labels of clusters
label = kmeans_model.fit_predict(scaled_array) #o scaled_dataframe
print("Etichette dei cluster")
print(label)



#Cluster 0
# filter rows of original data
'''filtered_label0 = scaled_array[label == 0]
# plotting the results
plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1], color="blue")
plt.title("Cluster 0")
plt.show()
print(filtered_label0)

#Cluster 1
# filter rows of original data
filtered_label1 = scaled_array[label == 1]
# plotting the results
plt.scatter(filtered_label1[:, 0], filtered_label1[:, 1], color="orange")
pl.title("Cluster 1")
plt.show()

#getting unique labels
u_labels = np.unique(label)
#plotting the results:
for i in u_labels:
    plt.scatter(scaled_array[label == i , 0] , scaled_array[label == i , 1] , label = i)
plt.title("Kmeans")
plt.legend()
plt.show()
print(kmeans_model.cluster_centers_.shape) #siamo in uno spazio R^41 (dove 41 è il numero di colonne del dataset) quindi i nostri punti (centroidi) avranno 41 coordinate

#getting the Centroids
centroids = kmeans_model.cluster_centers_
u_labels = np.unique(label)

#plotting the results:
for i in u_labels:
    plt.scatter(scaled_array[label == i, 0], scaled_array[label == i, 1], label=i)
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
plt.title("Centroids")
plt.legend()
plt.show()'''

#add the Cluster column to the dataset
raw_dataset["Cluster for Kmeans"] = kmeans_model.labels_
scaled_dataframe["Cluster for Kmeans"] = kmeans_model.labels_
print(raw_dataset)

#sotto dataset contenente solo la colonna 'Cluster for Kmeans' relativa all'etichetta assegnata ad ogni dato
#cluster = pd.DataFrame( scaled_dataframe, columns=["Cluster for Kmeans"])
#print(cluster)

#ora separo i dati in base all'etichetta assegnata
cluster0_filter = raw_dataset["Cluster for Kmeans"]==0
print(raw_dataset[cluster0_filter])

cluster1_filter = raw_dataset["Cluster for Kmeans"]==1
print(raw_dataset[cluster1_filter])

#boxplot a confronto tra cluster0 e cluster1 per rilevare le feature più significative
#Total cholesterol, cluster0 e cluster1
TotChol0_filter = raw_dataset[cluster0_filter]["Total cholesterol"]
#print(raw_dataset[cluster0_filter]["Total cholesterol"])
TotChol1_filter = raw_dataset[cluster1_filter]["Total cholesterol"]
#print(raw_dataset[cluster1_filter]["Total cholesterol"])

df_TotChol=[TotChol0_filter,TotChol1_filter]
sns.boxplot(data=df_TotChol, orient="v")
plt.title("Boxplot Total Cholesterol\nKmeans")
plt.xlabel("Cluster")
#plt.savefig("immagini/Kmeans/TotCholKmeans.pdf")
plt.show()

#HDL, cluster0 e cluster1
HDL0_filter = raw_dataset[cluster0_filter]["HDL"]
HDL1_filter = raw_dataset[cluster1_filter]["HDL"]

df_HDL=[HDL0_filter,HDL1_filter]
sns.boxplot(data=df_HDL, orient="v")
plt.title("Boxplot HDL\nKmeans")
plt.xlabel("Cluster")
#plt.savefig("immagini/Kmeans/HDLKmeans.pdf")
plt.show()

#LDL, cluster0 e cluster1
LDL0_filter = raw_dataset[cluster0_filter]["LDL"]
LDL1_filter = raw_dataset[cluster1_filter]["LDL"]

df_LDL=[LDL0_filter,LDL1_filter]
sns.boxplot(data=df_LDL, orient="v")
plt.title("Boxplot LDL\nKmeans")
plt.xlabel("Cluster")
#plt.savefig("immagini/Kmeans/LDLKmeans.pdf")
plt.show()

#Triglycerides, cluster0 e cluster1
Triglycerides0_filter = raw_dataset[cluster0_filter]["Triglycerides"]
Triglycerides1_filter = raw_dataset[cluster1_filter]["Triglycerides"]

df_Triglycerides=[Triglycerides0_filter,Triglycerides1_filter]
sns.boxplot(data=df_Triglycerides, orient="v")
plt.title("Boxplot Triglycerides\nKmeans")
plt.xlabel("Cluster")
#plt.savefig("immagini/Kmeans/TriglyKmeans.pdf")
plt.show()

#Glycemia, cluster0 e cluster1
Glycemia0_filter = raw_dataset[cluster0_filter]["Glycemia"]
Glycemia1_filter = raw_dataset[cluster1_filter]["Glycemia"]

df_Glycemia=[Glycemia0_filter,Glycemia1_filter]
sns.boxplot(data=df_Glycemia, orient="v")
plt.title("Boxplot Glycemia\nKmeans")
plt.xlabel("Cluster")
#plt.savefig("immagini/Kmeans/GlycemiaKmeans.pdf")
plt.show()

#Age, cluster0 e cluster1
Age0_filter = raw_dataset[cluster0_filter]["Age (1)"]
Age1_filter = raw_dataset[cluster1_filter]["Age (1)"]

df_Age=[Age0_filter,Age1_filter]
sns.boxplot(data=df_Age, orient="v")
plt.title("Boxplot Age\nKmeans")
plt.xlabel("Cluster")
#plt.savefig("immagini/Kmeans/AgeKmeans.pdf")
plt.show()

#HR, cluster0 e cluster1
HR0_filter = raw_dataset[cluster0_filter]["HR"]
HR1_filter = raw_dataset[cluster1_filter]["HR"]

df_HR=[HR0_filter,HR1_filter]
sns.boxplot(data=df_HR, orient="v")
plt.title("Boxplot HR\nKmeans")
plt.xlabel("Cluster")
#plt.savefig("immagini/Kmeans/HRKmeans.pdf")
plt.show()

#Weight, cluster0 e cluster1
Weight0_filter = raw_dataset[cluster0_filter]["Weight"]
Weight1_filter = raw_dataset[cluster1_filter]["Weight"]

df_Weight=[Weight0_filter,Weight1_filter]
sns.boxplot(data=df_Weight, orient="v")
plt.title("Boxplot Weight\nKmeans")
plt.xlabel("Cluster")
#plt.savefig("immagini/Kmeans/WeightKmeans.pdf")
plt.show()

#Height, cluster0 e cluster1
Height0_filter = raw_dataset[cluster0_filter]["Height"]
Height1_filter = raw_dataset[cluster1_filter]["Height"]

df_Height=[Height0_filter,Height1_filter]
sns.boxplot(data=df_Height, orient="v")
plt.title("Boxplot Height\nKmeans")
plt.xlabel("Cluster")
#plt.savefig("immagini/Kmeans/HeightKmeans.pdf")
plt.show()

#BMI, cluster0 e cluster1
BMI0_filter = raw_dataset[cluster0_filter]["BMI"]
BMI1_filter = raw_dataset[cluster1_filter]["BMI"]

df_BMI=[BMI0_filter,BMI1_filter]
sns.boxplot(data=df_BMI, orient="v")
plt.title("Boxplot BMI\nKmeans")
plt.xlabel("Cluster")
#plt.savefig("immagini/Kmeans/BMIKmeans.pdf")
plt.show()

#Diastolic blood pressure, cluster0 e cluster1
Diastolic_blood_pressure0_filter = raw_dataset[cluster0_filter]["Diastolic blood pressure"]
Diastolic_blood_pressure1_filter = raw_dataset[cluster1_filter]["Diastolic blood pressure"]

df_Diastolic_blood_pressure=[Diastolic_blood_pressure0_filter,Diastolic_blood_pressure1_filter]
sns.boxplot(data=df_Diastolic_blood_pressure, orient="v")
plt.title("Boxplot Diastolic blood pressure\nKmeans")
plt.xlabel("Cluster")
#plt.savefig("immagini/Kmeans/DiastolicBPKmeans.pdf")
plt.show()

#Systolic blood pressure, cluster0 e cluster1
Systolic_blood_pressure0_filter = raw_dataset[cluster0_filter]["Systolic blood pressure"]
Systolic_blood_pressure1_filter = raw_dataset[cluster1_filter]["Systolic blood pressure"]

df_Systolic_blood_pressure=[Systolic_blood_pressure0_filter,Systolic_blood_pressure1_filter]
sns.boxplot(data=df_Systolic_blood_pressure, orient="v")
plt.title("Boxplot Systolic blood pressure\nKmeans")
plt.xlabel("Cluster")
#plt.savefig("immagini/Kmeans/SystolicBPKmeans.pdf")
plt.show()

#T-test per rilevare le feature più significative
print("T-test")
#Total cholesterol, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Total cholesterol"],raw_dataset[cluster1_filter]["Total cholesterol"], equal_var=False)
print('Test statistic for Total cholesterol: %f'%float("{:.6f}".format(t_value)))
print('p-value for Total cholesterol: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#HDL, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["HDL"],raw_dataset[cluster1_filter]["HDL"], equal_var=False)
print('Test statistic for HDL: %f'%float("{:.6f}".format(t_value)))
print('p-value for HDL: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#LDL, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["LDL"],raw_dataset[cluster1_filter]["LDL"], equal_var=False)
print('Test statistic for LDL: %f'%float("{:.6f}".format(t_value)))
print('p-value for LDL: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Triglycerides, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Triglycerides"],raw_dataset[cluster1_filter]["Triglycerides"], equal_var=False)
print('Test statistic for Triglycerides: %f'%float("{:.6f}".format(t_value)))
print('p-value for Triglycerides: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Glycemia, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Glycemia"],raw_dataset[cluster1_filter]["Glycemia"], equal_var=False)
print('Test statistic for Glycemia: %f'%float("{:.6f}".format(t_value)))
print('p-value for Glycemia: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Age, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Age (1)"],raw_dataset[cluster1_filter]["Age (1)"], equal_var=False)
print('Test statistic for Age: %f'%float("{:.6f}".format(t_value)))
print('p-value for Age: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#HR, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["HR"],raw_dataset[cluster1_filter]["HR"], equal_var=False)
print('Test statistic for HR: %f'%float("{:.6f}".format(t_value)))
print('p-value for HR: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Weight, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Weight"],raw_dataset[cluster1_filter]["Weight"], equal_var=False)
print('Test statistic for Weight: %f'%float("{:.6f}".format(t_value)))
print('p-value for Weight: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Height, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Height"],raw_dataset[cluster1_filter]["Height"], equal_var=False)
print('Test statistic for Height: %f'%float("{:.6f}".format(t_value)))
print('p-value for Height: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#BMI, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["BMI"],raw_dataset[cluster1_filter]["BMI"], equal_var=False)
print('Test statistic for BMI: %f'%float("{:.6f}".format(t_value)))
print('p-value for BMI: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Diastolic blood pressure, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Diastolic blood pressure"],raw_dataset[cluster1_filter]["Diastolic blood pressure"], equal_var=False)
print('Test statistic for Diastolic blood pressure: %f'%float("{:.6f}".format(t_value)))
print('p-value for Diastolic blood pressure: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Systolic blood pressure, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Systolic blood pressure"],raw_dataset[cluster1_filter]["Systolic blood pressure"], equal_var=False)
print('Test statistic for Systolic blood pressure: %f'%float("{:.6f}".format(t_value)))
print('p-value for Systolic blood pressure: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")



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
#plt.savefig("immagini/Kmeans/SilhoutteIndex.pdf")
plt.show()

#plot delle feature più rilevanti
'''plt.plot(raw_dataset[cluster0_filter]["Total cholesterol"], raw_dataset[cluster1_filter]["Total cholesterol"], alpha=0.5)
plt.xlabel("Total cholesterol cluster0")
plt.ylabel("Total choletserol cluster1");
plt.show()'''


'''plt.plot(raw_dataset[cluster0_filter]["Total cholesterol"], label = 'cluster 0', color = 'b')
plt.plot(raw_dataset[cluster1_filter]["Total cholesterol"], label = 'cluster 1', color = 'r')
plt.title("Total cholesterol\nKmeans")
plt.ylabel("Total cholesterol")
plt.legend()
plt.show()'''


#alcuni confronti
'''fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(5, 2, figsize = (16,10))

sns.scatterplot(x = "Total cholesterol", y = "LDL", data = raw_dataset, hue = "Cluster for Kmeans", palette = "Accent", ax = ax1, legend=True)
sns.scatterplot(x = "Triglycerides", y = "Glycemia", data = raw_dataset, hue = "Cluster for Kmeans", palette = "Accent",ax = ax2, legend=True)
sns.scatterplot(x = "HDL", y = "fT3", data = raw_dataset, hue = "Cluster for Kmeans", palette = "Accent", ax = ax3, legend=True)
sns.scatterplot(x = "Weight", y = "Total cholesterol", data = raw_dataset, hue = "Cluster for Kmeans", palette = "Accent",ax = ax4, legend=True)
#plt.tight_layout()
plt.show()'''


#altra implementazione di Kmeans
'''from numpy import unique, where
# define the model
model = KMeans(n_clusters=2)
# fit the model
model.fit(scaled_dataframe)
# assign a cluster to each example
label = model.predict(scaled_dataframe)
# retrieve unique clusters
clusters = unique(label)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(label == cluster)
	# create scatter of these samples
	plt.scatter(scaled_array[row_ix, 0], scaled_array[row_ix, 1])
print("\nGrafico a dispersione con punti colorati in base al cluster assegnato.")
# show the plot
plt.title("Kmeans")
plt.show()'''


#Agglomerative clustering
#uso il dendrogramma per stabilire il numero ottimale di cluster
dendrogram = sch.dendrogram(sch.linkage(scaled_dataframe, method='ward'))
plt.title("Dendrogramma")
#plt.savefig("immagini/Agglomerative/Dendrogramma.pdf")
plt.show()

model = AgglomerativeClustering(n_clusters=2)
model.fit(scaled_dataframe)
label = model.labels_
print(label)

#add the Cluster column to the dataset for Agglomerative Clustering
raw_dataset["Cluster for Agglomerative"] = label
scaled_dataframe["Cluster for Agglomerative"] = label
print(raw_dataset)

#separo i dati in base all'etichetta assegnata
#ora separo i dati in base all'etichetta assegnata
cluster0_filter = raw_dataset["Cluster for Agglomerative"]==0
print(raw_dataset[cluster0_filter])

cluster1_filter = raw_dataset["Cluster for Agglomerative"]==1
print(raw_dataset[cluster1_filter])

#boxplot a confronto tra cluster0 e cluster1 per rilevare le feature più significative
#Total cholesterol, cluster0 e cluster1
TotChol0_filter = raw_dataset[cluster0_filter]["Total cholesterol"]
#print(raw_dataset[cluster0_filter]["Total cholesterol"])
TotChol1_filter = raw_dataset[cluster1_filter]["Total cholesterol"]
#print(raw_dataset[cluster1_filter]["Total cholesterol"])

df_TotChol=[TotChol0_filter,TotChol1_filter]
sns.boxplot(data=df_TotChol, orient="v")
plt.title("Boxplot Total Cholesterol\nAgglomerative")
plt.xlabel("Cluster")
#plt.savefig("immagini/Agglomerative/TotCholAgglomerative.pdf")
plt.show()

#HDL, cluster0 e cluster1
HDL0_filter = raw_dataset[cluster0_filter]["HDL"]
HDL1_filter = raw_dataset[cluster1_filter]["HDL"]

df_HDL=[HDL0_filter,HDL1_filter]
sns.boxplot(data=df_HDL, orient="v")
plt.title("Boxplot HDL\nAgglomerative")
plt.xlabel("Cluster")
#plt.savefig("immagini/Agglomerative/HDLAgglomerative.pdf")
plt.show()

#LDL, cluster0 e cluster1
LDL0_filter = raw_dataset[cluster0_filter]["LDL"]
LDL1_filter = raw_dataset[cluster1_filter]["LDL"]

df_LDL=[LDL0_filter,LDL1_filter]
sns.boxplot(data=df_LDL, orient="v")
plt.title("Boxplot LDL\nAgglomerative")
plt.xlabel("Cluster")
#plt.savefig("immagini/Agglomerative/LDLAgglomerative.pdf")
plt.show()

#Triglycerides, cluster0 e cluster1
Triglycerides0_filter = raw_dataset[cluster0_filter]["Triglycerides"]
Triglycerides1_filter = raw_dataset[cluster1_filter]["Triglycerides"]

df_Triglycerides=[Triglycerides0_filter,Triglycerides1_filter]
sns.boxplot(data=df_Triglycerides, orient="v")
plt.title("Boxplot Triglycerides\nAgglomerative")
plt.xlabel("Cluster")
#plt.savefig("immagini/Agglomerative/TriglyAgglomerative.pdf")
plt.show()

#Glycemia, cluster0 e cluster1
Glycemia0_filter = raw_dataset[cluster0_filter]["Glycemia"]
Glycemia1_filter = raw_dataset[cluster1_filter]["Glycemia"]

df_Glycemia=[Glycemia0_filter,Glycemia1_filter]
sns.boxplot(data=df_Glycemia, orient="v")
plt.title("Boxplot Glycemia\nAgglomerative")
plt.xlabel("Cluster")
#plt.savefig("immagini/Agglomerative/GlycemiaAgglomerative.pdf")
plt.show()

#Age, cluster0 e cluster1
Age0_filter = raw_dataset[cluster0_filter]["Age (1)"]
Age1_filter = raw_dataset[cluster1_filter]["Age (1)"]

df_Age=[Age0_filter,Age1_filter]
sns.boxplot(data=df_Age, orient="v")
plt.title("Boxplot Age\nAgglomerative")
plt.xlabel("Cluster")
#plt.savefig("immagini/Agglomerative/AgeAgglomerative.pdf")
plt.show()

#HR, cluster0 e cluster1
HR0_filter = raw_dataset[cluster0_filter]["HR"]
HR1_filter = raw_dataset[cluster1_filter]["HR"]

df_HR=[HR0_filter,HR1_filter]
sns.boxplot(data=df_HR, orient="v")
plt.title("Boxplot HR\nAgglomerative")
plt.xlabel("Cluster")
#plt.savefig("immagini/Agglomerative/HRAgglomerative.pdf")
plt.show()

#Weight, cluster0 e cluster1
Weight0_filter = raw_dataset[cluster0_filter]["Weight"]
Weight1_filter = raw_dataset[cluster1_filter]["Weight"]

df_Weight=[Weight0_filter,Weight1_filter]
sns.boxplot(data=df_Weight, orient="v")
plt.title("Boxplot Weight\nAgglomerative")
plt.xlabel("Cluster")
#plt.savefig("immagini/Agglomerative/WeightAgglomerative.pdf")
plt.show()

#Height, cluster0 e cluster1
Height0_filter = raw_dataset[cluster0_filter]["Height"]
Height1_filter = raw_dataset[cluster1_filter]["Height"]

df_Height=[Height0_filter,Height1_filter]
sns.boxplot(data=df_Height, orient="v")
plt.title("Boxplot Height\nAgglomerative")
plt.xlabel("Cluster")
#plt.savefig("immagini/Agglomerative/HeightAgglomerative.pdf")
plt.show()

#BMI, cluster0 e cluster1
BMI0_filter = raw_dataset[cluster0_filter]["BMI"]
BMI1_filter = raw_dataset[cluster1_filter]["BMI"]

df_BMI=[BMI0_filter,BMI1_filter]
sns.boxplot(data=df_BMI, orient="v")
plt.title("Boxplot BMI\nAgglomerative")
plt.xlabel("Cluster")
#plt.savefig("immagini/Agglomerative/BMIAgglomerative.pdf")
plt.show()

#Diastolic blood pressure, cluster0 e cluster1
Diastolic_blood_pressure0_filter = raw_dataset[cluster0_filter]["Diastolic blood pressure"]
Diastolic_blood_pressure1_filter = raw_dataset[cluster1_filter]["Diastolic blood pressure"]

df_Diastolic_blood_pressure=[Diastolic_blood_pressure0_filter,Diastolic_blood_pressure1_filter]
sns.boxplot(data=df_Diastolic_blood_pressure, orient="v")
plt.title("Boxplot Diastolic blood pressure\nAgglomerative")
plt.xlabel("Cluster")
#plt.savefig("immagini/Agglomerative/DiastolicBPAgglomerative.pdf")
plt.show()

#Systolic blood pressure, cluster0 e cluster1
Systolic_blood_pressure0_filter = raw_dataset[cluster0_filter]["Systolic blood pressure"]
Systolic_blood_pressure1_filter = raw_dataset[cluster1_filter]["Systolic blood pressure"]

df_Systolic_blood_pressure=[Systolic_blood_pressure0_filter,Systolic_blood_pressure1_filter]
sns.boxplot(data=df_Systolic_blood_pressure, orient="v")
plt.title("Boxplot Systolic blood pressure\nAgglomerative")
plt.xlabel("Cluster")
#plt.savefig("immagini/Agglomerative/SystolicBPAgglomerative.pdf")
plt.show()

#T-test per rilevare le feature più significative
print("T-test")
#Total cholesterol, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Total cholesterol"],raw_dataset[cluster1_filter]["Total cholesterol"], equal_var=False)
print('Test statistic for Total cholesterol: %f'%float("{:.6f}".format(t_value)))
print('p-value for Total cholesterol: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#HDL, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["HDL"],raw_dataset[cluster1_filter]["HDL"], equal_var=False)
print('Test statistic for HDL: %f'%float("{:.6f}".format(t_value)))
print('p-value for HDL: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#LDL, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["LDL"],raw_dataset[cluster1_filter]["LDL"], equal_var=False)
print('Test statistic for LDL: %f'%float("{:.6f}".format(t_value)))
print('p-value for LDL: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Triglycerides, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Triglycerides"],raw_dataset[cluster1_filter]["Triglycerides"], equal_var=False)
print('Test statistic for Triglycerides: %f'%float("{:.6f}".format(t_value)))
print('p-value for Triglycerides: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Glycemia, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Glycemia"],raw_dataset[cluster1_filter]["Glycemia"], equal_var=False)
print('Test statistic for Glycemia: %f'%float("{:.6f}".format(t_value)))
print('p-value for Glycemia: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Age, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Age (1)"],raw_dataset[cluster1_filter]["Age (1)"], equal_var=False)
print('Test statistic for Age: %f'%float("{:.6f}".format(t_value)))
print('p-value for Age: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#HR, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["HR"],raw_dataset[cluster1_filter]["HR"], equal_var=False)
print('Test statistic for HR: %f'%float("{:.6f}".format(t_value)))
print('p-value for HR: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Weight, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Weight"],raw_dataset[cluster1_filter]["Weight"], equal_var=False)
print('Test statistic for Weight: %f'%float("{:.6f}".format(t_value)))
print('p-value for Weight: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Height, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Height"],raw_dataset[cluster1_filter]["Height"], equal_var=False)
print('Test statistic for Height: %f'%float("{:.6f}".format(t_value)))
print('p-value for Height: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#BMI, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["BMI"],raw_dataset[cluster1_filter]["BMI"], equal_var=False)
print('Test statistic for BMI: %f'%float("{:.6f}".format(t_value)))
print('p-value for BMI: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Diastolic blood pressure, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Diastolic blood pressure"],raw_dataset[cluster1_filter]["Diastolic blood pressure"], equal_var=False)
print('Test statistic for Diastolic blood pressure: %f'%float("{:.6f}".format(t_value)))
print('p-value for Diastolic blood pressure: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Systolic blood pressure, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Systolic blood pressure"],raw_dataset[cluster1_filter]["Systolic blood pressure"], equal_var=False)
print('Test statistic for Systolic blood pressure: %f'%float("{:.6f}".format(t_value)))
print('p-value for Systolic blood pressure: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")


'''u_labels = np.unique(label)
#plotting the results:
for i in u_labels:
    plt.scatter(scaled_array[label == i , 0] , scaled_array[label == i , 1] , label = i)
plt.title("Agglomerative Clustering")
plt.legend()
plt.show()'''

'''plt.scatter(scaled_array[label==0, 0], scaled_array[label==0, 1], s=50, marker='o', color='red')
plt.scatter(scaled_array[label==1 , 0], scaled_array[label==1, 1], s=50, marker='o', color='blue')
plt.scatter(scaled_array[label==2, 0], scaled_array[label==2 , 1], s=50, marker='o', color='green')
plt.scatter(scaled_array[label==3, 0], scaled_array[label==3, 1], s=50, marker= 'o', color='purple')
plt.scatter(scaled_array[label==4, 0], scaled_array[label==4, 1], s=50, marker='o', color='orange')
plt.show()'''


#altra implementazione di Agglomerative Clustering
'''from numpy import unique, where

# define the model
model = AgglomerativeClustering(n_clusters=2)
# fit model and predict clusters
label = model.fit_predict(scaled_dataframe)
# retrieve unique clusters
clusters = unique(label)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(label == cluster)
	# create scatter of these samples
	plt.scatter(scaled_array[row_ix, 0], scaled_array[row_ix, 1])
# show the plot
plt.title("Agglomerative clustering")
plt.show()'''

#altra implementazione di agglomerative clustering
'''cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(scaled_dataframe)
plt.figure(figsize=(10, 7))
plt.scatter(scaled_array[:,0], scaled_array[:,1], c=cluster.labels_, cmap='rainbow')
plt.title("Agglomerative clustering")
plt.show()'''

#DBSCAN
neigh = NearestNeighbors(n_neighbors=5)
nbrs = neigh.fit(scaled_dataframe)
distances, indices = nbrs.kneighbors(scaled_dataframe)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.ylabel("eps")
plt.plot(distances)
#plt.savefig("immagini/DBSCAN/eps.pdf")
plt.show()

i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
#plt.xlabel("Points")
#plt.ylabel("Distance")
plt.title("knee point")
plt.ylabel("eps")
#plt.savefig("immagini/DBSCAN/kneepoint.pdf")
plt.show()
print(distances[knee.knee])

#cerco la miglior combinazione degli iperparametri con grid search
eps_to_test = [round(eps,1) for eps in np.arange(5, 5.5, 0.1)]
min_samples_to_test = range(5, 50, 5)

print("EPS:", eps_to_test)
print("MIN_SAMPLES:", list(min_samples_to_test))

#get_metric(), una volta all'interno del grid search, si occuperà di restituire automaticamente le metriche ad ogni combinazione di iperparametri
def get_metrics(eps, min_samples, dataset, iter_):
    # Fitting ======================================================================

    dbscan_model_ = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_model_.fit(dataset)

    # Mean Noise Point Distance metric =============================================
    noise_indices = dbscan_model_.labels_ == -1

    if True in noise_indices:
        neighboors = NearestNeighbors(n_neighbors=5).fit(dataset)
        distances, indices = neighboors.kneighbors(dataset)
        noise_distances = distances[noise_indices, 1:]
        noise_mean_distance = round(noise_distances.mean(), 3)
    else:
        noise_mean_distance = None

    # Number of found Clusters metric ==============================================

    number_of_clusters = len(set(dbscan_model_.labels_[dbscan_model_.labels_ >= 0]))

    # Log ==========================================================================

    print("%3d | Tested with eps = %3s and min_samples = %3s | %5s %4s" % (
    iter_, eps, min_samples, str(noise_mean_distance), number_of_clusters))


    return (noise_mean_distance, number_of_clusters)

#Istanzio i due dataframe che conterranno i risultati del grid search

# Dataframe per la metrica sulla distanza media dei noise points dai K punti più vicini
results_noise = pd.DataFrame(
    data = np.zeros((len(eps_to_test),len(min_samples_to_test))), # Empty dataframe
    columns = min_samples_to_test,
    index = eps_to_test
)

# Dataframe per la metrica sul numero di cluster
results_clusters = pd.DataFrame(
    data = np.zeros((len(eps_to_test),len(min_samples_to_test))), # Empty dataframe
    columns = min_samples_to_test,
    index = eps_to_test
)

#infine lanco il grid search
iter_ = 0

print("ITER| INFO%s |  DIST    CLUS" % (" " * 39))
print("-" * 65)

for eps in eps_to_test:
    for min_samples in min_samples_to_test:
        iter_ += 1

        # Calcolo le metriche
        noise_metric, cluster_metric = get_metrics(eps, min_samples, scaled_dataframe, iter_)

        # Inserisco i risultati nei relativi dataframe
        results_noise.loc[eps, min_samples] = noise_metric
        results_clusters.loc[eps, min_samples] = cluster_metric

#mostro i risultati del grid search appena eseguito per cercare la "miglior" combinazione di iperparametri
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8) )

sns.heatmap(results_noise, annot = True, ax = ax1, cbar = False).set_title("METRIC: Mean Noise Points Distance")
sns.heatmap(results_clusters, annot = True, ax = ax2, cbar = False).set_title("METRIC: Number of clusters")

ax1.set_xlabel("N"); ax2.set_xlabel("N")
ax1.set_ylabel("EPSILON"); ax2.set_ylabel("EPSILON")

plt.tight_layout();
#plt.savefig("immagini/DBSCAN/heatmapDBSCAN.pdf")
plt.show()


# define the model
dbscan_model = DBSCAN(eps=distances[knee.knee], min_samples=10)
# fit model and predict clusters
label = dbscan_model.fit_predict(scaled_dataframe)
print(label)
#aggiungo la colonna 'Cluster for DBSCAN' al dataset di partenza
raw_dataset["Cluster for DBSCAN"] = label
scaled_dataframe["Cluster for DBSCAN"] = label
print(raw_dataset)

n_clusters = len(np.unique(label) )
n_noise = np.sum(np.array(label) == -1, axis=0)

print('Estimated n. of clusters: %d + noise' % (n_clusters-1))
print('Estimated n. of noise points: %d' % n_noise)

#separo i dati in base all'etichetta assegnata
#ora separo i dati in base all'etichetta assegnata
noise_cluster_filter = raw_dataset["Cluster for DBSCAN"]==-1
print(raw_dataset[noise_cluster_filter])

cluster0_filter = raw_dataset["Cluster for DBSCAN"]==0
print(raw_dataset[cluster0_filter])

cluster1_filter = raw_dataset["Cluster for DBSCAN"]==1
print(raw_dataset[cluster1_filter])

#boxplot a confronto tra cluster0 e cluster1 per rilevare le feature più significative
#Total cholesterol, cluster0 e cluster1
TotChol0_filter = raw_dataset[cluster0_filter]["Total cholesterol"]
#print(raw_dataset[cluster0_filter]["Total cholesterol"])
TotChol1_filter = raw_dataset[cluster1_filter]["Total cholesterol"]
#print(raw_dataset[cluster1_filter]["Total cholesterol"])

df_TotChol=[TotChol0_filter,TotChol1_filter]
sns.boxplot(data=df_TotChol, orient="v")
plt.title("Boxplot Total Cholesterol\nDBSCAN")
plt.xlabel("Cluster")
#plt.savefig("immagini/DBSCAN/TotCholDBSCAN.pdf")
plt.show()

#HDL, cluster0 e cluster1
HDL0_filter = raw_dataset[cluster0_filter]["HDL"]
HDL1_filter = raw_dataset[cluster1_filter]["HDL"]

df_HDL=[HDL0_filter,HDL1_filter]
sns.boxplot(data=df_HDL, orient="v")
plt.title("Boxplot HDL\nDBSCAN")
plt.xlabel("Cluster")
#plt.savefig("immagini/DBSCAN/HDLDBSCAN.pdf")
plt.show()

#LDL, cluster0 e cluster1
LDL0_filter = raw_dataset[cluster0_filter]["LDL"]
LDL1_filter = raw_dataset[cluster1_filter]["LDL"]

df_LDL=[LDL0_filter,LDL1_filter]
sns.boxplot(data=df_LDL, orient="v")
plt.title("Boxplot LDL\nDBSCAN")
plt.xlabel("Cluster")
#plt.savefig("immagini/DBSCAN/LDLDBSCAN.pdf")
plt.show()

#Triglycerides, cluster0 e cluster1
Triglycerides0_filter = raw_dataset[cluster0_filter]["Triglycerides"]
Triglycerides1_filter = raw_dataset[cluster1_filter]["Triglycerides"]

df_Triglycerides=[Triglycerides0_filter,Triglycerides1_filter]
sns.boxplot(data=df_Triglycerides, orient="v")
plt.title("Boxplot Triglycerides\nDBSCAN")
plt.xlabel("Cluster")
#plt.savefig("immagini/DBSCAN/TriglyDBSCAN.pdf")
plt.show()

#Glycemia, cluster0 e cluster1
Glycemia0_filter = raw_dataset[cluster0_filter]["Glycemia"]
Glycemia1_filter = raw_dataset[cluster1_filter]["Glycemia"]

df_Glycemia=[Glycemia0_filter,Glycemia1_filter]
sns.boxplot(data=df_Glycemia, orient="v")
plt.title("Boxplot Glycemia\nDBSCAN")
plt.xlabel("Cluster")
#plt.savefig("immagini/DBSCAN/GlycemiaDBSCAN.pdf")
plt.show()

#Age, cluster0 e cluster1
Age0_filter = raw_dataset[cluster0_filter]["Age (1)"]
Age1_filter = raw_dataset[cluster1_filter]["Age (1)"]

df_Age=[Age0_filter,Age1_filter]
sns.boxplot(data=df_Age, orient="v")
plt.title("Boxplot Age\nDBSCAN")
plt.xlabel("Cluster")
#plt.savefig("immagini/DBSCAN/AgeDBSCAN.pdf")
plt.show()

#HR, cluster0 e cluster1
HR0_filter = raw_dataset[cluster0_filter]["HR"]
HR1_filter = raw_dataset[cluster1_filter]["HR"]

df_HR=[HR0_filter,HR1_filter]
sns.boxplot(data=df_HR, orient="v")
plt.title("Boxplot HR\nDBSCAN")
plt.xlabel("Cluster")
#plt.savefig("immagini/DBSCAN/HRDBSCAN.pdf")
plt.show()

#Weight, cluster0 e cluster1
Weight0_filter = raw_dataset[cluster0_filter]["Weight"]
Weight1_filter = raw_dataset[cluster1_filter]["Weight"]

df_Weight=[Weight0_filter,Weight1_filter]
sns.boxplot(data=df_Weight, orient="v")
plt.title("Boxplot Weight\nDBSCAN")
plt.xlabel("Cluster")
#plt.savefig("immagini/DBSCAN/WeightDBSCAN.pdf")
plt.show()

#Height, cluster0 e cluster1
Height0_filter = raw_dataset[cluster0_filter]["Height"]
Height1_filter = raw_dataset[cluster1_filter]["Height"]

df_Height=[Height0_filter,Height1_filter]
sns.boxplot(data=df_Height, orient="v")
plt.title("Boxplot Height\nDBSCAN")
plt.xlabel("Cluster")
#plt.savefig("immagini/DBSCAN/HeightDBSCAN.pdf")
plt.show()

#BMI, cluster0 e cluster1
BMI0_filter = raw_dataset[cluster0_filter]["BMI"]
BMI1_filter = raw_dataset[cluster1_filter]["BMI"]

df_BMI=[BMI0_filter,BMI1_filter]
sns.boxplot(data=df_BMI, orient="v")
plt.title("Boxplot BMI\nDBSCAN")
plt.xlabel("Cluster")
#plt.savefig("immagini/DBSCAN/BMIDBSCAN.pdf")
plt.show()

#Diastolic blood pressure, cluster0 e cluster1
Diastolic_blood_pressure0_filter = raw_dataset[cluster0_filter]["Diastolic blood pressure"]
Diastolic_blood_pressure1_filter = raw_dataset[cluster1_filter]["Diastolic blood pressure"]

df_Diastolic_blood_pressure=[Diastolic_blood_pressure0_filter,Diastolic_blood_pressure1_filter]
sns.boxplot(data=df_Diastolic_blood_pressure, orient="v")
plt.title("Boxplot Diastolic blood pressure\nDBSCAN")
plt.xlabel("Cluster")
#plt.savefig("immagini/DBSCAN/DiastolicBPDBSCAN.pdf")
plt.show()

#Systolic blood pressure, cluster0 e cluster1
Systolic_blood_pressure0_filter = raw_dataset[cluster0_filter]["Systolic blood pressure"]
Systolic_blood_pressure1_filter = raw_dataset[cluster1_filter]["Systolic blood pressure"]

df_Systolic_blood_pressure=[Systolic_blood_pressure0_filter,Systolic_blood_pressure1_filter]
sns.boxplot(data=df_Systolic_blood_pressure, orient="v")
plt.title("Boxplot Systolic blood pressure\nDBSCAN")
plt.xlabel("Cluster")
#plt.savefig("immagini/DBSCAN/SystolicBPDBSCAN.pdf")
plt.show()

#T-test per rilevare le feature più significative
print("T-test")
#Total cholesterol, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Total cholesterol"],raw_dataset[cluster1_filter]["Total cholesterol"], equal_var=False)
print('Test statistic for Total cholesterol: %f'%float("{:.6f}".format(t_value)))
print('p-value for Total cholesterol: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#HDL, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["HDL"],raw_dataset[cluster1_filter]["HDL"], equal_var=False)
print('Test statistic for HDL: %f'%float("{:.6f}".format(t_value)))
print('p-value for HDL: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#LDL, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["LDL"],raw_dataset[cluster1_filter]["LDL"], equal_var=False)
print('Test statistic for LDL: %f'%float("{:.6f}".format(t_value)))
print('p-value for LDL: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Triglycerides, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Triglycerides"],raw_dataset[cluster1_filter]["Triglycerides"], equal_var=False)
print('Test statistic for Triglycerides: %f'%float("{:.6f}".format(t_value)))
print('p-value for Triglycerides: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Glycemia, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Glycemia"],raw_dataset[cluster1_filter]["Glycemia"], equal_var=False)
print('Test statistic for Glycemia: %f'%float("{:.6f}".format(t_value)))
print('p-value for Glycemia: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Age, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Age (1)"],raw_dataset[cluster1_filter]["Age (1)"], equal_var=False)
print('Test statistic for Age: %f'%float("{:.6f}".format(t_value)))
print('p-value for Age: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#HR, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["HR"],raw_dataset[cluster1_filter]["HR"], equal_var=False)
print('Test statistic for HR: %f'%float("{:.6f}".format(t_value)))
print('p-value for HR: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Weight, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Weight"],raw_dataset[cluster1_filter]["Weight"], equal_var=False)
print('Test statistic for Weight: %f'%float("{:.6f}".format(t_value)))
print('p-value for Weight: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Height, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Height"],raw_dataset[cluster1_filter]["Height"], equal_var=False)
print('Test statistic for Height: %f'%float("{:.6f}".format(t_value)))
print('p-value for Height: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#BMI, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["BMI"],raw_dataset[cluster1_filter]["BMI"], equal_var=False)
print('Test statistic for BMI: %f'%float("{:.6f}".format(t_value)))
print('p-value for BMI: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Diastolic blood pressure, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Diastolic blood pressure"],raw_dataset[cluster1_filter]["Diastolic blood pressure"], equal_var=False)
print('Test statistic for Diastolic blood pressure: %f'%float("{:.6f}".format(t_value)))
print('p-value for Diastolic blood pressure: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")

#Systolic blood pressure, cluster0 e cluster1
t_value,p_value=stats.ttest_ind(raw_dataset[cluster0_filter]["Systolic blood pressure"],raw_dataset[cluster1_filter]["Systolic blood pressure"], equal_var=False)
print('Test statistic for Systolic blood pressure: %f'%float("{:.6f}".format(t_value)))
print('p-value for Systolic blood pressure: %f'%p_value)

alpha = 0.05

if p_value<=alpha:
    print("Poiché p-value(=%f)" % p_value, "<=", "alpha(=%.2f)" % alpha, "le due feature sono diverse")

else:

    print("Poichè p-value(=%f)" % p_value, ">", "alpha(=%.2f)" % alpha, "le due feature sono uguali")


'''u_labels = np.unique(label)
#plotting the results:
for i in u_labels:
    plt.scatter(scaled_array[label == i , 0] , scaled_array[label == i , 1] , label = i)
plt.title("DBSCAN")
plt.legend()
plt.show()

# Remove the noise
range_max = len(scaled_array)
scaled_array = np.array([scaled_array[i] for i in range(0, range_max) if label[i] != -1])
label = np.array([label[i] for i in range(0, range_max) if label[i] != -1])'''

# Generate scatter plot for training data
'''plt.scatter(scaled_array[:,0], scaled_array[:,1], marker="o", picker=True)
plt.title('Noise removed')
plt.show()'''

'''n_clusters = len(np.unique(label) )
n_noise = np.sum(np.array(label) == -1, axis=0)
print('Estimated n. of clusters: %d' % n_clusters)
print('Estimated n. of noise points: %d' % n_noise)
print(label)

#DBSCAN senza dati rumorosi
u_labels = np.unique(label)
#plotting the results:
for i in u_labels:
    plt.scatter(scaled_array[label == i , 0] , scaled_array[label == i , 1] , label = i)
plt.title("DBSCAN senza dati rumorosi")
plt.legend()
plt.show()

#alcuni confronti
#scatterplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (13,8))

sns.scatterplot(x = "Total cholesterol", y = "HDL", data = raw_dataset, hue = "LABEL for DBSCAN", palette = "Accent", ax = ax1)
sns.scatterplot(x = "Triglycerides", y = "Glycemia", data = raw_dataset, hue = "LABEL for DBSCAN", palette = "Accent", ax = ax2)
sns.scatterplot(x = "fT3", y = "Gender (Male = 1)", data = raw_dataset, hue = "LABEL for DBSCAN", palette = "Accent", ax = ax3)
sns.scatterplot(x = "Angina", y = "Height", data = raw_dataset, hue = "LABEL for DBSCAN", palette = "Accent", ax = ax4)

plt.tight_layout()
plt.show()'''

#pairplot
'''sns.pairplot(data = raw_dataset, hue = "LABEL for DBSCAN", palette = "Accent")
plt.show()'''

#altra implementazione di DBSCAN
'''from numpy import unique, where
# define the model
dbscan_model = DBSCAN(eps=2.5, min_samples=5)
# fit model and predict clusters
label = dbscan_model.fit_predict(scaled_dataframe)
print(label)
# retrieve unique clusters
clusters = unique(label)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(label == cluster)
	# create scatter of these samples
	plt.scatter(scaled_array[row_ix, 0], scaled_array[row_ix, 1])
# show the plot
plt.show()'''





















