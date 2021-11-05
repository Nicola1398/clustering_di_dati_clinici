import pandas as pd #for data management
import numpy as np #for data management
import pylab as pl
import seaborn as sns #for data visualization and specifically for pairplot()
import matplotlib.pyplot as plt #for data visualization
from sklearn.preprocessing import StandardScaler  #to transform the dataset
from scipy import stats

#DBSCAN
from sklearn.cluster import DBSCAN #to instantiate and fit the model
from sklearn.metrics import pairwise_distances #for Model evaluation
from sklearn.neighbors import NearestNeighbors #for Hyperparameter Tuning

from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

#import data from an Excel file(.xlsx) with pandas
raw_dataset = pd.read_excel("TH in inpts for ANNE.xlsx", engine='openpyxl', sheet_name="Foglio1", usecols='C,D:AQ', header=1)
print(raw_dataset)
print("Dati caricati correttamente\n")

df_features = pd.read_excel("TH in inpts for ANNE.xlsx", engine='openpyxl', sheet_name="Foglio1", usecols='C,D:AQ', header=None, nrows=1, skiprows=[0])
features = (np.asarray(df_features)).flatten() #array
print(features)
size_features = features.shape[0]
print(size_features) #dim array


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
print("La standardizzazione ha funzionato e adesso tutte le variabili risultano confrontabili")


#DBSCAN
neigh = NearestNeighbors(n_neighbors=5)
nbrs = neigh.fit(scaled_dataframe)
distances, indices = nbrs.kneighbors(scaled_dataframe)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.ylabel("eps")
plt.plot(distances)
plt.savefig("immagini/DBSCAN4/eps.pdf")
#plt.show()
plt.close()

i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
#plt.xlabel("Points")
#plt.ylabel("Distance")
plt.title("knee point")
plt.ylabel("eps")
plt.savefig("immagini/DBSCAN4/kneepoint.pdf")
#plt.show()
plt.close()
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
plt.savefig("immagini/DBSCAN4/heatmapDBSCAN.pdf")
#plt.show()
plt.close()

# define the model
dbscan_model = DBSCAN(eps=distances[knee.knee], min_samples=5)
# fit model and predict clusters
label = dbscan_model.fit_predict(scaled_dataframe)
print(label)
#aggiungo la colonna 'Cluster for DBSCAN' al dataset di partenza
raw_dataset["Cluster for DBSCAN"] = label
scaled_dataframe["Cluster for DBSCAN"] = label
print(raw_dataset)
print(scaled_dataframe)

n_clusters = len(np.unique(label) )
n_noise = np.sum(np.array(label) == -1, axis=0)

print('Estimated n. of clusters: %d + noise' % (n_clusters-1))
print('Estimated n. of noise points: %d' % n_noise)

#separo i dati in base all'etichetta assegnata
#ora separo i dati in base all'etichetta assegnata
noise_cluster_filter = scaled_dataframe["Cluster for DBSCAN"]==-1
print(scaled_dataframe[noise_cluster_filter])

cluster0_filter = scaled_dataframe["Cluster for DBSCAN"]==0
print(scaled_dataframe[cluster0_filter])

cluster1_filter = scaled_dataframe["Cluster for DBSCAN"]==1
print(scaled_dataframe[cluster1_filter])

cluster2_filter = scaled_dataframe["Cluster for DBSCAN"]==2
print(scaled_dataframe[cluster2_filter])

cluster3_filter = scaled_dataframe["Cluster for DBSCAN"]==3
print(scaled_dataframe[cluster3_filter])

# Kruskal-Wallis test e boxplot tra cluster0, cluster1, cluster2 e cluster3 per rilevare le feature più significative

#Features, cluster0, cluster1, cluster2 e cluster3
alpha = 0.05
for i in range (0,size_features,1):

    filter0 = raw_dataset[cluster0_filter][features[i]]
    filter1 = raw_dataset[cluster1_filter][features[i]]
    filter2 = raw_dataset[cluster2_filter][features[i]]
    filter3 = raw_dataset[cluster3_filter][features[i]]

    median0 = raw_dataset[cluster0_filter][features[i]].median()
    median1 = raw_dataset[cluster1_filter][features[i]].median()
    median2 = raw_dataset[cluster2_filter][features[i]].median()
    median3 = raw_dataset[cluster3_filter][features[i]].median()

    shape0 = raw_dataset[cluster0_filter][features[i]].shape
    shape1 = raw_dataset[cluster1_filter][features[i]].shape
    shape2 = raw_dataset[cluster2_filter][features[i]].shape
    shape3 = raw_dataset[cluster3_filter][features[i]].shape

    # Kruskal-Wallis test (versione non parametrica di ANOVA)
    stat, pvalue = stats.kruskal(filter0, filter1, filter2, filter3)
    print("Kruskal-Wallis test for %s (versione non parametrica di ANOVA)" % (features[i]))
    print("--------------------------------------------------------------")
    print("p-value: %.6f" % (pvalue))

    if pvalue <= alpha:
        print("Poiché p-value(=%f)" % pvalue, "<=", "alpha(=%.2f)" % alpha, "ci sono delle differenze tra le feature\n")

    else:

        print("Poichè p-value(=%f)" % pvalue, ">", "alpha(=%.2f)" % alpha, "le feature sono uguali\n")

    df=[filter0,filter1, filter2, filter3]
    sns.boxplot(data=df, orient="v", whis=10, showmeans=True)
    plt.title("Boxplot %s (DBSCAN)\nP-value: %.6f" % (features[i], pvalue))
    #plt.xlabel("Cluster")
    plt.xticks(np.arange(4), ('cluster 0\npatients: %d\nmed: %.1f' %(shape0[0], median0), 'cluster 1\npatients: %d\nmed: %.1f' %(shape1[0], median1), 'cluster 2\npatients: %d\nmed: %.1f' %(shape2[0], median2), 'cluster 3\npatients: %d\nmed: %.1f' %(shape3[0], median3)))
    plt.savefig("immagini/DBSCAN4/%s.pdf" %(features[i]))
    #plt.show()
    plt.close()














