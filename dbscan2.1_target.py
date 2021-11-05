import pandas as pd #for data management
import numpy as np #for data management
import pylab as pl
import seaborn as sns #for data visualization and specifically for pairplot()
import matplotlib.pyplot as plt #for data visualization
from sklearn.preprocessing import StandardScaler  #to transform the dataset
from scipy import stats
import math

#DBSCAN
from sklearn.cluster import DBSCAN #to instantiate and fit the model
from sklearn.metrics import pairwise_distances #for Model evaluation
from sklearn.neighbors import NearestNeighbors #for Hyperparameter Tuning

from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

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

n_patients = raw_dataset_target.shape[0]
print(n_patients)
log_n_patients = math.ceil(np.log(n_patients)) #per minPts
print(log_n_patients)



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


#DBSCAN
neigh = NearestNeighbors(n_neighbors=n_patients)
nbrs = neigh.fit(only_features)
distances, indices = nbrs.kneighbors(only_features)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.ylabel("eps")
plt.plot(distances)
plt.savefig("immagini/DBSCAN2.1_target/eps.pdf")
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
plt.savefig("immagini/DBSCAN2.1_target/kneepoint.pdf")
#plt.show()
plt.close()
print(distances[knee.knee])

#cerco la miglior combinazione degli iperparametri con grid search
eps_to_test = [round(eps,1) for eps in np.arange(5.5, 6.1, 0.1)]
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
        neighboors = NearestNeighbors(n_neighbors=n_patients).fit(dataset)
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
        noise_metric, cluster_metric = get_metrics(eps, min_samples, only_features, iter_)

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
plt.savefig("immagini/DBSCAN2.1_target/heatmapDBSCAN.pdf")
#plt.show()
plt.close()

# define the model
dbscan_model = DBSCAN(eps=distances[knee.knee], min_samples=n_patients)
# fit model and predict clusters
label = dbscan_model.fit_predict(only_features)
print(label)
#aggiungo la colonna 'Cluster for DBSCAN' al dataset di partenza
raw_dataset_target["Cluster for DBSCAN"] = label
scaled_dataframe["Cluster for DBSCAN"] = label
print(raw_dataset_target)
print(scaled_dataframe)

n_clusters = len(np.unique(label) )
n_noise = np.sum(np.array(label) == -1, axis=0)

print('Estimated n. of clusters: %d + noise' % (n_clusters-1))
print('Estimated n. of noise points: %d' % n_noise)
