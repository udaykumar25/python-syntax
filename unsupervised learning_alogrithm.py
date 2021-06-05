#unsupervised learning_alogrithm

#Hclust_clustering
# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
import matplotlib.pylab as plt
#here we calculating distance matrix and creating dendrogram
z = linkage(df_norm, method = "complete", metric = "euclidean")
# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,leaf_rotation = 0,leaf_font_size = 10 )
plt.show()
#from the plot we decide how many clusters are to be made 
# Now applying AgglomerativeClustering choosing 4 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering
#agglomerative function is fitted or applied on dataset
h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_
#from the function take labels
#converting into series
cluster_labels = pd.Series(h_complete.labels_)


#kmeans_clustering
from sklearn.cluster import	KMeans
###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
TWSS #here we capturing inertia
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")
#by seeing above plot we decided to cluster into two plots by (interia)
# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)
model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 


#principle compound analysis
from sklearn.decomposition import PCA
pca = PCA(n_components = 6)
pca_values = pca.fit_transform(uni_normal)
# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_;var
pca.components_#weights
# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100);var1
# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")
# PCA scores
pca_values
#converting array to dataframe
pca_data = pd.DataFrame(pca_values)


#association_apriori
# pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.sort_values('lift', ascending = False)

#Recomendation_system
from sklearn.metrics.pairwise import linear_kernel
# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
from sklearn.metrics.pairwise import euclidean_distances
# distance between rows of X
euclidean_distances=euclidean_distances(tfidf_matrix, tfidf_matrix)
import gower
gow=(gower.gower_matrix(Entertainment))

#Network_Analytics
import networkx as nx 
#convert it into graph
g = nx.Graph()
g = nx.from_pandas_edgelist(G, source = 'Source Airport', target = 'Destination Airport')
b = nx.degree_centrality(g)
closeness = nx.closeness_centrality(g)
b = nx.betweenness_centrality(g) # Betweeness_Centrality
evg = nx.eigenvector_centrality(g) # Eigen vector centrality
cluster_coeff = nx.clustering(g)
cc = nx.average_clustering(g) 

