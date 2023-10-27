import random
import numpy as np
from sklearn.metrics import silhouette_score


class KMeans():
    def __init__(self, dataset, n_clusters, max_iter=None, init="random_init", p_minkowski=2):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        if init in ["random_init", "space_division_init", "separated_init"] and hasattr(KMeans, init):
            init_func = getattr(KMeans, init)
            self.centroides = init_func(self)
        else:
            print("Error")
        self.p_minkowsi = p_minkowski

    def random_init(self):
        #print(self.dataset)
        return random.sample(self.dataset, self.n_clusters)

    def space_division_init(self):
        # Dividir el espacio en cuadrantes y seleccionar aleatoriamente un punto de cada cuadrante
        dimensions = len(self.dataset[0])
        min_values = [min(data[i] for data in self.dataset) for i in range(dimensions)]
        max_values = [max(data[i] for data in self.dataset) for i in range(dimensions)]

        centroids = []
        for _ in range(self.n_clusters):
            centroid = [random.uniform(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]
            centroids.append(tuple(centroid))
        return centroids

    def separated_init(self):
        # Comprobar posibles errores
        if 2 * self.n_clusters > len(self.dataset):
            raise AssertionError("Hay menos clusters que la mitad de elementos")

        # Se cojen aleatoriamente el doble de centroides
        k2_centroids = random.sample(self.dataset, 2 * self.n_clusters)

        # ...Comentar este codigo...
        selected_centroids = []
        for _ in range(self.n_clusters):
            # Calcular la distancia entre centroides y seleccionar los más separados
            pairwise_distances = [(c1, c2, sum((x - y) ** 2 for x, y in zip(c1, c2)) ** 0.5) for c1 in k2_centroids for
                                  c2 in k2_centroids if c1 != c2]
            selected_pair = max(pairwise_distances, key=lambda x: x[2])
            selected_centroids.extend([selected_pair[0], selected_pair[1]])
        return selected_centroids

    def distancia_mikowski(self, x, y, p=None):
        if p == None:
            p = self.p_minkowsi
        return np.power(np.sum(np.power(np.abs(np.array(x) - np.array(y)), p)), 1 / p)

    def single_linkage(self, cluster1, cluster2):
        # Calcular la distancia single-linkage entre dos clusters
        min_distance = float('inf')
        for point1 in cluster1:
            for point2 in cluster2:
                distance = self.distancia_minkowski(point1, point2)
                min_distance = min(min_distance, distance)
        return min_distance

    def complete_linkage(self, cluster1, cluster2):
        # Calcular la distancia completa-linkage entre dos clusters
        max_distance = float('-inf')
        for point1 in cluster1:
            for point2 in cluster2:
                distance = self.distancia_minkowski(point1, point2)
                max_distance = max(max_distance, distance)
        return max_distance

    # Función para asignar cada punto de datos al centroide más cercano
    def assign_to_clusters(self, dataset, centroids):
        clusters = [[] for _ in range(len(centroids))]
        for data_point in dataset:
            closest_centroid_index = min(
                range(len(centroids)),
                key=lambda i: self.distancia_mikowski(data_point, centroids[i])
            )
            clusters[closest_centroid_index].append(data_point)
        return clusters

    # Función para recalcular los centroides
    def recalculate_centroids(self, clusters):
        new_centroids = []
        for cluster in clusters:
            if not cluster:
                continue
            cluster_center = [sum(x) / len(cluster) for x in zip(*cluster)]
            new_centroids.append(cluster_center)
        return new_centroids

    # Función para verificar si los centroides han convergido
    def has_converged(self, old_centroids, new_centroids, tol=1e-4):
        return all(self.distancia_mikowski(old, new) < tol for old, new in zip(old_centroids, new_centroids))

    def fit(self):
        dataset = self.dataset
        centroids = self.centroides
        clusters = self.assign_to_clusters(dataset, self.centroides)
        if self.max_iter == None:
            while True:
                new_centroids = self.recalculate_centroids(clusters)
                if self.has_converged(centroids, new_centroids):
                    break
                centroids = new_centroids
            self.centroides = new_centroids
        else:
            for _ in range(self.max_iter):
                new_centroids = self.recalculate_centroids(clusters)
                self.centroides = new_centroids
                # Dentro del método fit de la clase KMeans, después de la convergencia
        clusters = self.assign_to_clusters(dataset, self.centroides)
        #silhouette_avg = silhouette_score(dataset, [self.dataset[i] for i in clusters])
        #print(f"Silhouette Score: {silhouette_avg}")
        # SSE
        sse = 0
        for i, cluster in enumerate(clusters):
            for point in cluster:
                sse += self.distancia_mikowski(point, self.centroides[i])
        #print(f"SSE (Sum of Squared Errors): {sse}")
        self.inertia = sse
        return centroids, clusters
    
    def predict(self, dataset):
        clusters = self.assign_to_clusters(dataset, self.centroides)
        return self.assign_numeric_labels(clusters)
    
    def assign_numeric_labels(self, clusters_array):
        labels = {}
        for idx,cluster in enumerate(clusters_array):
            for instance in cluster:
                labels[id(instance)] = idx
        return labels
"""
dataset_example = [(3, 4, 5), (3, 3, 1), (5, 6, 10), (7, 5, 2), (1, 9, 5), (4,6,7), (9,8,7)]
kmeans = KMeans(dataset_example, n_clusters=3, init="random_init", p_minkowski=5)
ct, cl = kmeans.fit()
print(ct)
print(cl)
print(kmeans.assign_numeric_labels(cl))
"""
"""
# Función para cargar datos desde un archivo CSV y manejar valores no numéricos o celdas vacías
def load_data(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            data_point = []
            for x in row:
                try:
                    data_point.append(float(x))
                except ValueError:
                    pass  # Ignorar celdas no numéricas
            if data_point:  # Asegurarse de que el punto de datos no esté vacío
                dataset.append(data_point)
    return dataset

# Aplicar K-Means
centroids, clusters = k_means(dataset, k)

# Imprimir los resultados
for i, centroid in enumerate(centroids):
    print(f"Centroide {i + 1}: {centroid}")
    print(f"Elementos en el cluster {i + 1}: {len(clusters[i])}")
    print()

# También puedes guardar los resultados en un archivo si lo deseas
with open('resultados.csv', 'w', newline='') as result_file:
    csv_writer = csv.writer(result_file)
    for i, cluster in enumerate(clusters):
        for data_point in cluster:
            csv_writer.writerow([i, *data_point])
"""
