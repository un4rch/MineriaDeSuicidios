import random
import numpy as np

class KMeans():
    def __init__(self, dataset, n_clusters, max_iter=100, init="random_init", p_minkowski=2):
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
        if 2*self.n_clusters > len(self.dataset):
            raise AssertionError("Hay mas clusters que elementos")
        
        # Se cojen aleatoriamente el doble de centroides
        k2_centroids = random.sample(self.dataset, 2*self.n_clusters)
        
        # ...Comentar este codigo...
        selected_centroids = []
        for _ in range(self.n_clusters):
            # Calcular la distancia entre centroides y seleccionar los más separados
            pairwise_distances = [(c1, c2, sum((x - y) ** 2 for x, y in zip(c1, c2)) ** 0.5) for c1 in k2_centroids for c2 in k2_centroids if c1 != c2]
            selected_pair = max(pairwise_distances, key=lambda x: x[2])
            selected_centroids.extend([selected_pair[0], selected_pair[1]])
        return selected_centroids
    
    def distancia_mikowski(self, x, y, p=None):
        if p==None:
            p=self.p_minkowsi
        return np.power(np.sum(np.power(np.abs(np.array(x) - np.array(y)), p)), 1/p)

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

    def fit(self, dataset):
        for _ in range(self.max_iter):
            clusters = self.assign_to_clusters(dataset, self.centroides)
            new_centroids = self.recalculate_centroids(clusters)
            if self.has_converged(centroids, new_centroids):
                break
            centroids = new_centroids
        return centroids, clusters

dataset_example = [(3, 4), (3, 1), (5, 10), (7, 2), (9, 5)]
kmeans = KMeans(dataset_example, n_clusters=2, init="separated_init", p_minkowski=5)
print(dataset_example)
print(kmeans.centroides)
print(kmeans.fit(dataset_example))

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
