import random

class KMeans():
    def __init__(self, dataset, n_clusters, max_iter=100, init="random_init"):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        if init in ["random_init", "space_division_init", "separated_init"] and hasattr(KMeans, init):
            init_func = getattr(KMeans, init)
            self.centroides = init_func(self)
        else:
            print("Error")
    
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
    
    def fit(self, dataset):
        return 0

dataset_example = [(3, 4), (3, 1), (5, 10), (7, 2), (9, 5)]
kmeans = KMeans(dataset_example, n_clusters=2, init="separated_init")
print(dataset_example)
print(kmeans.centroides)
