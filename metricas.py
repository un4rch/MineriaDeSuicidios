from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score, jaccard_score

class Metrics:
    def __init__(self):
        pass

    # Medidas Internas

    def silhouette_coefficient(self, X, labels):
        """
        Calcula el coeficiente de Silhouette para evaluar la calidad de los clústeres.
        - X: Datos de entrada
        - labels: Etiquetas de clústeres
        """
        return silhouette_score(X, labels)

    def davies_bouldin_index(self, X, labels):
        """
        Calcula el índice de Davies-Bouldin para evaluar la calidad de los clústeres.
        - X: Datos de entrada
        - labels: Etiquetas de clústeres
        """
        return davies_bouldin_score(X, labels)

    def calinski_harabasz_score(self, X, labels):
        """
        Calcula el índice de Calinski-Harabasz para evaluar la calidad de los clústeres.
        - X: Datos de entrada
        - labels: Etiquetas de clústeres
        """
        return calinski_harabasz_score(X, labels)

    # Medidas Externas

    def adjusted_rand_index(self, true_labels, predicted_labels):
        """
        Calcula el Índice de Rand Ajustado para comparar asignaciones de clústeres con etiquetas verdaderas.
        - true_labels: Etiquetas verdaderas
        - predicted_labels: Etiquetas de clústeres predichas
        """
        return adjusted_rand_score(true_labels, predicted_labels)

    def normalized_mutual_information(self, true_labels, predicted_labels):
        """
        Calcula la Información Mutua Normalizada para comparar asignaciones de clústeres con etiquetas verdaderas.
        - true_labels: Etiquetas verdaderas
        - predicted_labels: Etiquetas de clústeres predichas
        """
        return normalized_mutual_info_score(true_labels, predicted_labels)

    def fowlkes_mallows_index(self, true_labels, predicted_labels):
        """
        Calcula el Índice de Fowlkes-Mallows para comparar asignaciones de clústeres con etiquetas verdaderas.
        - true_labels: Etiquetas verdaderas
        - predicted_labels: Etiquetas de clústeres predichas
        """
        return fowlkes_mallows_score(true_labels, predicted_labels)

    def jaccard_index(self, true_labels, predicted_labels):
        """
        Calcula el Índice de Jaccard para comparar asignaciones de clústeres con etiquetas verdaderas.
        - true_labels: Etiquetas verdaderas
        - predicted_labels: Etiquetas de clústeres predichas
        """
        return jaccard_score(true_labels, predicted_labels)
