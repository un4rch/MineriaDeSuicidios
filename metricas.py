from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances, fowlkes_mallows_score, jaccard_score, precision_score, recall_score, accuracy_score, f1_score, cohen_kappa_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

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

    # Métricas de matriz de confusión
    def precision(self, true_labels, predicted_labels):
        """
        Calcula la métrica de precisión.
        - true_labels: Etiquetas verdaderas
        - predicted_labels: Etiquetas de clústeres predichas
        """
        return precision_score(true_labels, predicted_labels)

    def recall(self, true_labels, predicted_labels):
        """
        Calcula la métrica de exhaustividad (recall).
        - true_labels: Etiquetas verdaderas
        - predicted_labels: Etiquetas de clústeres predichas
        """
        return recall_score(true_labels, predicted_labels)

    def accuracy(self, true_labels, predicted_labels):
        """
        Calcula la métrica de exactitud (accuracy).
        - true_labels: Etiquetas verdaderas
        - predicted_labels: Etiquetas de clústeres predichas
        """
        return accuracy_score(true_labels, predicted_labels)

    def f1_score(self, true_labels, predicted_labels):
        """
        Calcula la puntuación F (F-score).
        - true_labels: Etiquetas verdaderas
        - predicted_labels: Etiquetas de clústeres predichas
        """
        return f1_score(true_labels, predicted_labels)

    def calculate_kappa(self, true_labels, predicted_labels):
        kappa = cohen_kappa_score(true_labels, predicted_labels)
        return kappa

    def calculate_all_metrics(self, true_labels, predicted_labels, X):
        # Métricas Internas
        silhouette = silhouette_score(X, predicted_labels)
        davies_bouldin = davies_bouldin_score(X, predicted_labels)
        calinski_harabasz = calinski_harabasz_score(X, predicted_labels)

        # Métricas Externas
        adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
        normalized_mutual_info = normalized_mutual_info_score(true_labels, predicted_labels)
        fowlkes_mallows = fowlkes_mallows_score(true_labels, predicted_labels)
        jaccard = jaccard_score(true_labels, predicted_labels)

        # Métricas de matriz de confusión
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        kappa = calculate_kappa(true_labels, predicted_labels)

        # Imprimir los resultados
        print("Métricas Internas:")
        print(f"Coeficiente de Silhouette: {silhouette}")
        print(f"Índice de Davies-Bouldin: {davies_bouldin}")
        print(f"Índice de Calinski-Harabasz: {calinski_harabasz}")

        print("\nMétricas Externas:")
        print(f"Índice de Rand Ajustado: {adjusted_rand}")
        print(f"Información Mutua Normalizada: {normalized_mutual_info}")
        print(f"Índice de Fowlkes-Mallows: {fowlkes_mallows}")
        print(f"Índice de Jaccard: {jaccard}")

        print("\nMétricas de Matriz de Confusión:")
        print(f"Precisión: {precision}")
        print(f"Exhaustividad (Recall): {recall}")
        print(f"Exactitud (Accuracy): {accuracy}")
        print(f"Puntuación F (F-score): {f1}")
        print(f"Estadística Kappa: {kappa}")