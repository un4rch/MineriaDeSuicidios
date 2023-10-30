from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances, fowlkes_mallows_score, jaccard_score, precision_score, recall_score, accuracy_score, f1_score, cohen_kappa_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

class Metrics:
    def __init__(self):
        pass

    def calculate_all_metrics(self, true_labels, predicted_labels, X):
        # Métricas Internas
        silhouette = silhouette_score(X, predicted_labels)
        davies_bouldin = davies_bouldin_score(X, predicted_labels)
        calinski_harabasz = calinski_harabasz_score(X, predicted_labels)

        # Métricas Externas
        adjusted_rand = adjusted_rand_score(true_labels, predicted_labels)
        normalized_mutual_info = normalized_mutual_info_score(true_labels, predicted_labels)
        fowlkes_mallows = fowlkes_mallows_score(true_labels, predicted_labels)
        jaccard = jaccard_score(true_labels, predicted_labels, average='weighted')

        # Métricas de matriz de confusión
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        kappa = cohen_kappa_score(true_labels, predicted_labels)

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

        return [silhouette, davies_bouldin, calinski_harabasz, adjusted_rand, normalized_mutual_info, fowlkes_mallows, jaccard, precision, recall, accuracy, f1, kappa]
