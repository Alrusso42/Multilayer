import numpy as np

# Fonction de chargement des poids
def load_weights(file):
    data = np.load(file, allow_pickle=True)
    return {
        'weights1': data['weights1'],
        'weights2': data['weights2'],
        'weights3': data['weights3']
    }

# Fonction sigmoid (pour la classification binaire)
def sigmoid(x):
    # Normalisation pour éviter les grands nombres dans np.exp()
    x = np.clip(x, -500, 500)  # Limiter les valeurs de x dans une plage raisonnable
    return 1 / (1 + np.exp(-x))

# Fonction softmax (pour la classification multiclasse, deux classes ici)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Pour éviter les problèmes de débordement numérique
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Fonction pour effectuer une prédiction
def predict(X, weights):
    # Calcul du feedforward avec les poids chargés
    Z1 = np.dot(X, weights['weights1'])  # Calcul de la première couche cachée
    A1 = sigmoid(Z1)  # Application de la fonction sigmoid
    Z2 = np.dot(A1, weights['weights2'])  # Calcul de la deuxième couche cachée
    A2 = sigmoid(Z2)  # Application de la fonction sigmoid
    Z3 = np.dot(A2, weights['weights3'])  # Calcul de la couche de sortie

    # Appliquer softmax sur la sortie (deux classes)
    A3 = softmax(Z3)  # Probabilité pour chaque classe (ici deux classes)

    return A3

# Fonction de calcul de la perte (binary cross-entropy)
def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-8  # pour éviter log(0)

    # Vérifier si y_pred a deux colonnes (softmax)
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 2:
        y_pred = y_pred[:, 1]  # Sélectionner la probabilité pour la classe 1 (binaire)
    elif len(y_pred.shape) == 1:  # Si y_pred est unidimensionnel (un seul neurone pour la probabilité)
        y_pred = y_pred.flatten()  # Aplatir pour avoir une dimension (N,)

    # S'assurer que y_pred est une valeur entre 0 et 1
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calcul de la loss binaire pour une sortie de dimension (N,)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Fonction principale
def main():
    # Charger les poids du modèle
    weights = load_weights('model_weights.npz')

    # Charger les données de validation (remplacer par ton propre fichier de données)
    data_valid = np.loadtxt('dataset/valid.csv', delimiter=',')
    X_valid = data_valid[:, 1:]  # Les 30 features
    y_valid = data_valid[:, 0]  # La colonne des labels (0 ou 1)

    # Effectuer une prédiction
    predictions = predict(X_valid, weights)

    # Si les prédictions ont plus d'une colonne (cas softmax avec 2 classes), on prend la probabilité de la classe 1
    if predictions.shape[1] == 2:
        predictions = predictions[:, 1]  # Sélectionner la probabilité pour la classe 1

    # Calculer la perte sur les données de validation
    valid_loss = binary_crossentropy(y_valid, predictions)

    # Afficher les résultats
    print(f"Validation Loss: {valid_loss}")

    # Évaluer la précision (accuracy)
    predictions_class = (predictions > 0.5).astype(int)  # Convertir en classes (0 ou 1)
    accuracy = np.mean(predictions_class == y_valid)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
