import numpy as np
import matplotlib.pyplot as plt
import argparse
import configparser
import os

# Initialisation des paramètres aléatoires avec une seed pour la reproductibilité
np.random.seed(42)

# Fonction d'activation sigmoïde
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Fonction softmax
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # pour stabiliser les exponentielles
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# Fonction de propagation avant (Feedforward)
def feedforward(X, weights1, weights2, weights3):
    z1 = np.dot(X, weights1)
    a1 = sigmoid(z1)  # Première couche cachée
    z2 = np.dot(a1, weights2)
    a2 = sigmoid(z2)  # Deuxième couche cachée
    z3 = np.dot(a2, weights3)
    a3 = softmax(z3)  # Sortie softmax pour la classification binaire (2 neurones)
    return a1, a2, a3

# Fonction de la perte avec softmax pour deux classes
def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # pour éviter log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Fonction de calcul de l'accuracy
def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

# Fonction de backpropagation et mise à jour des poids
def backpropagate(X, y_true, a1, a2, a3, weights1, weights2, weights3, learning_rate):
    m = X.shape[0]  # Nombre d'échantillons

    # Calcul de l'erreur de la couche de sortie
    error_output = a3 - y_true
    d_weights3 = np.dot(a2.T, error_output) / m

    # Calcul de l'erreur de la couche cachée 2
    error_hidden2 = np.dot(error_output, weights3.T) * a2 * (1 - a2)
    d_weights2 = np.dot(a1.T, error_hidden2) / m

    # Calcul de l'erreur de la couche cachée 1
    error_hidden1 = np.dot(error_hidden2, weights2.T) * a1 * (1 - a1)
    d_weights1 = np.dot(X.T, error_hidden1) / m

    # Mise à jour des poids
    weights1 -= learning_rate * d_weights1
    weights2 -= learning_rate * d_weights2
    weights3 -= learning_rate * d_weights3

    return weights1, weights2, weights3

# Fonction de sauvegarde du modèle
def save_model(weights1, weights2, weights3, filename='model_weights.npz'):
    np.savez(filename, weights1=weights1, weights2=weights2, weights3=weights3)
    print(f"Model saved to {filename}")

# Chargement des données
def load_data():
    train_data = np.loadtxt('dataset/train.csv', delimiter=',')
    valid_data = np.loadtxt('dataset/valid.csv', delimiter=',')

    X_train = train_data[:, 1:]  # Toutes les colonnes sauf la première (features)
    y_train = train_data[:, 0]   # La première colonne (étiquette)
    X_valid = valid_data[:, 1:]  # Toutes les colonnes sauf la première (features)
    y_valid = valid_data[:, 0]   # La première colonne (étiquette)

    # Normalisation des données (optionnel mais recommandé)
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_valid = (X_valid - np.mean(X_valid, axis=0)) / np.std(X_valid, axis=0)

    # Conversion des étiquettes en format one-hot
    y_train_one_hot = np.zeros((y_train.size, 2))
    y_train_one_hot[np.arange(y_train.size), y_train.astype(int)] = 1

    y_valid_one_hot = np.zeros((y_valid.size, 2))
    y_valid_one_hot[np.arange(y_valid.size), y_valid.astype(int)] = 1

    return X_train, y_train_one_hot, X_valid, y_valid_one_hot

# Fonction principale de l'entraînement
def train(X_train, y_train, X_valid, y_valid, epochs=500, learning_rate=0.01, batch_size=32, hidden1_size=24, hidden2_size=24):
    # Initialisation des poids avec la méthode de Xavier
    input_size = X_train.shape[1]
    output_size = 2    # Deux neurones pour la classification binaire

    # Initialisation Xavier
    weights1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2 / input_size)
    weights2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2 / hidden1_size)
    weights3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2 / hidden2_size)

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    # Entraînement
    for epoch in range(epochs):
        # Propagation avant
        a1, a2, a3 = feedforward(X_train, weights1, weights2, weights3)

        # Calcul de la perte et de l'accuracy pour le jeu d'entraînement
        loss_train = cross_entropy(y_train, a3)
        acc_train = accuracy(y_train, a3)

        # Propagation avant pour le jeu de validation
        _, _, a3_valid = feedforward(X_valid, weights1, weights2, weights3)

        # Calcul de la perte et de l'accuracy pour le jeu de validation
        loss_valid = cross_entropy(y_valid, a3_valid)
        acc_valid = accuracy(y_valid, a3_valid)

        # Mise à jour des poids
        weights1, weights2, weights3 = backpropagate(X_train, y_train, a1, a2, a3, weights1, weights2, weights3, learning_rate)

        # Stockage des résultats
        train_losses.append(loss_train)
        valid_losses.append(loss_valid)
        train_accuracies.append(acc_train)
        valid_accuracies.append(acc_valid)

        # Affichage des métriques pour chaque epoch
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {loss_train:.4f}, Valid Loss: {loss_valid:.4f} - Train Accuracy: {acc_train:.4f}, Valid Accuracy: {acc_valid:.4f}")

    # Sauvegarde du modèle après l'entraînement
    save_model(weights1, weights2, weights3)

    # Affichage des courbes de perte et de précision
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(epochs), valid_accuracies, label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    return train_losses, valid_losses, train_accuracies, valid_accuracies

# Fonction pour charger les paramètres à partir d'un fichier de configuration
import configparser

def load_config(config_file):
    config = configparser.ConfigParser()

    # Lire le fichier de configuration
    config.read(config_file)

    # Extraire les paramètres
    epochs = int(config.get('network', 'epochs'))
    learning_rate = float(config.get('network', 'learning_rate'))
    batch_size = int(config.get('network', 'batch_size'))

    # Lire la ligne de 'network' et retirer les commentaires
    network_str = config.get('network', 'network')
    network_str = network_str.split('#')[0].strip()  # Retirer le commentaire et les espaces inutiles
    network = list(map(int, network_str.strip('[]').split(',')))  # Convertir la liste de neurones

    return epochs, learning_rate, batch_size, network


# Fonction principale pour gérer les arguments
def main():
    parser = argparse.ArgumentParser(description="Train a neural network on the breast cancer dataset.")

    # Arguments pour la configuration du modèle
    parser.add_argument('--config', type=str, help="Path to the configuration file.")
    parser.add_argument('--epochs', type=int, default=500, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate.")
    parser.add_argument('--layer', type=int, nargs='+', default=[24, 24], help="Number of neurons in each hidden layer.")

    args = parser.parse_args()

    # Si un fichier de configuration est passé en argument
    if args.config:
        print(f"Loading configuration from {args.config}")
        epochs, learning_rate, batch_size, network = load_config(args.config)
        hidden1_size, hidden2_size = network
    else:
        # Utiliser les arguments en ligne de commande
        epochs = args.epochs
        learning_rate = args.learning_rate
        batch_size = args.batch_size
        hidden1_size, hidden2_size = args.layer

    # Charger les données
    X_train, y_train, X_valid, y_valid = load_data()

    # Entraîner le modèle
    train_losses, valid_losses, train_accuracies, valid_accuracies = train(
        X_train, y_train, X_valid, y_valid, epochs, learning_rate, batch_size, hidden1_size, hidden2_size
    )

if __name__ == '__main__':
    main()
