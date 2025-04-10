# 📡 Projet de Géolocalisation avec Réseaux de Neurones

Ce projet utilise les **puissances de signal RSSI** issues de 4 passerelles pour estimer la **position (x, y)** d’un objet ou d’un utilisateur à l’aide de **réseaux de neurones** avec TensorFlow/Keras.

---

## 🧠 Objectif

Prédire la position réelle à partir des signaux RSSI mesurés, en utilisant un modèle de deep learning entraîné sur des données simulées ou réelles.

---

## 📁 Données utilisées

Les données proviennent de **4 fichiers CSV** :
- `RSSI_0.csv`
- `RSSI_1.csv`
- `RSSI_2.csv`
- `RSSI_3.csv`

Chaque fichier représente les valeurs de puissance de signal (RSSI) reçues depuis une passerelle.

---

## 🔧 Étapes du pipeline

1. **Chargement & Visualisation** des données avec heatmaps.
2. **Nettoyage** : suppression de la dernière ligne et colonne + remplissage des `NaN` par la moyenne.
3. **Création de dataset** (RSSI comme input, position (x, y) comme output).
4. **Modélisation avec Keras** :
   - Modèle dense simple (64-64-2)
   - Modèle amélioré avec `Dropout`, `ReduceLROnPlateau`, et plus de couches.
5. **Évaluation** :
   - `MSE`, `MAE`, `R²`
   - Affichage des courbes de pertes
   - Histogramme des erreurs de localisation

---

## 📦 Librairies utilisées

- `numpy`
- `pandas`
- `matplotlib`, `seaborn`
- `tensorflow`, `keras`
- `scikit-learn`

---

## ⚙️ Lancer le projet

### 1. Cloner le dépôt
```bash
git clone https://github.com/bachir6c/Geolocalisation.git
cd Geolocalisation
