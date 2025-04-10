import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np   # manipulation efficace vecteur matrice
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Charger les données
geo_dt0 = pd.read_csv('/home/mouhamed-bachir.cisse/ResauxNeurones/RSSI_0.csv')
geo_dt1 = pd.read_csv('/home/mouhamed-bachir.cisse/ResauxNeurones/RSSI_1.csv')
geo_dt2 = pd.read_csv('/home/mouhamed-bachir.cisse/ResauxNeurones/RSSI_2.csv')
geo_dt3 = pd.read_csv('/home/mouhamed-bachir.cisse/ResauxNeurones/RSSI_3.csv')

data = [geo_dt0, geo_dt1, geo_dt2, geo_dt3]
names = ['Passerelle 1', 'Passerelle 2', 'Passerelle 3', 'Passerelle 4']
# Création d'une figure avec 2x2 sous-graphiques
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
fig.suptitle('Visualisation des valeurs manquantes pour chaque passerelle', fontsize=16)

for i, (dt, name) in enumerate(zip(data, names)):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    
    # Comptage des valeurs manquantes
    missing_values = dt.isnull().sum().sum()
    missing_percentage = (missing_values / dt.size) * 100
    
    print(f"Les valeurs manquantes dans {name}:")
    print(f"Nombre total: {missing_values}")
    print(f"Pourcentage: {missing_percentage:.2f}%\n")
    
    # Visualisation des valeurs manquantes
    sns.heatmap(dt.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title(f"{name}\nValeurs manquantes: {missing_values} ({missing_percentage:.2f}%)")
    ax.set_xlabel('Colonnes')
    ax.set_ylabel('Lignes')
    
plt.tight_layout()
plt.show()

data_mis_a_jour=[]
for dt, name in zip(data, names):
    mis_a_jour=dt.iloc[:-1,:-1]
    data_mis_a_jour.append(mis_a_jour)
    print(f"Tableau mis à jour pour {name} :")
    print(f"Dimensions originales : {dt.shape}, Dimensions après modification : {mis_a_jour.shape}\n")
data = data_mis_a_jour

# Remplacer les valeurs manquantes par la moyenne
data_filled = [dt.fillna(dt.mean()) for dt in data]

# Visualisation des valeurs manquantes après remplacement
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
fig.suptitle("Visualisation après remplissage des valeurs manquantes", fontsize=16)


for i, (dt, name) in enumerate(zip(data_filled, names)):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    
    # Vérification des valeurs manquantes
    #missing_values = dt.isnull().sum().sum()
    #missing_percentage = (missing_values / dt.size) * 100
    
    # Affichage des informations
    print(f"Après remplissage des NaN pour {name}:")
    print(f"Nombre total: {missing_values}")
    print(f"Pourcentage: {missing_percentage:.2f}%\n")
    
    # Heatmap pour visualiser les NaN restants (devrait être vide après remplissage)
    sns.heatmap(dt.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title(f"{name} - Après remplissage\nValeurs manquantes: {missing_values} ({missing_percentage:.2f}%)")
    ax.set_xlabel("Colonnes")
    ax.set_ylabel("Lignes")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# Supposons que geo_dt0, geo_dt1, geo_dt2, geo_dt3 sont des tableaux NumPy
# Initialisation de la liste pour les données
#data = [

# Parcourir chaque point (x, y) de vos matrices RSSI (DataFrame pandas)
for x in range(geo_dt0.shape[0]):
    for y in range(geo_dt0.shape[1]):
        # Utilisez .iloc pour accéder aux valeurs
        rssi1 = geo_dt0[x, y]
        rssi2 = geo_dt1[x, y]
        rssi3 = geo_dt2[x, y]
        rssi4 = geo_dt3[x, y]
        
        # Vérifier si les valeurs sont valides (pas de NaN)
        if not np.isnan([rssi1, rssi2, rssi3, rssi4]).any():
            # Ajouter les RSSI (features) et les coordonnées (targets)
            data.append([rssi1, rssi2, rssi3, rssi4, x, y])


# Convertir la liste en un tableau NumPy
data = np.array(data)

# Séparer les features (X) et les targets (Y)
X = data[:, :4]  # Les 4 premières colonnes (RSSI1 à RSSI4)
Y = data[:, 4:]  # Les deux dernières colonnes (X et Y)

print("Shape of X:", X.shape)  # Caractéristiques d'entrée
print("Shape of Y:", Y.shape)  # Coordonnées de sortie

# Fonction pour afficher une carte RSSI
def plot_rssi(data, title, ax):
    
    im = ax.imshow(data, cmap='viridis', origin='lower')
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    return im


# Chargement des fichiers RSSI (remplacez les chemins par vos fichiers)
geo_dt0 = pd.read_csv('RSSI_0.csv').to_numpy()
geo_dt1 = pd.read_csv('RSSI_1.csv').to_numpy()
geo_dt2 = pd.read_csv('RSSI_2.csv').to_numpy()
geo_dt3 = pd.read_csv('RSSI_3.csv').to_numpy()

print(type(geo_dt0))  # Remplacez geo_dt0 par votre variable


# Création de la figure et des sous-graphiques
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Cartes RSSI des 4 passerelles', fontsize=16)

# Affichage des 4 cartes dans une grille
datasets = [geo_dt0, geo_dt1, geo_dt2, geo_dt3]
for i, data in enumerate(datasets):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    im = plot_rssi(data, f'Passerelle {i+1}', ax)

# Ajuster l'espacement
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Laisser de la place pour le titre
plt.show()

#Ici on va choisir la fonction de perte MSE car on cherche à minimiser les coordonnées prédites et réelles
# Division des données en ensembles d'entraînement et de test

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Affichage de la taille des ensembles
print("Taille de l'ensemble d'entraînement :", x_train.shape[0])
print("Taille de l'ensemble de test :", x_test.shape[0])



model = Sequential([
    Dense(64, activation='relu', input_dim=4),  # 4 entrées : RSSI1, RSSI2, RSSI3, RSSI4
    Dense(64, activation='relu'),
    Dense(2)  # 2 sorties : X et Y
])

# Compilation du modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])

# Entraîner le modèle
entrainement = model.fit(x_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Évaluation du modèle
loss, mse = model.evaluate(x_test, y_test, verbose=1)


from sklearn.metrics import r2_score

# Prédictions sur l'ensemble de test
Y_pred = model.predict(x_test)

# Calcul du R²
r2 = r2_score(y_test, Y_pred)
print(f"R² sur les données de test: {r2}")



# Tracer de la perte (Loss)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(entrainement.history['loss'], label='Perte (entraînement)',color='blue')
plt.plot(entrainement.history['val_loss'], label='Perte (test)', color='red')
plt.title("Évolution de la perte")
plt.xlabel("Époques")
plt.ylabel("Perte ")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 2)
plt.plot(entrainement.history['loss'], label='R² (entraînement)',color='green')
plt.plot(entrainement.history['val_loss'], label='R² (test)', color='orange')
plt.title("Évolution de R²")
plt.xlabel("Époques")
plt.ylabel("R² ")
plt.legend()
plt.show()
distances = np.sqrt(np.sum((Y_pred - y_test)**2, axis=1))
plt.figure(figsize=(12, 6))

# Tracer l'histogramme des distances
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogramme des distances entre positions prédites et exactes', fontsize=16)
plt.xlabel('Distance ', fontsize=14)
plt.ylabel('Fréquence', fontsize=14)
plt.grid(alpha=0.5)
plt.show()
#On se propose de faire un réseaux de neuronnes améliorés
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
# Modèle amélioré
model = Sequential([
    Dense(128, activation='relu', input_dim=x_train.shape[1]),
    Dropout(0.2),  # Régularisation pour éviter le surapprentissage
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='linear')  # Deux sorties pour X et Y
])

# Compiler le modèle
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# Callback pour réduire le learning rate
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# Entraîner le modèle
history = model.fit(
    X_train_scaled, y_train, 
    validation_data=(X_test_scaled, y_test), 
    epochs=100, 
    batch_size=32, 
    callbacks=[lr_scheduler],
    verbose=1
)

# Évaluer le modèle
Y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, Y_pred)
print(f"R2 Score: {r2:.2f}")

# Tracer les métriques et la perte
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss (train)', color='blue')
plt.plot(history.history['val_loss'], label='Loss (val)', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(np.linalg.norm(Y_pred - y_test, axis=1), bins=20, color='purple', alpha=0.7)
plt.xlabel('Distance',fontsize=14)
plt.ylabel('Frequence',fontsize=14)
plt.title('Histogramme des distances entre positions prédites et exactes')
plt.grid(alpha=0.5)
plt.show()
