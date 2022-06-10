#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:36:47 2022

@author: abdarrahmane
"""
 # COURBES D'APPRENTISSAGE & SAUVEGARDE DU RESEAU #
 # Importation des modules TensorFlow & Keras
 # => construction et exploitation de réseaux de neurones
import tensorflow as tf
 
 # Importation du module numpy
# => manipulation de tableaux multidimensionnels
import numpy as np

# Importation du module graphique
# => tracé de courbes et diagrammes
import matplotlib.pyplot as plt


 # Chargement des données d'apprentissage et de tests 
 # Chargement en mémoire de la base de données des caractères MNIST
# => tableaux de type ndarray (Numpy) avec des valeur entières
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Changements de format pour exploitation
# the values associatd to pixels are integer between 0 and 25 
# => transformation to reel value between 0.0 et 1.0 
x_train, x_test = x_train / 255.0, x_test / 255.0


# The data of input are a matrices of pixels 28x28 # => transformation to vecteur 28x28=784 pixels 
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
# x_test = x_test.reshape(10, 784)

# The output data are integers associated with the numbers to be identified
# => transformation to booléens vecteurs for a classification for 10 values 
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# DESCRIPTION du modèle Perceptron multicouches (MLP)
# => 1 couche cachée avec 150 neurones
#--- # Creation of a multilayer perceptron
MyNetwork = tf.keras.Sequential()


# Description of the first hiden layer
MyNetwork.add(tf.keras.layers.Dense(
units=150, # 150 neurones
input_shape=(784,), # nombre of input (because it is the first layer) 
activation='relu'))# activation function

# Description of the first hiden layer
MyNetwork.add(tf.keras.layers.Dense(
units=150, # 150 neurones
input_shape=(784,), # nombre of input (because it is the first layer) 
activation='relu'))# activation function

# Description of the first hiden layer
MyNetwork.add(tf.keras.layers.Dense(
units=150, # 150 neurones
input_shape=(784,), # nombre of input (because it is the first layer) 
activation='relu'))# activation function

# Description of the first hiden layer
MyNetwork.add(tf.keras.layers.Dense(
units=150, # 150 neurones
input_shape=(784,), # nombre of input (because it is the first layer) 
activation='relu'))# activation function

# Description of the first hiden layer
MyNetwork.add(tf.keras.layers.Dense(
units=150, # 150 neurones
input_shape=(784,), # nombre of input (because it is the first layer) 
activation='relu'))# activation function

# Description of output layer 
MyNetwork.add(tf.keras.layers.Dense(
units=10, # 10 neurones
activation='softmax')), # fonction d'activation (outputs over [0,1])

#------------------------- 
# COMPILATION du réseau
# => configuration de la procédure pour l'apprentissage
 #---------------------------------------- 
MyNetwork.compile(optimizer='adam', # algo d'apprentissage
                    loss='categorical_crossentropy', # mesure de l'erreur 
                    metrics=['accuracy']) # mesure du taux de succès

# APPRENTISSAGE du réseau
# => calcul des paramètres du réseau à partir des exemples
hist=MyNetwork.fit(x=x_train, # données d'entrée pour l'apprentissage
                    y=y_train, # sorties désirées associées aux données d'entrée 
                    epochs=15, # nombre de cycles d'apprentissage 
                    validation_data=(x_test,y_test)) # données de test


# GRAPHIQUE pour analyser l'évolution de l'apprentissage
# => courbes erreurs / fiabilité au cours des cycles d'apprentissage

# création de la figure ('figsize' pour indiquer la taille) 
plt.figure(figsize=(8,8))
# evolution du pourcentage des bonnes classifications
plt.subplot(2,1,1)
plt.plot(hist.history['accuracy'],'o-') 
plt.plot(hist.history['val_accuracy'],'x-')
plt.title("Taux d'exactitude des prévisions",fontsize=15)
plt.ylabel('Taux exactitude',fontsize=12)
plt.xlabel("Itérations d'apprentissage",fontsize=15) 
plt.legend(['apprentissage', 'validation'], loc='lower right',fontsize=12) # Evolution des valeurs de l'erreur résiduelle moyenne
plt.subplot(2,1,2)
plt.plot(hist.history['loss'],'o-') 
plt.plot(hist.history['val_loss'],'x-')
plt.title('Erreur résiduelle moyenne',fontsize=15) 
plt.ylabel('Erreur',fontsize=12)
plt.xlabel("Itérations d'apprentissage",fontsize=15) 
plt.legend(['apprentissage', 'validation'], loc='upper right',fontsize=12) # espacement entre les 2 figures
plt.tight_layout(h_pad=2.5)
plt.show()

#SAUVEGARDE du réseau après apprentissage
# => stockage dans un fichier de la description du réseau et des paramètres

# Utilisation de la méthode 'save' de la classe 'tf.keras.Sequential'
MyNetwork.save('MyNetwork.keras')
# Utilisation de la fonction 'save_model' du module 'tf.keras.models' 
tf.keras.models.save_model(MyNetwork, "MyNetwork.keras")


#----------------------------------------------------------------------------
# CHARGEMENT d'un réseau depuis un fichier
#  => pour exploiter le réseau avec les paramètres calculés par apprentissage 
#----------------------------------------------------------------------------
MyNetwork=tf.keras.models.load_model("MyNetwork.keras")
# Affichage des caractéristiques du réseau
print('\nCARACTERISTIQUES DU RESEAUX:')
print('==============================')
MyNetwork.summary()


#----------------------------------------------------------------------------
# EXPLOITATION du réseau
#  => calcul des sorties associées à une image transmise en entrée
#----------------------------------------------------------------------------
# N° de l'exemple à tester dans la base de tests
i=25
# y_test est un tableau de booléens avec une seule valeur à 1
# => argmax() retourne la position du 1 qui correspond au chiffre cherché
print('\nCALCUL DES SORTIES ASSOCIEES A UNE ENTREE:')
print('============================================')
print("Test N°{} => chiffre attendu {}".format(i,y_test[i].argmax()))
print("Résultat du réseau :")
# Utilisation des fonctions "predict" associées à l'objet MyNetwork
#  => Entrée: un tableau de vecteurs (ici un seul vecteur x_test[i:i+1])
#  => Sortie: un tableau avec les sorties pour chaque vecteur d'entrée
print("Sorties brutes:",  MyNetwork.predict(x_test[i:i+1])[0])
#print("Classe de sortie:",MyNetwork.predict_classes(x_test[i:i+1])[0],'\n')
print("Classe de sortie:",MyNetwork.predict(x_test[i:i+1])[0].argmax(),'\n')



#----------------------------------------------------------------------------
# EXPLOITATION du réseau
#  => affichage des exemples de caractères bien et mal reconnus
#----------------------------------------------------------------------------
print('FIABILITE DU RESEAU:')
print('====================')
# Résultat du réseau avec des données de tests
perf=MyNetwork.evaluate(x=x_test, # données d'entrée pour le test
                        y=y_test) # sorties désirées pour le test
print("Taux d'exactitude sur le jeu de test: {:.2f}%".format(perf[1]*100))
NbErreurs=int(10000*(1-perf[1]))
print("==>",NbErreurs," erreurs de classification !")
print("==>",10000-NbErreurs," bonnes classifications !")
# Calcul des prédictions du réseaux pour l'ensemble des données de test
Predictions=MyNetwork.predict(x_test)
# Affichage des caractères bien/mal reconnus avec une matrice d'images
i=-1
Couleur='Green' # à remplacer par 'Green' pour les bonnes reconnaissances
plt.figure(figsize=(12,8), dpi=200)
for NoImage in range(12*8):
    i=i+1
    # '!=' pour les bonnes reconnaissances, '==' pour les erreurs
    while y_test[i].argmax() != Predictions[i].argmax(): i=i+1
    plt.subplot(8,12,NoImage+1)
    # affichage d'une image de digit, en format niveau de gris
    plt.imshow(x_test[i].reshape(28,28), cmap='Greys', interpolation='none')
    # affichage du titre (utilisatin de la méthode format du type str)
    plt.title("Prédit:{} - Correct:{}".format(MyNetwork.predict(
                        x_test[i:i+1])[0].argmax(),y_test[i].argmax()),
                        pad=2,size=5, color=Couleur)
    # suppression des graduations sur les axes X et Y
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
# Affichage de la figure
plt.show()








