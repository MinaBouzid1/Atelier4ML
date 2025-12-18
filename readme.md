# 🏓 Pong Q-Learning (Reinforcement Learning)

Ce projet implémente un jeu **Pong intelligent** utilisant l’algorithme de **Q-Learning**. L’objectif est de montrer comment un agent d’apprentissage par renforcement peut apprendre à jouer au Pong uniquement par interaction avec l’environnement.

---

## 🎯 Objectifs du projet

* Appliquer les concepts de **Reinforcement Learning**
* Implémenter l’algorithme **Q-Learning**
* Créer un environnement de jeu avec **pygame**
* Visualiser l’apprentissage à l’aide de **matplotlib**

---

## 🛠️ Technologies utilisées

* **Python**
* **pygame** – moteur du jeu
* **NumPy** – calculs numériques
* **Matplotlib** – visualisation des performances

---

## 📁 Structure du projet

```
atelier4ML/
│── main.py          # Lancement, entraînement et test
│── game.py          # Logique du jeu Pong
│── agent.py         # Agent Q-Learning
│── README.md        # Documentation
```

---

## ⚙️ Installation

### 1️⃣ Cloner le dépôt

```bash
git clone https://github.com/MinaBouzid1/Atelier4ML.git
cd Atelier4ML
```

### 2️⃣ Installer les dépendances

```bash
pip install pygame 
pip install numpy 
pip install matplotlib    

```


---

## ▶️ Utilisation

Lancer le programme principal :

```bash
python main.py
```

Menu disponible :

* Entraîner l’agent (**RL vs AI**)
* Entraîner l’agent (**RL vs Humain**)
* Entraîner deux agents (**RL vs RL**)
* Tester un agent entraîné

---

## 📊 Visualisation des résultats

À la fin de l’entraînement, des graphiques sont générés avec **Matplotlib** :

* Évolution des récompenses par épisode
* Moyenne mobile (apprentissage)
* Distribution des récompenses

Un fichier `agent_rewards.png` est automatiquement sauvegardé.

---

## 🚀 Conseils de performance

Pour accélérer l’entraînement :

* Désactiver l’affichage pygame pendant l’entraînement
* Réduire le nombre d’épisodes (200–500 suffisent pour l’analyse)

---

## 📌 Améliorations possibles

* Implémenter un **Deep Q-Network (DQN)**
* Améliorer la discrétisation des états
* Ajouter une sauvegarde automatique avancée

---


✨ *Apprentissage par renforcement avec Python*
