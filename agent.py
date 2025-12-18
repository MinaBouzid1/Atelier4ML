import numpy as np
import random
import pickle

class QLearningAgent:
    """Agent utilisant l'algorithme Q-learning pour jouer à PONG"""

    def __init__(self, state_size=4, action_size=3, learning_rate=0.1,
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        """
        Initialise l'agent Q-learning
        Args:
            state_size: Taille de l'espace d'état discrétisé
            action_size: Nombre d'actions possibles (0=rester, 1=haut, 2=bas)
            learning_rate: Taux d'apprentissage (alpha)
            discount_factor: Facteur d'actualisation (gamma)
            epsilon: Taux d'exploration initial
            epsilon_decay: Décroissance du taux d'exploration
            epsilon_min: Taux d'exploration minimum
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: dictionnaire pour stocker les valeurs Q
        # Clé: état discrétisé, Valeur: array des Q-values pour chaque action
        self.q_table = {}

        # Statistiques d'apprentissage
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.losses = 0

    def get_q_values(self, state):
        """Retourne les Q-values pour un état donné"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        return self.q_table[state]

    def get_action(self, state, training=True):
        """
        Choisit une action selon la politique epsilon-greedy
        Args:
            state: État actuel (discrétisé)
            training: Si True, utilise epsilon-greedy, sinon greedy
        Returns:
            action: Action choisie (0, 1, 2)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values)

    def update_q_value(self, state, action, reward, next_state, done):
        """Met à jour la Q-value selon l'équation de Bellman"""
        current_q = self.get_q_values(state)[action]

        if done:
            target_q = reward
        else:
            next_q_values = self.get_q_values(next_state)
            max_next_q = np.max(next_q_values)
            target_q = reward + self.discount_factor * max_next_q

        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """Réduit le taux d'exploration après chaque épisode"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename='q_learning_model.pkl'):
        """Sauvegarde la Q-table et les paramètres de l'agent"""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'wins': self.wins,
            'losses': self.losses
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Modèle sauvegardé dans {filename}")

    def load_model(self, filename='q_learning_model.pkl'):
        """Charge une Q-table et des paramètres sauvegardés"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            self.q_table = model_data['q_table']
            self.epsilon = model_data['epsilon']
            self.episode_rewards = model_data.get('episode_rewards', [])
            self.wins = model_data.get('wins', 0)
            self.losses = model_data.get('losses', 0)
            print(f"Modèle chargé depuis {filename}")
        except FileNotFoundError:
            print(f"Fichier {filename} non trouvé. Initialisation d'un nouveau modèle.")

    def add_episode_stats(self, total_reward, episode_length, won):
        """Enregistre les statistiques d'un épisode"""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        if won:
            self.wins += 1
        else:
            self.losses += 1

    def get_stats(self):
        """Retourne les statistiques d'apprentissage"""
        if len(self.episode_rewards) == 0:
            return {
                'avg_reward': 0,
                'total_episodes': 0,
                'win_rate': 0,
                'q_table_size': len(self.q_table)
            }
        return {
            'avg_reward': np.mean(self.episode_rewards[-100:]),
            'total_episodes': len(self.episode_rewards),
            'win_rate': self.wins / (self.wins + self.losses) if (self.wins + self.losses) > 0 else 0,
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon
        }

    def print_stats(self, episode):
        """Affiche les statistiques actuelles"""
        stats = self.get_stats()
        print(f"Épisode {episode}")
        print(f"  Récompense moyenne (100 derniers): {stats['avg_reward']:.2f}")
        print(f"  Taux de victoire: {stats['win_rate']*100:.1f}%")
        print(f"  Epsilon: {stats['epsilon']:.3f}")
        print(f"  Taille Q-table: {stats['q_table_size']}")
        print("-" * 50)
