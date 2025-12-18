import pygame
import matplotlib.pyplot as plt
import numpy as np
from game import PongGame
from agent import QLearningAgent

def plot_agent_reward(rewards, window_size=100):
    """
    Affiche le graphique des récompenses de l'agent
    Args:
        rewards: Liste des récompenses par épisode
        window_size: Taille de la fenêtre pour la moyenne mobile
    """
    plt.figure(figsize=(12, 6))

    # Graphique des récompenses brutes
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, label='Récompense par épisode')

    # Calcul de la moyenne mobile
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg,
                 linewidth=2, label=f'Moyenne mobile ({window_size} épisodes)')

    plt.xlabel('Épisode')
    plt.ylabel('Récompense totale')
    plt.title("Évolution des récompenses pendant l'entraînement")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Graphique de la distribution des récompenses
    plt.subplot(1, 2, 2)
    plt.hist(rewards, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Récompense totale')
    plt.ylabel('Fréquence')
    plt.title("Distribution des récompenses")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('agent_rewards.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Graphique sauvegardé: agent_rewards.png")

def train_agent(mode='RL_vs_AI', num_episodes=1000, render_every=100, save_every=100):
    """
    Entraîne l'agent Q-learning
    Args:
        mode: Mode de jeu ('RL_vs_AI', 'RL_vs_Human', 'RL_vs_RL')
        num_episodes: Nombre d'épisodes d'entraînement
        render_every: Afficher le jeu tous les N épisodes
        save_every: Sauvegarder le modèle tous les N épisodes
    """
    print(f"=== ENTRAÎNEMENT MODE: {mode} ===")
    print(f"Nombre d'épisodes: {num_episodes}")
    print("-" * 50)

    # Initialisation du jeu et de l'agent
    game = PongGame(mode=mode)
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    # Agent adversaire pour le mode RL vs RL
    opponent_agent = None
    if mode == 'RL_vs_RL':
        opponent_agent = QLearningAgent(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )

    # Boucle d'entraînement
    for episode in range(1, num_episodes + 1):
        state = game.reset()
        discrete_state = game.discretize_state()
        total_reward = 0
        steps = 0
        done = False

        render = (episode % render_every == 0)

        while not done:
            # Gestion des événements Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.close()
                    return agent

            # Gestion de l'entrée humaine si nécessaire
            if mode == 'RL_vs_Human' and render:
                game.handle_human_input()

            # Choix de l'action par l'agent RL
            action = agent.get_action(discrete_state, training=True)

            # Action de l'adversaire en mode RL vs RL
            if mode == 'RL_vs_RL' and opponent_agent:
                opponent_discrete_state = game.discretize_state()
                opponent_action = opponent_agent.get_action(opponent_discrete_state, training=True)
                if opponent_action == 1:
                    game.opponent_paddle.move_up()
                elif opponent_action == 2:
                    game.opponent_paddle.move_down()

            # Exécution de l'action
            next_state, reward, done = game.step(action)
            next_discrete_state = game.discretize_state()

            # Mise à jour de la Q-table
            agent.update_q_value(discrete_state, action, reward, next_discrete_state, done)

            # Mise à jour pour l'adversaire RL
            if mode == 'RL_vs_RL' and opponent_agent:
                opponent_reward = -reward
                opponent_agent.update_q_value(opponent_discrete_state, opponent_action,
                                             opponent_reward, next_discrete_state, done)

            total_reward += reward
            steps += 1
            discrete_state = next_discrete_state

            # Affichage
            if render:
                game.render()
                game.clock.tick(60)

        # Enregistrement des statistiques
        won = game.player_score > game.opponent_score
        agent.add_episode_stats(total_reward, steps, won)

        # Décroissance de l'epsilon
        agent.decay_epsilon()
        if opponent_agent:
            opponent_agent.decay_epsilon()

        # Affichage périodique
        if episode % 50 == 0:
            agent.print_stats(episode)

        # Sauvegarde périodique
        if episode % save_every == 0:
            agent.save_model(f'q_learning_{mode}_{episode}.pkl')

    # Sauvegarde finale
    agent.save_model(f'q_learning_{mode}_final.pkl')

    # Affichage des graphiques
    plot_agent_reward(agent.episode_rewards)

    game.close()
    return agent

def test_agent(mode='RL_vs_AI', model_file='q_learning_RL_vs_AI_final.pkl', num_games=10):
    """
    Teste un agent entraîné
    Args:
        mode: Mode de jeu
        model_file: Fichier du modèle à charger
        num_games: Nombre de parties à jouer
    """
    print(f"=== TEST MODE: {mode} ===")
    print(f"Chargement du modèle: {model_file}")
    print("-" * 50)

    game = PongGame(mode=mode)
    agent = QLearningAgent()
    agent.load_model(model_file)
    agent.epsilon = 0  # Pas d'exploration en test

    wins = 0
    total_rewards = []

    for game_num in range(1, num_games + 1):
        state = game.reset()
        discrete_state = game.discretize_state()
        total_reward = 0
        done = False

        print(f"\nPartie {game_num}/{num_games}")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.close()
                    return

            if mode == 'RL_vs_Human':
                game.handle_human_input()

            action = agent.get_action(discrete_state, training=False)
            next_state, reward, done = game.step(action)
            next_discrete_state = game.discretize_state()

            total_reward += reward
            discrete_state = next_discrete_state

            game.render()
            game.clock.tick(60)

        won = game.player_score > game.opponent_score
        if won:
            wins += 1

        total_rewards.append(total_reward)
        print(f"  Score: {game.player_score} - {game.opponent_score}")
        print(f"  Résultat: {'VICTOIRE' if won else 'DÉFAITE'}")
        print(f"  Récompense totale: {total_reward:.2f}")

    print("\n" + "=" * 50)
    print(f"RÉSULTATS FINAUX:")
    print(f"  Victoires: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
    print(f"  Récompense moyenne: {np.mean(total_rewards):.2f}")
    print("=" * 50)

    game.close()

def main():
    """Fonction principale"""
    print("=" * 50)
    print("PONG Q-LEARNING - ATELIER 4")
    print("=" * 50)
    print("\nChoisissez une option:")
    print("1. Entraîner agent (RL vs AI)")
    print("2. Entraîner agent (RL vs Humain)")
    print("3. Entraîner agent (RL vs RL)")
    print("4. Tester agent entraîné")
    print("5. Quitter")

    choice = input("\nVotre choix (1-5): ")

    if choice == '1':
        train_agent(mode='RL_vs_AI', num_episodes=1000)
    elif choice == '2':
        train_agent(mode='RL_vs_Human', num_episodes=500)
    elif choice == '3':
        train_agent(mode='RL_vs_RL', num_episodes=1000)
    elif choice == '4':
        mode = input("Mode (RL_vs_AI/RL_vs_Human/RL_vs_RL): ")
        model_file = input("Fichier du modèle: ")
        test_agent(mode=mode, model_file=model_file, num_games=5)
    elif choice == '5':
        print("Au revoir!")
    else:
        print("Choix invalide!")

if __name__ == "__main__":
    main()
