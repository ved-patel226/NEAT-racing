import gym
import neat
import numpy as np

# Define the fitness function
def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make('CartPole-v1')
    total_reward = 0
    done = False

    observation = env.reset()
    while not done:
        action = net.activate(observation)
        action = np.argmax(action)  # Choose the action with the highest output
        observation, reward, done, _ = env.step(action)
        total_reward += reward

    env.close()
    return total_reward

def run(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Create the population
    population = neat.Population(config)

    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.Checkpointer(5))

    # Run NEAT algorithm
    winner = population.run(evaluate_genome, 50)

    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    run('neat.config')
