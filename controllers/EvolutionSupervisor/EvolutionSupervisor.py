from controller import Supervisor, Robot
from WalkingBot import SimpleNetwork, Agent, startController
import random
import torch
import numpy as np

class EvolSupervisor(Supervisor):
    m_Controller : Robot
    m_population_size : int
    m_population : list

    ############################## new ##################################
    # I don't know if the following should be done in your run, relocate if necessary
    # das haut von der struktur noch nicht hin, class und defs müssen noch geordnet werden
    def generate_agents(population, network: SimpleNetwork):
        # each agent represents one robot
        # could initialize network here with different random weights or similar with apply funtion
        return [Agent(network) for _ in range(population)]

    def select(self, agents, selection_size: int = 10):
        # different approaches possible, maybe google
        # here for know: only select x first, others die and are not used anymore
        agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
        # get number of agents according to selection size
        agents = agents[:selection_size]
        return agents

    def crossover(self, agents, network: SimpleNetwork, population_size: int):
        # cross two individuals, I guess this should be done with all selected
        # so f.e. 4 individuals: 1 with 2, 1 with 3, 1 with 4, 2 with 3, 2 with 4, 3 with 4 = 6*2 = 12 (see below)
        # common crossover technique: choose crossover point randomly
        # exchange genes (e.g. parameters) until crossover point is reached
        # example:
        # individual 1: [111111], individual 2: [000000]
        # crossover point: 2
        # child 1: [001111], child 2: [110000]
        offspring = []
        for _ in range((population_size - len(agents)) // 2):
            parent1 = random.choice(agents)
            parent2 = random.choice(agents)
            child1 = Agent(network)
            child2 = Agent(network)

            # weights of neural network of agent
            # maybe have to flatten a beforehand, we gotta try
            genes1 = np.concatenate([a.numpy() for a in parent1.SimpleNetwork.parameters()])
            genes2 = np.concatenate([a.numpy() for a in parent2.SimpleNetwork.parameters()])

            # randomly choose crossover point
            split = random.ragendint(0, len(genes1) - 1)

            child1_genes = np.asrray(genes1[0:split].tolist() + genes2[split:].tolist())
            child2_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())

            # convert back to tensor, maybe gotta reshape
            child1_genes = torch.from_numpy(child1_genes)
            child2_genes = torch.from_numpy(child2_genes)

            # assign new crossover genes to child
            # if we have to flatten above we will have to unflatten here
            child1.m_Network.weights = child1_genes
            child2.m_Network.weights = child2_genes

            offspring.append(child1)
            offspring.append(child2)
            # append childs two population
            agents.extend(offspring)
            return agents
        return agents

    def mutation(agents: Agent, mutation_rate : float = 0.1):
        # after crossover mutation is common: flip a few parameters of offspring with very low propability
        # add the offspring to the population
        for agent in agents:
            # mutate with a very low propability
            if random.uniform(0.0, 1.0) <= mutation_rate:
                weights = agent.SimpleNetwork.parameters().numpy()
                # TODO: generate random index in range weights and replace it with random new value
                agent.m_Network.weights = torch.from_numpy(weights)
        return agents


    def calculate_fitness(self, agent, meters_walked, time, penalty):
        # penalty for falling or standing still
        # walked meters, time (or constant?)
        # save score in object? (neural network)
        agent.fitness = (meters_walked / time) - penalty
        return agent


    def train(self, agents, epochs: int = 5):
        # a little bit lost here
        # I think we have to do the following:
        for i in range(epochs):
            # is this correct? do we do the evolution <generations> times?
            for agent in agents:
                # run forward pass of agents neural network to get motor controls
                motor_controls = agent.m_network.forward(self.m_inputVector)
                # TODO: somehow run simulation with these motor controls???
                # calculate the fitness of the agent (robot)
                meters = 0
                time = 0
                penalty = 0
                agent = self.calculate_fitness(self, agent, meters, time, penalty)

            # select best agents
            agents = self.select(self, agents)
            # do crossover
            agents = self.crossover(self, agents, self.m_network, population_size)
            agents = self.mutation(agents)
            # now we have a new generation of agents but never do loss calculation or backward pass in nn of agents?? is that correct??


timestep = 32
# initialize population randomly
population_size = 10
population = np.random.uniform(low=-4.0, high=4.0, size=population_size)

# start supervisor
supervisor = EvolSupervisor()


# start extern controller
controller = startController(timestep)