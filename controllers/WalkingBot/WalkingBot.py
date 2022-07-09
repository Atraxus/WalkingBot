from controller import Robot, Device, Node, Motor, Accelerometer, Supervisor
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy


# Input vector of network
@dataclass
class InVec:
    leftAngle1: float
    leftAngle2: float
    rightAngle1: float
    rightAngle2: float

    leftVelocity: float
    leftVelocity2: float
    rightVelocity: float
    rightVelocity2: float

    accelX: float
    accelY: float
    accelZ: float

@dataclass
class OutVec:
    torqueLeftMotor1: float
    torqueLeftMotor2: float
    torqueRightMotor1: float
    torqueRightMotor2: float


# Simple network that connects sensor inputs directly to motor torques.
class SimpleNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNetwork, self).__init__()
        # maybe we should add another layer? one could be to weak for evolution maybe
        self.inoutTransform = torch.nn.Linear(input_size, output_size)
        self.fitness = 0

    def forward(self, x):
        # send through linear
        features = self.inoutTransform(x)
        # return softmax classification (can also use relu or or or..., dim may has to be adjusted)
        return F.log_softmax(features, dim=1)


class Agent:
    def __init__(self, network):


class WalkingBot(Robot):
    m_timeStep: int
    
    m_LeftMotor1: Motor
    m_LeftMotor2: Motor
    m_RightMotor1: Motor
    m_RightMotor2: Motor
    
    m_Accelerometer: Accelerometer

    # Network
    m_network: SimpleNetwork # TODO(Jannis): init with specific nn

    ############################################# new ##############################################
    # get weights of a trained!! model using
    weights = m_network.parameters()
    # you can pass a weight matrix by doing:
    m_network.apply(weights)
    ################################################################################################


    def initialize(self):
        self.timeStep = 32
        # Get motors
        self.m_LeftMotor1 = self.getDevice('LeftLegMotor1')
        self.m_LeftMotor2 = self.getDevice('LeftLegMotor2')
        self.m_RightMotor1 = self.getDevice('RightLegMotor1')
        self.m_RightMotor2 = self.getDevice('RightLegMotor2')
        # Get sensors
        self.m_Accelerometer = self.getDevice('Accelerometer')
        self.m_Accelerometer.enable(self.timeStep)
        # Get network
        self.m_network = SimpleNetwork(12, 4)

    def run(self):
        while self.step(self.timeStep) != -1:
            value = self.m_Accelerometer.getValues()
            #TODO(Jannis): run loop sensordata -> network -> motor torques

    def printDevices(self):
        numDevices = self.getNumberOfDevices()
        print("Number of devices:", numDevices)
        for i in range(numDevices):
            print("Device", i, ":", self.getDeviceByIndex(i).getName())



############################## new ##################################
# I don't know if the following should be done in your run, relocate if necessary
# das haut von der struktur noch nicht hin, class und defs müssen noch geordnet werden

# initialize population randomly
population_size = 10
population = numpy.random.uniform(low=-4.0, high=4.0, size=population_size)

def generate_agents(population, network):
    # each agent represents one robot
    # could initialize network here with different random weights or similar with apply funtion
    return [Agent(network) for _ in range(population)]

def select(self, agents):
    # different approaches possible, maybe google
    # here for know: only select x first, others die and are not used anymore
    agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
    # get number of agents selected proportional to number of agents
    agents = agents[:int(0.2 * len(agents))]
    return agents

def crossover(self, agents, network, population_size):
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
        child1.SimpleNetwork.apply() = child1_genes
        child2.SimpleNetwork.apply() = child2_genes

        offspring.append(child1)
        offspring.append(child2)
        # append childs two population
        agents.extend(offspring)
        return agents
    return agents

def mutation(agents):
    # after crossover mutation is common: flip a few parameters of offspring with very low propability
    # add the offspring to the population
    for agent in agents:
        # mutate with a very low propability
        if random.uniform(0.0, 1.0) <= 0.1:
            weights = agent.SimpleNetwork.parameters().numpy()
            # TODO: generate random index in range weights and replace it with random new value
            agent.SimpleNetwork.apply() = torch.from_numpy(weights)
    return agents


def calculate_fitness(self, agent, meters_walked, time, penalty):
    # penalty for falling or standing still
    # walked meters, time (or constant?)
    # save score in object? (neural network)
    agent.fitness = (meters_walked / time) - penalty
    return agent


def train(self):
    generations = 5
    agents = generate_agents(population_size, m_network)
    # a little bit lost here
    # I think we have to do the following:
    for i in range(generations):
        # is this correct? do we do the evolution <generations> times?
        for agent in agents:
            # run forward pass of agents neural network to get motor controls
            motor_controls = agent.m_network.forward(input_vec)
            # TODO: somehow run simulation with these motor controls???
            # calculate the fitness of the agent (robot)
            meters = 0
            time = 0
            penalty = 0
            agent = calculate_fitness(self, agent, meters, time, penalty)

        # select best agents
        agents = select(self, agents)
        # do crossover
        agents = crossover(self, agents, m_network, population_size)
        agents = mutation(agents)
        # now we have a new generation of agents but never do loss calculation or backward pass in nn of agents?? is that correct??


####################################################################



# Torch
x = torch.rand(5, 3)
print(x)
# For testing purposes
net = SimpleNetwork(12, 4) # TODO(Jannis): random numbers?
controller = WalkingBot()
controller.printDevices()
controller.initialize()
controller.run()






# Motors


# Sensors
# accelerometer = robot.getDevice("accelerometer")
# accelerometer.enable(TIME_STEP)

# Main Loop to get the robot to walk
# while robot.step(TIME_STEP) != -1:
#     i = 0
    # Read in sensordata 