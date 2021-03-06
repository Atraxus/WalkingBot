from math import tanh
from time import sleep
from controller import Supervisor, Motor, Accelerometer, GPS
import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt


# Network defines:
#   InVec
#     leftAngle1: float
#     leftAngle2: float
#     rightAngle1: float
#     rightAngle2: float
#     leftVelocity: float
#     leftVelocity2: float
#     rightVelocity: float
#     rightVelocity2: float
#     accelX: float
#     accelY: float
#     accelZ: float
#     gain: float
#   OutVec
#     torqueLeftMotor1: float
#     torqueLeftMotor2: float
#     torqueRightMotor1: float
#     torqueRightMotor2: float


# Simple network that connects sensor inputs directly to motor torques.
class SimpleNetwork(torch.nn.Module):
    m_inputSize: int
    m_outputSize: int
    m_fitness: float

    def __init__(self, input_size: int, output_size: int, weights: torch.Tensor, biases: torch.Tensor):
        self.m_inputSize = input_size
        self.m_outputSize = output_size
        self.m_fitness = 0.0
        super(SimpleNetwork, self).__init__()
        # maybe we should add another layer? one could be to weak for evolution maybe
        self.inoutTransform = torch.nn.Linear(input_size, output_size)
        self.inoutTransform.weight.data = weights
        self.inoutTransform.bias.data = biases
        self.fitness = 0

    def forward(self, x: torch.Tensor):
        # send through linear
        features = self.inoutTransform(x)
        # return softmax classification (can also use relu or or or..., dim may has to be adjusted)
        return F.log_softmax(features, dim=0)


class WalkingBot(Supervisor):
    m_timeStep: int

    m_LeftMotor1: Motor
    m_LeftMotor2: Motor
    m_RightMotor1: Motor
    m_RightMotor2: Motor

    m_Accelerometer: Accelerometer
    m_GPS: GPS

    # Network
    m_Network: SimpleNetwork

    def initialize(self, timeStep: int = 32):
        self.m_timeStep = timeStep
        # Get motors
        self.m_LeftMotor1 = self.getDevice('LeftLegMotor1')
        self.m_LeftMotor2 = self.getDevice('LeftLegMotor2')
        self.m_RightMotor1 = self.getDevice('RightLegMotor1')
        self.m_RightMotor2 = self.getDevice('RightLegMotor2')
        # Get sensors
        self.m_Accelerometer = self.getDevice('Accelerometer')
        self.m_Accelerometer.enable(timeStep)
        self.m_GPS = self.getDevice('gps')
        self.m_GPS.enable(timeStep)
        # Get parameters
        self.m_LeftLegSensor1 = self.getDevice('LeftLegSensor1')
        self.m_LeftLegSensor1.enable(timeStep)
        self.m_LeftLegSensor2 = self.getDevice('LeftLegSensor2')
        self.m_LeftLegSensor2.enable(timeStep)
        self.m_RightLegSensor1 = self.getDevice('RightLegSensor1')
        self.m_RightLegSensor1.enable(timeStep)
        self.m_RightLegSensor2 = self.getDevice('RightLegSensor2')
        self.m_RightLegSensor2.enable(timeStep)

    def run(self, runtime: int):
        while self.step(self.m_timeStep) != -1:
            value = self.m_Accelerometer.getValues()
            # get angles and velocities of motors
            leftAngle1 = self.m_LeftLegSensor1.getValue()
            leftAngle2 = self.m_LeftLegSensor2.getValue()
            rightAngle1 = self.m_RightLegSensor1.getValue()
            rightAngle2 = self.m_RightLegSensor2.getValue()
            leftVelocity = self.m_LeftMotor1.getVelocity()
            leftVelocity2 = self.m_LeftMotor2.getVelocity()
            rightVelocity = self.m_RightMotor1.getVelocity()
            rightVelocity2 = self.m_RightMotor2.getVelocity()
            # get accelerometer values
            accelX = value[0]
            accelY = value[1]
            accelZ = value[2]
            # gain
            gain = 0.1
            # create input vector
            inVec = torch.Tensor([leftAngle1, leftAngle2, rightAngle1, rightAngle2, leftVelocity, leftVelocity2, rightVelocity,
                 rightVelocity2, accelX, accelY, accelZ, gain])

            # network
            outVals = self.m_Network.forward(inVec.float())

            # get torque values
            torqueLeftMotor1 = tanh(outVals[0]) * 10
            torqueLeftMotor2 = tanh(outVals[1]) * 10
            # torqueRightMotor1 = tanh(outVals[2]) * 10
            # torqueRightMotor2 = tanh(outVals[3]) * 10

            

            # apply torque values
            self.m_LeftMotor1.setTorque(float(torqueLeftMotor1))
            self.m_LeftMotor2.setTorque(float(torqueLeftMotor2))
            self.m_RightMotor1.setTorque(float(torqueLeftMotor1))
            self.m_RightMotor2.setTorque(float(torqueLeftMotor2))

            if self.getTime() > runtime:
                self.m_LeftMotor1.setTorque(0)
                self.m_LeftMotor2.setTorque(0)
                self.m_RightMotor1.setTorque(0)
                self.m_RightMotor2.setTorque(0)
                return

    def printDevices(self):
        numDevices = self.getNumberOfDevices()
        print("Number of devices:", numDevices)
        for i in range(numDevices):
            print("\tDevice", i, ":", self.getDeviceByIndex(i).getName())

def net_fitness(net: SimpleNetwork):
    return net.m_fitness

def select(population: list, selection_size: int = 10):
    # sort by fitness
    population.sort(key=net_fitness, reverse=True)
    # get number of population according to selection size
    population = population[:selection_size]
    return population


def spliceLists(parent1: np.ndarray, parent2: np.ndarray, ):
    # weights of neural network of agent
    # maybe have to flatten a beforehand, we gotta try
    values1 = parent1.ravel()
    values2 = parent2.ravel()

    # randomly choose crossover point
    split = random.randint(0, len(values1) - 1)

    returnValues1 = np.asarray(values1[0:split].tolist() + values2[split:].tolist())
    returnValues2 = np.asarray(values1[0:split].tolist() + values2[split:].tolist())

    # convert back to tensor, maybe gotta reshape
    return returnValues1, returnValues2


def crossover(currPopulation: list, population_size: int):
    # cross two individuals, I guess this should be done with all selected
    # so f.e. 4 individuals: 1 with 2, 1 with 3, 1 with 4, 2 with 3, 2 with 4, 3 with 4 = 6*2 = 12 (see below)
    # common crossover technique: choose crossover point randomly
    # exchange genes (e.g. parameters) until crossover point is reached
    # example:
    # individual 1: [111111], individual 2: [000000]
    # crossover point: 2
    # child 1: [001111], child 2: [110000]
    offspring = []
    for _ in range(int((population_size - len(currPopulation))/2)):
        parent1 = random.choice(currPopulation)
        parent2 = random.choice(currPopulation)
        # get dimensions of parent
        inSize = parent1.m_inputSize
        outSize = parent1.m_outputSize

        # get new crossover weights
        weights1, weights2 = spliceLists(parent1.inoutTransform.weight, parent2.inoutTransform.weight)
        # get new crossover biases
        biases1, biases2 = spliceLists(parent1.inoutTransform.bias, parent2.inoutTransform.bias)
        
        weights1 = weights1.reshape(outSize, inSize)
        weights2 = weights2.reshape(outSize, inSize)

        # Convert to tensor
        weights1 = torch.from_numpy(weights1).float()
        weights2 = torch.from_numpy(weights2).float()
        biases1 = torch.from_numpy(biases1).float()
        biases2 = torch.from_numpy(biases2).float()

        child1 = SimpleNetwork(inSize, outSize, weights1, biases1)
        child2 = SimpleNetwork(inSize, outSize, weights2, biases2)

        offspring.append(child1)
        offspring.append(child2)
    return offspring

def crossover2(currPopulation: list, population_size: int):
    offspring = []
    for _ in range(population_size - len(currPopulation)):
        parent1 = random.choice(currPopulation)
        parent2 = random.choice(currPopulation)
        parent1Weights = parent1.inoutTransform.weight.data
        parent2Weights = parent2.inoutTransform.weight.data
        parent1Biases = parent1.inoutTransform.bias.data
        parent2Biases = parent2.inoutTransform.bias.data
        # get dimensions of parent
        inSize = parent1.m_inputSize
        outSize = parent1.m_outputSize

        childWeights = torch.zeros(outSize, inSize)
        # get new crossover weights
        for i in range(outSize):
            for j in range(inSize):
                if random.random() < 0.5:
                    childWeights[i][j] = parent1Weights[i][j]
                else:
                    childWeights[i][j] = parent2Weights[i][j]
        # get new crossover biases
        childBiases = torch.zeros(outSize)
        for i in range(outSize):
            if random.random() < 0.5:
                childBiases[i] = parent1Biases[i]
            else:
                childBiases[i] = parent2Biases[i]
        childWeights = childWeights.reshape(outSize, inSize)
        child = SimpleNetwork(inSize, outSize, childWeights, childBiases)
        offspring.append(child)
    return offspring





def mutation(population: list, mutation_rate: float = 0.1):
    # after crossover mutation is common: flip a few parameters of offspring with very low propability
    # add the offspring to the population
    for pop in population:
        # mutate with a very low propability
        weights = pop.inoutTransform.weight.data
        for i in range(len(weights)):
            if random.uniform(0.0, 1.0) < mutation_rate:
                weights[i] = random.uniform(-1.0, 1.0)
    return population


def calculate_fitness(meters_walked, time, penalty=0):
    # penalty for falling or standing still
    # walked meters, time (or constant?)
    # save score in object? (neural network)
    return (meters_walked / time) - penalty

def max_fitness(networks):
    max_fitness = 0
    for net in networks:
        if net.m_fitness > max_fitness:
            max_fitness = net.m_fitness
    return max_fitness

# General definitions
print("Starting...")
timestep = 10
bot = WalkingBot()
bot.initialize(timestep)

inputSize = 12
outputSize = 2

population_size = 20
networks = []
best_fitnesses = []
epochs = 300
runtime = 10

for _ in range(population_size):
    weights = np.random.uniform(-0.5, 0.5, (outputSize, inputSize))  # TODO(Jannis): add negative values
    biases = np.random.uniform(-0.5, 0.5, outputSize)
    weights = torch.from_numpy(weights).float()
    biases = torch.from_numpy(biases).float()
    net = SimpleNetwork(inputSize, outputSize, weights, biases)
    networks.append(net)

bot.simulationSetMode(2)
for i in range(epochs):
    print("Epoch:", i)

    # run for each pop in population
    for j, net in enumerate(networks):
        bot.m_Network = net
        bot.run(runtime)
        position = bot.m_GPS.getValues()
        print("\tNetwork: %d - travelled: %.2f meters" % (j, position[1]))
        fitness = calculate_fitness(position[1], runtime)
        networks[j].m_fitness = fitness
        # reset simulation
        bot.simulationReset()

    # get best fitness and save to best fitnesses
    best_fitness = max_fitness(networks)
    best_fitnesses.append(best_fitness)

    # select best population
    networks = select(networks, 10)

    # crossover population
    offspring = crossover(networks, population_size)
    networks = networks + offspring

    # mutate population
    networks = mutation(networks, 0.1)

def plot_fitness(fitnesses):
    # pass list of best fitnesses (best per gen)
    generations = list(range(1,epochs+1))
    plt.plot(generations, fitnesses)
    plt.title('Fitness per Generation')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.savefig('fitness_per_gen.png')

print("Exporting fitness plot... ")
plot_fitness(best_fitnesses)

bot.simulationReset()
bot.simulationSetMode(1)


# select best, worst and median networks
networks.sort(key=net_fitness, reverse=True)
best = networks[0]
worst = networks[-1]
median = networks[int(population_size/2)]
videoNets = [best, worst, median]

for j, net in enumerate(videoNets):
    # wait for move to be ready
    while bot.movieIsReady() == 0:
        sleep(1)
        pass

    filename = "net_" + str(j) + ".mp4"
    # record video
    bot.movieStartRecording(filename, 1920, 1080, 1, 80, 1, True)
    bot.m_Network = net
    bot.run(runtime)
    bot.movieStopRecording()

    bot.simulationReset()