from pickle import TRUE
from time import sleep
from controller import Supervisor, Motor, Accelerometer, GPS
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np
import random
import time


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
#   OutVec
#     torqueLeftMotor1: float
#     torqueLeftMotor2: float
#     torqueRightMotor1: float
#     torqueRightMotor2: float


# Simple network that connects sensor inputs directly to motor torques.
class SimpleNetwork(torch.nn.Module):
    m_inputSize: int
    m_outputSize: int

    def __init__(self, input_size: int, output_size: int, weights: np.ndarray, biases: np.ndarray):
        self.m_inputSize = input_size
        self.m_outputSize = output_size
        super(SimpleNetwork, self).__init__()
        # maybe we should add another layer? one could be to weak for evolution maybe
        self.inoutTransform = torch.nn.Linear(input_size, output_size)
        self.inoutTransform.weight.data = torch.from_numpy(weights).float()
        self.inoutTransform.bias.data = torch.from_numpy(biases).float()
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
    m_Network: SimpleNetwork  # TODO(Jannis): init with specific nn

    def initialize(self, network: SimpleNetwork, timeStep: int = 32):
        self.m_timeStep = timeStep
        self.m_Network = network
        # Get motors
        self.m_LeftMotor1 = self.getDevice('LeftLegMotor1')
        self.m_LeftMotor2 = self.getDevice('LeftLegMotor2')
        self.m_RightMotor1 = self.getDevice('RightLegMotor1')
        self.m_RightMotor2 = self.getDevice('RightLegMotor2')
        # Get sensors
        self.m_Accelerometer = self.getDevice('Accelerometer')
        self.m_Accelerometer.enable(timeStep)
        self.m_GPS = self.getDevice('gps')
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

            print(outVals)

            # get torque values
            torqueLeftMotor1 = outVals[0][0]
            torqueLeftMotor2 = outVals[0][1]
            torqueRightMotor1 = outVals[0][2]
            torqueRightMotor2 = outVals[0][3]

            # apply torque values
            self.m_LeftMotor1.setTorque(torqueLeftMotor1)
            self.m_LeftMotor2.setTorque(torqueLeftMotor2)
            self.m_RightMotor1.setTorque(torqueRightMotor1)
            self.m_RightMotor2.setTorque(torqueRightMotor2)

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


############################## new ##################################
# I don't know if the following should be done in your run, relocate if necessary
# das haut von der struktur noch nicht hin, class und defs müssen noch geordnet werden

def select(agents, selection_size: int = 10):
    # different approaches possible, maybe google
    # here for know: only select x first, others die and are not used anymore
    agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
    # get number of agents according to selection size
    agents = agents[:selection_size]
    return agents


def spliceLists(parent1: np.ndarray, parent2: np.ndarray, ):
    # weights of neural network of agent
    # maybe have to flatten a beforehand, we gotta try
    values1 = parent1.ravel()
    values2 = parent2.ravel()

    # randomly choose crossover point
    split = random.ragendint(0, len(values1) - 1)

    returnValues1 = np.asrray(values1[0:split].tolist() + values2[split:].tolist())
    returnValues2 = np.array(values1[0:split].tolist() + values2[split:].tolist())

    # convert back to tensor, maybe gotta reshape
    return torch.from_numpy(returnValues1), torch.from_numpy(returnValues2)


def crossover(population: list, population_size: int):
    # cross two individuals, I guess this should be done with all selected
    # so f.e. 4 individuals: 1 with 2, 1 with 3, 1 with 4, 2 with 3, 2 with 4, 3 with 4 = 6*2 = 12 (see below)
    # common crossover technique: choose crossover point randomly
    # exchange genes (e.g. parameters) until crossover point is reached
    # example:
    # individual 1: [111111], individual 2: [000000]
    # crossover point: 2
    # child 1: [001111], child 2: [110000]
    offspring = []
    for _ in range((population_size - len(population)) // 2):
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        # get dimensions of parent
        inSize = len(parent1.m_inputSize)
        outSize = len(parent1.m_outputSize)

        # get new crossover weights
        weights1, weights2 = spliceLists(parent1.inOutTransform.weight, parent2.inoutTransform.weight)
        # get new crossover biases
        biases1, biases2 = spliceLists(parent1.inOutTransform.bias, parent2.inOutTransform.bias)

        child1 = SimpleNetwork(inSize, outSize, weights1, biases1)
        child2 = SimpleNetwork(inSize, outSize, weights2, biases2)

        offspring.append(child1)
        offspring.append(child2)
    return offspring


def mutation(population: list, mutation_rate: float = 0.1):
    # after crossover mutation is common: flip a few parameters of offspring with very low propability
    # add the offspring to the population
    for pop in population:
        # mutate with a very low propability
        weights = pop.inOutTransform.weight.data
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                if random.uniform(0.0, 1.0) < mutation_rate:
                    weights[i][j] = random.random()
    return population


def calculate_fitness(meters_walked, time, penalty=0):
    # penalty for falling or standing still
    # walked meters, time (or constant?)
    # save score in object? (neural network)
    return (meters_walked / time) - penalty


# General definitions
print("Starting...")
bot = WalkingBot()
timestep = 10
inputSize = 12
outputSize = 4

population_size = 10
networks = []
epochs = 100
runtime = 5

for _ in range(population_size):
    print(_)
    weights = np.random.rand(outputSize, inputSize)  # TODO(Jannis): add negative values
    print("weights:", weights)
    biases = np.random.rand(outputSize)
    print("biases:", biases)
    net = SimpleNetwork(inputSize, outputSize, weights, biases)
    networks.append(net)

for i in range(epochs):
    print("Epoch:", i)
    evaluations = []

    # run for each pop in population
    for j, net in enumerate(networks):
        print("\tRunning network:", j)
        bot.initialize(net, timestep)
        bot.run(runtime)
        position = bot.getPosition()
        fitness = calculate_fitness(position[0], runtime)
        evaluations.append(fitness)

    # select best population
    networks = select(networks, population_size, 5)

    # crossover population
    networks = crossover(networks, population_size)

    # mutate population
    networks = mutation(networks, 0.05)

    # reset simulation
    bot.simulationReset()
