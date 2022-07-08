from controller import Robot, Device, Node, Motor, Accelerometer, Supervisor
from dataclasses import dataclass
import torch
import torch.nn.functional as F


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

    def forward(self, x):
        # send through linear
        features = self.inoutTransform(x)
        # return softmax classification (can also use relu or or or..., dim may has to be adjusted)
        return F.log_softmax(features, dim=1)



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

    # initialize population
    population = []

    def select(self, individuals_with_scores):
        # different approaches possible, example: steady state
        # individuals with highest score are selected (f.e. 4)
        # few individuals with lowest score (in this example twelve) are removed and replaced with children of highest
        # remaining individuals stay in population
        # return new population without weakest and selected individuals
        return [parents], population

    def crossover(self, individual_1, individual_2):
        # cross two individuals, I guess this should be done with all selected
        # so f.e. 4 individuals: 1 with 2, 1 with 3, 1 with 4, 2 with 3, 2 with 4, 3 with 4 = 6*2 = 12 (see below)
        # common crossover technique: choose crossover point randomly
        # exchange genes (e.g. parameters) until crossover point is reached
        # example:
        # individual 1: [111111], individual 2: [000000]
        # crossover point: 2
        # child 1: [001111], child 2: [110000]

        # after that mutation is common: flip a few parameters of offspring with very low probability
        # add the offspring to the population
        return population


    def calculate_fitness(self, individual):
        # fitness function, can be based on distance traveled in given time interval
        # plus penalty for falling or standing still
        score = 0
        # set score in relation two individual (maybe dict?)
        return score


    # define loss function and optimiizer, here very simple one
    def train(self):
        # TODO: get sensor data as torch.tensor size (12,1) or other way round?
        # I think input vec is individual?
        input_vec = []
        # add input vec to population?
        # run forward pass
        motor_controls = self.m_network.forward(input_vec)
        # TODO: run simulation with motor_controls
        # calculate fitness score for individual
        scores = calculate_fitness(input_vec)
        selected_individuals, population = select(scores)
        # for loop to do described crossover everyone with everyone
        population = crossover(selected_individuals[0], selected_individuals[1])
        # restart with new population


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