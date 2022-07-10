from time import sleep
from controller import Robot, Device, Node, Motor, Accelerometer, Supervisor
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np
import os

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
    def __init__(self, input_size: int, output_size: int, weights: np.ndarray, biases: np.ndarray):
        super(SimpleNetwork, self).__init__()
        # maybe we should add another layer? one could be to weak for evolution maybe
        self.inoutTransform = torch.nn.Linear(input_size, output_size)
        self.inoutTransform.weight = torch.from_numpy(weights)
        self.inoutTransform.bias = torch.from_numpy(biases)
        self.fitness = 0

    def forward(self, x: torch.Tensor):
        # send through linear
        features = self.inoutTransform(x)
        # return softmax classification (can also use relu or or or..., dim may has to be adjusted)
        return F.log_softmax(features, dim=1)


class Agent:
    m_Network: SimpleNetwork
    m_fitness: float

    def __init__(self, network: SimpleNetwork):
        self.m_Network = network
        self.m_fitness = 0.0


class WalkingBot(Robot):
    m_timeStep: int

    m_LeftMotor1: Motor
    m_LeftMotor2: Motor
    m_RightMotor1: Motor
    m_RightMotor2: Motor

    m_Accelerometer: Accelerometer

    # Network
    m_Network: SimpleNetwork  # TODO(Jannis): init with specific nn


    def initialize(self, timeStep: int = 32):
        self.m_timeStep = timeStep
        self.m_Network = None  #! robot needs to wait for network in run loop
        # Get motors
        self.m_LeftMotor1 = self.getDevice('LeftLegMotor1')
        self.m_LeftMotor2 = self.getDevice('LeftLegMotor2')
        self.m_RightMotor1 = self.getDevice('RightLegMotor1')
        self.m_RightMotor2 = self.getDevice('RightLegMotor2')
        # Get sensors
        self.m_Accelerometer = self.getDevice('Accelerometer')
        self.m_Accelerometer.enable(self.timeStep)

    def run(self):
        # Get the network
        while self.m_Network is None:
            sleep(0.1)
            continue
        while self.step(self.timeStep) != -1:
            value = self.m_Accelerometer.getValues()
            # get angles and velocities of motors
            leftAngle1 = self.m_LeftMotor1.getPosition()
            leftAngle2 = self.m_LeftMotor2.getPosition()
            rightAngle1 = self.m_RightMotor1.getPosition()
            rightAngle2 = self.m_RightMotor2.getPosition()
            leftVelocity = self.m_LeftMotor1.getVelocity()
            leftVelocity2 = self.m_LeftMotor2.getVelocity()
            rightVelocity = self.m_RightMotor1.getVelocity()
            rightVelocity2 = self.m_RightMotor2.getVelocity()
            # get accelerometer values
            accelX = value[0]
            accelY = value[1]
            accelZ = value[2]
            # create input vector
            inVec = np.array([leftAngle1, leftAngle2, rightAngle1, rightAngle2, leftVelocity, leftVelocity2, rightVelocity,
                                rightVelocity2, accelX, accelY, accelZ])

            # network
            outVals = self.m_Network.forward(torch.from_numpy(inVec).float())

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


    def printDevices(self):
        numDevices = self.getNumberOfDevices()
        print("Number of devices:", numDevices)
        for i in range(numDevices):
            print("\tDevice", i, ":", self.getDeviceByIndex(i).getName())


def startController(timeStep: int):
    # define env variable WEBOTS_ROBOT_NAME as "WalkBot"
    os.environ['WEBOTS_ROBOT_NAME'] = "WalkBot"

    print("Starting controller for WalkBot...")
    controller = WalkingBot()
    controller.initialize(timeStep)
    controller.printDevices()


############################################# new ##############################################
# get weights of a trained!! model using
#weights = m_Network.parameters()
# you can pass a weight matrix by doing:
#m_Network.apply(weights)
################################################################################################