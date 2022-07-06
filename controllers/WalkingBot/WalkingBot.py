from controller import Robot, Device, Node, Motor, Accelerometer, Supervisor
from dataclasses import dataclass
import torch


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
        self.inoutTransform = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.inoutTransform(x)



class WalkingBot(Robot):
    m_timeStep: int
    
    m_LeftMotor1: Motor
    m_LeftMotor2: Motor
    m_RightMotor1: Motor
    m_RightMotor2: Motor
    
    m_Accelerometer: Accelerometer

    # Network
    m_network: SimpleNetwork

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

    def printDevices(self):
        numDevices = self.getNumberOfDevices()
        print("Number of devices:", numDevices)
        for i in range(numDevices):
            print("Device", i, ":", self.getDeviceByIndex(i).getName())

# Torch
x = torch.rand(5, 3)
print(x)
# For testing purposes
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