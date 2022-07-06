from controller import Robot, Device, Node, Motor, Accelerometer, Supervisor
import torch


class WalkingBot(Robot):
    m_timeStep: int
    
    m_LeftMotor1: Motor
    m_LeftMotor2: Motor
    m_RightMotor1: Motor
    m_RightMotor2: Motor
    
    m_Accelerometer: Accelerometer

    # Network


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