from controller import Robot, Device, Node, Motor, Accelerometer, Supervisor


class WalkingBot(Robot):
    """Make the NAO robot run as fast as possible."""
    m_timeStep: int
    m_LeftMotor1: Motor
    m_LeftMotor2: Motor
    m_RightMotor1: Motor
    m_RightMotor2: Motor

    def initialize(self):
        """Get device pointers, enable sensors and set robot initial pose."""
        # This is the time step (ms) used in the motion file.
        self.timeStep = 32
        # Get pointers to the shoulder motors.
        self.m_LeftMotor1 = self.getMotor('LeftLegMotor1')
        self.m_LeftMotor2 = self.getMotor('LeftLegMotor2')
        self.m_RightMotor1 = self.getMotor('RightLegMotor1')
        self.m_RightMotor2 = self.getMotor('RightLegMotor2')


    def run(self):
        while True:
            continue


controller = WalkingBot()
controller.initialize()
controller.run()



# For testing purposes
# numDevices = robot.getNumberOfDevices()
# print("Number of devices:", numDevices)
# for i in range(numDevices):
#     print("Device", i, ":", robot.getDeviceByIndex(i).getName())


# Motors


# Sensors
# accelerometer = robot.getDevice("accelerometer")
# accelerometer.enable(TIME_STEP)

# Main Loop to get the robot to walk
# while robot.step(TIME_STEP) != -1:
#     i = 0
    # Read in sensordata 