from controller import Robot, Device, Node, Motor, Accelerometer

# Global defines
TIME_STEP = 32
yTarget = 1

# Robot initialization
robot = Robot()

# For testing purposes
numDevices = robot.getNumberOfDevices()
print("Number of devices:", numDevices)
for i in range(numDevices):
    print("Device", i, ":", robot.getDeviceByIndex(i).getName())


# Motors
leftMotor1 = robot.getDevice("LeftLegtMotor1")
leftMotor2 = robot.getDevice("LeftLegtMotor2")
rightMotor1 = robot.getDevice("RightLegtMotor1")
rightMotor2 = robot.getDevice("RightLegtMotor2")

# Set PID gains
leftMotor1.setControlPID(100, 10, 0.1)
leftMotor2.setControlPID(100, 10, 0.1)
rightMotor1.setControlPID(100, 10, 0.1)
rightMotor2.setControlPID(100, 10, 0.1)

# Sensors
accelerometer = robot.getDevice("accelerometer")
accelerometer.enable(TIME_STEP)

# Main Loop to get the robot to walk
while robot.step(TIME_STEP) != -1:
    # Get robot y position
    # y = accelerometer.getValues()[1]
    # Calculate error
    # error = abs(yTarget - y)
    # Calculate motor speeds
    # leftMotor1.setVelocity(error * 0.5)
    # leftMotor2.setVelocity(error * 0.5)
    # rightMotor1.setVelocity(error * 0.5)
    # rightMotor2.setVelocity(error * 0.5)
    i = 0



