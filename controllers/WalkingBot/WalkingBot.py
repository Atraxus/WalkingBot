from controller import Robot

# Global defines
TIME_STEP = 32

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
leftMotor1.setVeloctity(0.0)
leftMotor2.setVeloctity(0.0)
rightMotor1.setVeloctity(0.0)
rightMotor2.setVeloctity(0.0)


# Sensors
accelerometer = robot.getDevice("accelerometer")
accelerometer.enable(TIME_STEP)

# Main Loop
while robot.step(TIME_STEP) != -1:
    value = accelerometer.getValues()
    print("Sensor value is: ", value)
