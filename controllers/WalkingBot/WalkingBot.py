from controller import Robot

# Global defines
TIME_STEP = 32

# Robot initialization
robot = Robot()
leftArm = robot.getDevice('PRM:/r1/c1/c2-Joint2:12')
rightArm = robot.getDevice("RighArmJoint")
print(leftArm)
print(rightArm)

# Sensors
accelerometer = robot.getDevice("accelerometer")
accelerometer.enable(TIME_STEP)

# Main Loop
while robot.step(TIME_STEP) != -1:
    value = accelerometer.getValues()
    print("Sensor value is: ", value)
