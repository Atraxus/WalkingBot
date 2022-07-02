from controller import Robot, DistanceSensor

TIME_STEP = 32

robot = Robot()

accelerometer = robot.getDevice("accelerometer")
accelerometer.enable(TIME_STEP)

while robot.step(TIME_STEP) != -1:
    value = accelerometer.getValues()
    print("Sensor value is: ", value)