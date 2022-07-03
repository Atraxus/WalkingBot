#include <webots/robot.h>
#include <stdio.h>

#define TIMESTEP 32

void listDevices() {
    int numDevices = wb_robot_get_number_of_devices();
    for (int i = 0; i < numDevices; i++) {
        WbDeviceTag tag = wb_robot_get_device_by_index(i);
        const char* name = wb_device_get_name(tag);
        printf("Name of device [%d]: %s", i, name);
    }
}

int main() {
    wb_robot_init();

    listDevices();

    while (wb_robot_step(TIMESTEP) != -1)
        printf("Hello World!\n");

    wb_robot_cleanup();
    return 0;
}