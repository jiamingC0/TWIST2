# TWIST2 Neck

# Step 1: Build Neck

BOM:

Building tutorial: 

# Step 2: Setup Dynamixel (ID the motor)
A video tutorial: [youtube](https://youtu.be/q3mCdYYJPNY?si=tGaAALMcXKZuwG5P)
1. Download Dynamixel Wizard 2.0 from [here](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)
2. `chmod 775 DynamixelWizard2Setup_x64`
3. `./DynamixelWizard2Setup_x64` to start the installer
4. Connect the Dynamixel to the computer
5. Grant your user permission to access serial ports
```bash
sudo chmod 777 /dev/ttyUSB0
```
6. then ID the yaw motor as ID 0 and the pitch motor as ID 1 (also change bard rate to 2Mbps)


# Step 3: Setup onboard neck controller

Refer to our onboard repo for more details.
