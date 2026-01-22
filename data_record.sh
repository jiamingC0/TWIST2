
source ~/miniconda3/bin/activate twist2
# task_name="0819_shelf"

cd deploy_real

robot_ip="192.168.123.164"
# robot_ip="192.168.110.24"
data_frequency=30


python server_data_record.py --frequency ${data_frequency} --robot_ip ${robot_ip}
        