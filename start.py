import os
import json
import socket

if __name__ == "__main__":
   
    hosts = json.loads(os.environ['SM_HOSTS'])
    current_host = os.environ['SM_CURRENT_HOST']
    host_rank = int(hosts.index(current_host))
    
    #Parse the IP address of the master node in the multiple nodes cluster of SageMaker training.
    master = json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname']
    master_addr = socket.gethostbyname(master)
    
    os.environ['NODE_INDEX'] = str(host_rank)
    os.environ['SM_MASTER'] = str(master)
    os.environ['SM_MASTER_ADDR'] = str(master_addr)
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    
    #invoke the torch launcher shell script.
    #Note: we will use the pytorch launcher to launch deepspeed for multi-nodes training.
    #Note: we will use the s5cmd to speed up the uploading model assets to S3.
    os.system("chmod +x ./run.sh")
    # os.system("chmod +x ./T5_configz_and_code/scripts/s5cmd")
    os.system("/bin/bash -c ./run.sh")