import os
num_rays = 24
devices = ["cpu_xeon", "gpu_titanxp", "cpu_raspberrypi"]
tasks = ["wmt14.en-de", "wmt14.en-fr"]
data = ["data/binary/wmt16_en_de/", "data/binary/wmt14_en_fr/"]
import subprocess
for task in tasks:
  for device in devices:
    for i in range(num_rays):
        if device == "cpu_xeon":
           arg1 = "config_xeon.yml"
        elif device == "gpu_titanxp":
            arg1 = "config_titanxp.yml"
        else:
            arg1 = "config_raspberrypi.yml"
        arg2 = "normal"
        arg3 = 0
        arg4 = "test"
        arg5 = i
        arg6 = device
        arg7 = task
        if "fr" in task:
            arg8 = data[1]
            model_path = "/path/to/checkpoint.pt"
        else:
            arg8 = data[0]
            model_path = "/path/to/checkpoint.pt"
        arg9 = model_path
        subprocess.check_call("bash test.sh %s %s %s %s %s %s %s %s %s" % (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9), shell=True)