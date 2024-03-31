import datetime
import time

stamp = int(time.time())
cur_time = datetime.datetime.fromtimestamp(stamp)

file_name = cur_time.strftime("%Y%m%d_%H%M%S")

print(cur_time)
print(file_name)

import torch
print("torch_version:",torch.__version__)
print("cuda_version:",torch.version.cuda)
print("cudnn_version:",torch.backends.cudnn.version())
print("----------------------------------")
flag = torch.cuda.is_available()
print(flag)