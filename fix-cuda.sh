# Backup the old ptxas
cp /home/binadiegha/anaconda3/envs/dreamer_cuda12/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas /home/binadiegha/anaconda3/envs/dreamer_cuda12/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas.old

# Replace with the newer one
cp /usr/local/cuda-13.0/bin/ptxas /home/binadiegha/anaconda3/envs/dreamer_cuda12/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas

