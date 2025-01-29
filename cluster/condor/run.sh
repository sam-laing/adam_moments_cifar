# Define variables
# NOTE: executable should be a full path
LOGS_DIR=/home/slaing/minimalLM/llm_pretrain/logs
executable=/home/slaing/minimalLM/cluster/condor/single_gpu/run.sh

# Job specific vars
config=/home/slaing/minimalLM/config/sweep_config.yaml
n_jobs=16

# Args
arguments = $(config) $(Process)

# Logs
error = $(LOGS_DIR)/err/job.$(Cluster).$(Process).err
output = $(LOGS_DIR)/out/job.$(Cluster).$(Process).out
log = $(LOGS_DIR)/log/job.$(Cluster).$(Process).log

# Specs
request_memory = 250000
request_cpus = 12
request_gpus = 1
requirements = (TARGET.CUDADeviceName == "NVIDIA A100-SXM4-80GB")

queue $(n_jobs)
