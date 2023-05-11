#!/bin/bash
#SBATCH --job-name=llama-7b
#SBATCH --account=project_462000229
#SBATCH --output=/pfs/lustref1/flash/project_462000229/logs/%j.log
#SBATCH --nodes=1               # Total number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --mem=0			# All memory on the node
#SBATCH --partition=small-g

module load LUMI/22.08 partition/G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
export NCCL_DEBUG=INFO
export PYTHONPATH=.:${PYTHONPATH}
export WANDB_PROJECT=alpacaX-gpu
export ROCM_PATH=/opt/rocm
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64
# variables for script
export MODEL_SIZE="7B"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_DISABLED=true
export HF_HOME=${FLASH_DIR}/open-instruct/hf_cache

srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --distribution=block:block \
  --kill-on-bad-exit \
  run_with_environment.sh \
    singularity exec \
    -B"$PROJECT_DIR:$PROJECT_DIR" \
    -B"$SCRATCH_DIR:$SCRATCH_DIR" \
    -B"$FLASH_DIR:$FLASH_DIR" \
    -B /opt/cray:/opt/cray \
    -B /usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjson-c.so.3:/usr/lib64/libjson-c.so.3 \
    $PROJECT_DIR/containers/open-instruct/lumi-open-instruct_latest.sif \
    torchrun --nproc_per_node=8 --nnode=1 --node_rank=0 --master_addr=$(scontrol show hostnames | head -n 1) --master_port=39591 finetune.py \
        --model_name_or_path gpt2-xl \
        --tokenizer_name gpt2-xl  \
        --train_file dummy_data.jsonl \
        --bf16 \
        --gradient_checkpointing \
        --max_seq_length 2048 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 2 \
        --do_train \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "linear" \
        --evaluation_strategy "no" \
        --logging_steps 1 \
        --save_steps 400 \
        --save_total_limit 1 \
        --output_dir ${SCRATCH_DIR}/open-instruct-data/test_output \
        --overwrite_output_dir \
	    --deepspeed "stage3_no_offloading.conf"
