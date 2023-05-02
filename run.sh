MASTER_HOST=$SM_MASTER
MASTER_ADDR=$SM_MASTER_ADDR
MASTER_PORT="23456"
NNODES="$NODE_NUMBER"
NODE_RANK="$NODE_INDEX"

MODEL_S3_BUCKET=s3://sagemaker-us-east-1-107457652907/pretrained-models/llama-7b

wandb login d9bc4cccef46949e9fdffb3df442996d803d43d2

chmod +x ./s5cmd

# ======================================

#./s5cmd sync $MODEL_S3_BUCKET/* /tmp/llama-7b/

#OUTPUT_DIR=/tmp/llama.7b.zh_instruct.10M.v1.0.seq1024.w8.adamw.NA100.0421.ds
#AWS_OUTPUT_BUCKET=s3://sagemaker-us-east-1-107457652907/experiments/llama.7b.zh_instruct.10M.v1.0.seq1024.w8.adamw.NA100.0421.ds

#./s5cmd sync $AWS_OUTPUT_BUCKET/checkpoint-1750/* /tmp/checkpoints/checkpoint-1750/

#python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mul_aws.py output_dir=$OUTPUT_DIR aws_output_bucket=$AWS_OUTPUT_BUCKET resume=/tmp/checkpoints/checkpoint-1750 -cp conf/llama/zh/ -cn llama_7b_zh_instruct_v1_0_ds



# ============ COIG SFT ============

./s5cmd sync s3://sagemaker-us-east-1-107457652907/experiments/llama.7b.zh_instruct.10M.v1.0.seq1024.w8.adamw.NA100.0421.ds/checkpoint-1750/* /tmp/zh_instruct_v1_0/checkpoint-1750/


AWS_OUTPUT_BUCKET=s3://sagemaker-us-east-1-107457652907/experiments/llama.7b.zh_instruct.10M.coig.sft.v1.0.seq2048.w8.adamw.NA100.0428.ds


python -m torch.distributed.run --nproc_per_node 8 --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT trainer_base_ds_mul_aws.py aws_output_bucket=$AWS_OUTPUT_BUCKET -cp conf/llama/zh/ -cn llama_7b_zh_instruct_coig_sft_v1_0_ds


# ./s5cmd sync $OUTPUT_DIR s3://sagemaker-us-east-1-107457652907/experiments/
./s5cmd sync /tmp/log_dir/* s3://sagemaker-us-east-1-107457652907/experiments/log_dir/
