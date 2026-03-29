export CLIENT_ID="/home/haibo/haibo_workspace/weights/Qwen3-VL-32B-Instruct"
export SAM_CKPT="/home/haibo/haibo_workspace/weights/sam3"
export DATA_DIR="/home/haibo/haibo_workspace/data"
export DATASET_NAME="SCANREF" # "SCANREF" OR "NR3D"
export NUM_GPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py