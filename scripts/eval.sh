export CLIENT_ID="your_path_to/Qwen3-VL-32B-Instruct"
export SAM_CKPT="your_path_to/sam3"
export DATA_DIR="your_path_to/data"
export DATASET_NAME="SCANREF" # "SCANREF" OR "NR3D"
export NUM_GPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py