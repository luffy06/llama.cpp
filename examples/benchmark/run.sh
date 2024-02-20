PROJECT_DIR=/home/wsy/Project/llama.cpp
DATA_PATH=MohamedRashad/ChatGPT-prompts
OUTPUT_PATH=$PROJECT_DIR/data/chatgpt-prompt.txt
MODEL_PATH=$PROJECT_DIR/models/ggml-model-llama-7b-q4_0.gguf
EXEC_PATH=$PROJECT_DIR/build/bin/benchmark-prefetch
NUM_PRE=8

echo `python $PROJECT_DIR/examples/benchmark/gen_chatgpt_prompt.py --data_path $DATA_PATH --output_path $OUTPUT_PATH`

echo `$PROJECT_DIR/build/bin/benchmark-prefetch -m $MODEL_PATH --prompt-file $OUTPUT_PATH --n_predict $NUM_PRE`