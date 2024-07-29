exec_path='/home/wsy/Project/LLMInference/prefetch/llama.cpp'
gguf_path='/home/wsy/Project/LLMInference/prefetch/llama.cpp/models'
model_path='/mnt/wsy/models'
model_list=('llama-7b')

for model_name in ${model_list[@]}; do
    python convert-hf-to-gguf.py $model_path/$model_name --outfile $model_path/$model_name/ggml-model-$model_name-f16-unsorted.gguf --outtype f16
    $exec_path/llama-quantize $model_path/$model_name/ggml-model-$model_name-f16-unsorted.gguf $gguf_path/ggml-model-$model_name-q4_0-unsorted.gguf q4_0

    python convert-hf-to-gguf.py $model_path/$model_name --outfile $model_path/$model_name/ggml-model-$model_name-f16-512.gguf --outtype f16 --align 512 --sort
    $exec_path/llama-quantize $model_path/$model_name/ggml-model-$model_name-f16-512.gguf $gguf_path/ggml-model-$model_name-q4_0-512.gguf q4_0
done