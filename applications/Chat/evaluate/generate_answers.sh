device_number=number of your devices
model_name="name of your model"
model_path="path to your model"
dataset="path to the question dataset"
answer_path="path to save the model answers"

torchrun --standalone --nproc_per_node=$device_number generate_answers.py \
    --model 'llama' \
    --strategy ddp \
    --model_path $model_path \
    --model_name $model_name \
    --dataset $dataset \
    --batch_size 8 \
    --max_datasets_size 80 \
    --answer_path $answer_path \
    --max_length 512

python merge.py \
    --model_name $model_name \
    --shards $device_number \
    --answer_path $answer_path \

for (( i=0; i<device_number; i++ )) do
    rm -rf "${answer_path}/${model_name}_answers_rank${i}.json"
done
