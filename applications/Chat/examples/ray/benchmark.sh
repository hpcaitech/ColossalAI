
PROMPT_PATH=/home/lccsr/data3/awesome-chatgpt-prompts/prompts.csv

num_trainers=4
num_makers=4

#  "facebook/opt-2.7b" 
for pretrain in "facebook/opt-1.3b" "facebook/opt-6.7b" "facebook/opt-13b"
do
    
    for experience_batch_size in 16 32 64
    do
        for train_batch_size in 16 32 64
        do
            for update_steps in 8 32 128
            do
                # set a big enough experience_steps for twice maker-update
                experience_steps=$((2*num_trainers*train_batch_size*update_steps/num_makers/experience_batch_size))

                config_string=${num_trainers}_${num_makers}_pretrain_${pretrain##*/}_experience_batch_size_${experience_batch_size}_train_batch_size_${train_batch_size}_update_steps_${update_steps}_experience_steps_${experience_steps}
                echo running: ${config_string}

                nohup python mmmt_prompt.py \
                    --prompt_path $PROMPT_PATH \
                    --trainer_strategy colossalai_gemini --maker_strategy naive \
                    --model 'opt' \
                    --pretrain $pretrain \
                    --critic_pretrain "facebook/opt-350m" \
                    --num_trainers $num_trainers \
                    --num_makers $num_makers \
                    --experience_steps $experience_steps \
                    --experience_batch_size $experience_batch_size \
                    --update_steps $update_steps \
                    --train_batch_size $train_batch_size \
                    --debug > logs/output_${config_string}.txt 2>&1
            done
        done
    done
done