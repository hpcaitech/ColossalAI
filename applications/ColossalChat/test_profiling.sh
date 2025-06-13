MAX_NEW_TOKENS=$((4096-512))
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
python rl_example.py --dataset /mnt/nfs/yeanbang/experiments/RLHF/grpo/train-alignment-samll.jsonl --model /home/share/data/model/Qwen2.5-Math-7B/ -t 4 -i 2 -b vllm -a GRPO -ibs 32 -tbs 8 -e 1 -rt boxed -si 100 -s "Please reason step by step, and put your final answer within \\boxed{}." -tmbs 2 -p GRPO-Math-Profile -ei -5 -zero 1 -pp 2 -ptp 2 -mnt $MAX_NEW_TOKENS 2>&1| tee ibs_64_tbs_32_tmbs_2_pp_2_ptp_2_4096_GRPO_profile.txt
python profile_grpo.py --visualization actor_timelines_ibs_64_tbs_32_tmbs_2_pp_2_ptp_2_4096_GRPO_profile.png
