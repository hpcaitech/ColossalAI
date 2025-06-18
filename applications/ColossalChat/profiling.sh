# profile under different setups
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# zero2 ibs32 tbs32
# python rl_example.py --dataset /mnt/nfs/yeanbang/experiments/RLHF/grpo/train-alignment-samll.jsonl --model /home/share/data/model/Qwen2.5-Math-7B/ -t 4 -i 4 -b vllm -a GRPO -ibs 8 -tbs 8 -e 1 -rt boxed -si 100 -s "Please reason step by step, and put your final answer within \\boxed{}." -tmbs 2 -p GRPO-Math-Profile -ei -5 -zero 2 2>&1| tee ibs_32_tbs_32_tmbs_2_zero_2_GRPO_profile.txt
# python profile_grpo.py --visualization actor_timelines_ibs_32_tbs_32_tmbs_2_zero_2_GRPO_profile.png

# # zero2 ibs64 tbs32
# python rl_example.py --dataset /mnt/nfs/yeanbang/experiments/RLHF/grpo/train-alignment-samll.jsonl --model /home/share/data/model/Qwen2.5-Math-7B/ -t 4 -i 4 -b vllm -a GRPO -ibs 16 -tbs 8 -e 1 -rt boxed -si 100 -s "Please reason step by step, and put your final answer within \\boxed{}." -tmbs 2 -p GRPO-Math-Profile -ei -5 -zero 2 2>&1| tee ibs_64_tbs_32_tmbs_2_zero_2_GRPO_profile.txt
# python profile_grpo.py --visualization actor_timelines_ibs_64_tbs_32_tmbs_2_zero_2_GRPO_profile.png

# # zero2 ibs96 tbs32
# python rl_example.py --dataset /mnt/nfs/yeanbang/experiments/RLHF/grpo/train-alignment-samll.jsonl --model /home/share/data/model/Qwen2.5-Math-7B/ -t 4 -i 4 -b vllm -a GRPO -ibs 24 -tbs 8 -e 1 -rt boxed -si 100 -s "Please reason step by step, and put your final answer within \\boxed{}." -tmbs 2 -p GRPO-Math-Profile -ei -5 -zero 2 2>&1| tee ibs_96_tbs_32_tmbs_2_zero_2_GRPO_profile.txt
# python profile_grpo.py --visualization actor_timelines_ibs_96_tbs_32_tmbs_2_zero_2_GRPO_profile.png

# 4K
# MAX_NEW_TOKENS=$((4096-512))
# python rl_example.py --dataset /mnt/nfs/yeanbang/experiments/RLHF/grpo/train-alignment-samll.jsonl --model /home/share/data/model/Qwen2.5-Math-7B/ -t 4 -i 2 -b vllm -a GRPO -ibs 32 -tbs 8 -e 1 -rt boxed -si 100 -s "Please reason step by step, and put your final answer within \\boxed{}." -tmbs 2 -p GRPO-Math-Profile -ei -5 -zero 1 -pp 2 -ptp 2 -mpt $MAX_NEW_TOKENS 2>&1| tee ibs_64_tbs_32_tmbs_2_pp_2_ptp_2_4096_GRPO_profile.txt
# python profile_grpo.py --visualization actor_timelines_ibs_64_tbs_32_tmbs_2_pp_2_ptp_2_16384_GRPO_profile.png

# # 32K
# MAX_NEW_TOKENS=$((32768-512))
# python rl_example.py --dataset /mnt/nfs/yeanbang/experiments/RLHF/grpo/train-alignment-samll.jsonl --model /home/share/data/model/Qwen2.5-Math-7B/ -t 4 -i 2 -b vllm -a GRPO-DUMMY-TEST -ibs 32 -tbs 32 -e 1 -rt boxed -si 100 -s "Please reason step by step, and put your final answer within \\boxed{}." -tmbs 4 -imbs 1 -p GRPO-Math-Profile -ei -5 -zero 1 -pp 4 -ptp 2 -mnt $MAX_NEW_TOKENS 2>&1| tee ibs_64_tbs_32_tmbs_2_pp_2_ptp_2_32768_GRPO_profile.txt
# python profile_grpo.py --visualization actor_timelines_ibs_64_tbs_32_tmbs_2_pp_2_ptp_2_32768_GRPO_profile.png

# 16K
# MAX_NEW_TOKENS=$((16384-512))
# python rl_example.py --dataset /mnt/nfs/yeanbang/experiments/RLHF/grpo/train-alignment-samll.jsonl --model /home/share/data/model/Qwen2.5-Math-7B/ -t 4 -i 2 -b vllm -a GRPO-DUMMY-TEST -ibs 32 -tbs 16 -e 1 -rt boxed -si 100 -s "Please reason step by step, and put your final answer within \\boxed{}." -tmbs 2 -imbs 8 -p GRPO-Math-Profile -ei -5 -zero 1 -pp 2 -ptp 2 -mnt $MAX_NEW_TOKENS 2>&1| tee ibs_64_tbs_32_tmbs_2_pp_s_2_ptp_2_16384_GRPO_profile.txt
# python profile_grpo.py --visualization actor_timelines_ibs_64_tbs_32_tmb2_pp_2_ptp_2_16384_GRPO_profile.png

# 8K
# MAX_NEW_TOKENS=$((8192-512))
# python rl_example.py --dataset /mnt/nfs/yeanbang/experiments/RLHF/grpo/train-alignment-samll.jsonl --model /home/share/data/model/Qwen2.5-Math-7B/ -t 4 -i 4 -b vllm -a GRPO-DUMMY-TEST -ibs 16 -tbs 16 -e 1 -rt boxed -si 100 -s "Please reason step by step, and put your final answer within \\boxed{}." -tmbs 4 -p GRPO-Math-Profile -ei -5 -zero 1 -pp 2 -mnt $MAX_NEW_TOKENS 2>&1| tee ibs_64_tbs_32_tmbs_2_pp_2_ptp_2_8192_GRPO_profile.txt
# python profile_grpo.py --visualization actor_timelines_ibs_64_tbs_32_tmbs_2_pp_2_ptp_2_8192_GRPO_profile.png


# # 32K
MAX_NEW_TOKENS=$((32768-512))
python rl_example.py --dataset /mnt/nfs/yeanbang/experiments/RLHF/grpo/train-alignment-samll.jsonl --model /home/share/data/model/Qwen2.5-Math-7B/ -t 4 -i 2 -b vllm -a GRPO-DUMMY-TEST -ibs 32 -tbs 32 -tMbs 2 -e 1 -rt boxed -si 100 -s "Please reason step by step, and put your final answer within \\boxed{}." -tmbs 2 -imbs 1 -p GRPO-Math-Profile -ei -5 -zero 1 -pp 2 -tp 2 -ptp 2 -mnt $MAX_NEW_TOKENS 2>&1| tee ibs_64_tbs_32_tmbs_2_pp_2_tp_2_ptp_2_32768_GRPO_profile.txt
python profile_grpo.py --visualization actor_timelines_ibs_64_tbs_32_tmbs_2_pp_2_tp_2_ptp_2_32768_GRPO_profile.png
