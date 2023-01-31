python scripts/txt2img.py --prompt "Teyvat, Name:Layla, Element: Cryo, Weapon:Sword, Region:Sumeru, Model type:Medium Female, Description:a woman in a blue outfit holding a sword" --plms \
    --outdir ./output \
    --ckpt /tmp/2022-11-18T16-38-46_train_colossalai/checkpoints/last.ckpt \
    --config /tmp/2022-11-18T16-38-46_train_colossalai/configs/2022-11-18T16-38-46-project.yaml  \
    --n_samples 4
