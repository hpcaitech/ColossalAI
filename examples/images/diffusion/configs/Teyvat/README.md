# Dataset Card for Teyvat BLIP captions
Dataset used to train [Teyvat characters text to image model](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion).

BLIP generated captions for characters images from [genshin-impact fandom wiki](https://genshin-impact.fandom.com/wiki/Character#Playable_Characters)and [biligame wiki for genshin impact](https://wiki.biligame.com/ys/%E8%A7%92%E8%89%B2).

For each row the dataset contains `image` and `text` keys. `image` is a varying size PIL png, and `text` is the accompanying text caption. Only a train split is provided.

The `text` include the tag `Teyvat`, `Name`,`Element`, `Weapon`, `Region`, `Model type`, and `Description`, the `Description` is captioned with the [pre-trained BLIP model](https://github.com/salesforce/BLIP).
