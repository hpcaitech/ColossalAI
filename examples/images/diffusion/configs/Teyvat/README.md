# Dataset Card for Teyvat BLIP captions
Dataset used to train [Teyvat characters text to image model](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion).

BLIP generated captions for characters images from [genshin-impact fandom wiki](https://genshin-impact.fandom.com/wiki/Character#Playable_Characters)and [biligame wiki for genshin impact](https://wiki.biligame.com/ys/%E8%A7%92%E8%89%B2).

For each row the dataset contains `image` and `text` keys. `image` is a varying size PIL png, and `text` is the accompanying text caption. Only a train split is provided.

The `text` include the tag `Teyvat`, `Name`,`Element`, `Weapon`, `Region`, `Model type`, and `Description`, the `Description` is captioned with the [pre-trained BLIP model](https://github.com/salesforce/BLIP).
## Examples

<img src = "https://huggingface.co/datasets/Fazzie/Teyvat/resolve/main/data/Ganyu_001.png" title = "Ganyu_001.png" style="max-width: 20%;" >

> Teyvat, Name:Ganyu, Element:Cryo, Weapon:Bow, Region:Liyue, Model type:Medium Female, Description:an anime character with blue hair and blue eyes

<img src = "https://huggingface.co/datasets/Fazzie/Teyvat/resolve/main/data/Ganyu_002.png" title = "Ganyu_002.png" style="max-width: 20%;" >

> Teyvat, Name:Ganyu, Element:Cryo, Weapon:Bow, Region:Liyue, Model type:Medium Female, Description:an anime character with blue hair and blue eyes

<img src = "https://huggingface.co/datasets/Fazzie/Teyvat/resolve/main/data/Keqing_003.png" title = "Keqing_003.png" style="max-width: 20%;" >

> Teyvat, Name:Keqing, Element:Electro, Weapon:Sword, Region:Liyue, Model type:Medium Female, Description:a anime girl with long white hair and blue eyes

<img src = "https://huggingface.co/datasets/Fazzie/Teyvat/resolve/main/data/Keqing_004.png" title = "Keqing_004.png" style="max-width: 20%;" >

> Teyvat, Name:Keqing, Element:Electro, Weapon:Sword, Region:Liyue, Model type:Medium Female, Description:an anime character wearing a purple dress and cat ears
