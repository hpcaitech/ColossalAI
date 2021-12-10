# Train ViLT on COCO dataset with Colossal-AI

Colossal-AI implementation for the ICML 2021 (long talk) paper: "[ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)"

---
<p align="center">
  <img align="middle" src="./assets/vilt.png" alt="The main figure"/>
</p>



## Prepare Environment
```bash
pip install -r requirements.txt
```

## Prepare Dataset
In this example we use the COCO Captions (COCO) dataset.

We do not distribute datasets because of the license issue.
Please download the datasets by yourself.
We use `pyarrow` to serialize the datasets, conversion scripts are located in `vilt/utils/write_*.py`.
Please organize the datasets as follows and run `make_arrow` functions to convert the dataset to pyarrow binary file.


https://cocodataset.org/#download

Download [2014 train images](http://images.cocodataset.org/zips/train2014.zip), [2014 val images](http://images.cocodataset.org/zips/val2014.zip) and [karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)

    root
    ├── train2014            
    │   ├── COCO_train2014_000000000009.jpg                
    |   └── ...
    ├── val2014              
    |   ├── COCO_val2014_000000000042.jpg
    |   └── ...          
    └── karpathy
        └── dataset_coco.json

```python
from vilt.utils.write_coco_karpathy import make_arrow
make_arrow(root, arrows_root)
```

## Train image text matching (ITM) and masked language (MLM) Models

```bash

mpirun -np <NUM_GPUS> python runcai.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_mlm_itm whole_word_masking=True step200k per_gpu_batchsize=<BS_FITS_YOUR_GPU>

ex)
mpirun -np 4 python runcai.py with data_root=./vilt_data/arrow_coco num_gpus=1 num_nodes=1 task_mlm_itm_s whole_word_masking=True step200k per_gpu_batchsize=96
```


## Citation
If you use any part of this code and pretrained weights for your own purpose, please cite the original [paper](https://arxiv.org/abs/2102.03334).
```
@InProceedings{pmlr-v139-kim21k,
  title = 	 {ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision},
  author =       {Kim, Wonjae and Son, Bokyung and Kim, Ildoo},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {5583--5594},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/kim21k/kim21k.pdf},
  url = 	 {http://proceedings.mlr.press/v139/kim21k.html},
  abstract = 	 {Vision-and-Language Pre-training (VLP) has improved performance on various joint vision-and-language downstream tasks. Current approaches to VLP heavily rely on image feature extraction processes, most of which involve region supervision (e.g., object detection) and the convolutional architecture (e.g., ResNet). Although disregarded in the literature, we find it problematic in terms of both (1) efficiency/speed, that simply extracting input features requires much more computation than the multimodal interaction steps; and (2) expressive power, as it is upper bounded to the expressive power of the visual embedder and its predefined visual vocabulary. In this paper, we present a minimal VLP model, Vision-and-Language Transformer (ViLT), monolithic in the sense that the processing of visual inputs is drastically simplified to just the same convolution-free manner that we process textual inputs. We show that ViLT is up to tens of times faster than previous VLP models, yet with competitive or better downstream task performance. Our code and pre-trained weights are available at https://github.com/dandelin/vilt.}
}
```


