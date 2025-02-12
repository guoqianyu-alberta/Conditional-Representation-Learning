# CRLNet

PyTorch implementation for "**Enhancing Environmental Robustness in Few-shot Learning via Conditional Representation Learning**"

![image](https://github.com/user-attachments/assets/068dec8e-4348-49dc-909e-2ba82a5f9455)


# Requirements
fvcore==0.1.5
 
numpy==1.24.1
 
Pillow==10.4.0
 
pynvml==11.5.0
 
scikit_learn==1.3.1
 
timm==0.9.5
 
torch==2.0.1+cu118
 
torchvision==0.15.2+cu118
 
tqdm==4.66.4
# Pretrained Weights
Pretrained backbone and CRLNet checkpoints are available. Put backbone in  `/pretrained` and put checkpoints in `/checkpoints` . Accuracy of 5-way 1-shot, 5-way 5-shot and 5-way 10-shot shown in the table is evaluated separately on each dataset. 

Link to backbone and checkpoints: [password: 5wbz ](https://pan.baidu.com/s/1Q3ICj3zaJIldKg6DQfh68g)

| dataset | 1-shot | 5-shot | 10-shot |
| ------- | ------ | ------ | ------- |
| Animal | 46.71±0.84| 55.47±0.82|60.14±0.79 |
| Insect |54.08±0.81 | 64.89±0.76 | 69.38±0.72 |
| Mineral| 40.05±0.63 |57.37±0.62 | 65.61±0.58 |
| Oracle | 29.34±0.60 | 38.52±0.66 | 42.68±0.72 |
| Fungus | 35.50±0.74 | 51.52±0.71 | 58.84±0.72 |
| Plant Virus| 73.31±0.94 | 82.77±0.81 | 86.04±0.77 |


# Datasets
We provide evaluation datasets in the link below. Put data in `/data/datasets folder`.

Link to dataset: [pwd: m49f ](https://pan.baidu.com/s/1_dEwyfysZTGj98bpLfOJpA?pwd=m49f)



# Training
Note that for training, two pickle files should be generated: 

(1) pickle file contains paths for all the images in training dataset 

(2) pickle file contains paths organized by categories



We also provide the training code for miniImageNet and ImageNet, run:
~~~python
torchrun --nnodes=1 --nproc_per_node={gpu_num} --master_port=20000 train.py  --image_size 224 --train_bsz 80 --epochs 100 --backbone Resnet12 --train_dataset {ImageNet/miniImageNet} --lr 1e-3  
~~~
or directly run:
~~~python
./train.sh
~~~

# Evaluation
To evaluate the performance on different datasets and settings, run:
~~~python
python test.py --n_shot {1/5/10} --backbone Resnet12 --pretrained_path path_to_pretrained_model --load path_to_checkpoint --eval_dataset dataset_name  --eval_episodes 600 --gpu gpu_id
~~~
or directly run:
~~~python
./test.sh
~~~

# LICENSE
This repository is released under the MIT license as found in the [LICENSE](LICENSE) file.

