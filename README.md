# CRLNet

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
# Datasets
# Training
# Evaluation
To evaluate the performance on different datasets and settings, run:
~~~python
python test.py --n_shot {1/5/10} --backbone {Resnet12/Resnet50/ViT} --pretrained_path path_to_pretrained_model --load path_to_checkpoint --eval_dataset dataset_name  --eval_episodes 600 --gpu gpu_id
~~~
or directly run:
~~~python
./test.sh
~~~

