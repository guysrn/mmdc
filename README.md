# Multi-Modal Deep Clustering: Unsupervised Partitioning of Images
[[Paper]](https://arxiv.org/abs/1912.02678)

Pytorch code used for running the main experiments in Multi-Modal Deep Clustering: Unsupervised Partitioning of Images, accepted at ICPR 2020.

### Command Examples
These are the commands used for the main table results:
```python
python3 main.py --dataset mnist --k 10 --arch vgg  --lr 0.05 --wd 0.0005 --epochs 50 --lr_decay_epochs 40 --lr_decay_gamma 0.1 --refine_epoch 20 --crop_size 24 20 16 --input_size 32 --rot_degree 25

python3 main.py --dataset cifar10 --k 10 --arch resnet18 --rotnet --lr 0.05 --wd 0.0005 --epochs 400 --lr_decay_epochs 300 --lr_decay_gamma 0.2 --refine_epoch 350 --crop_size 20 --input_size 32 --flip --color_jitter

python3 main.py --dataset cifar100 --k 20 --arch resnet18 --rotnet --lr 0.05 --wd 0.0001 --epochs 400 --lr_decay_epochs 300 --lr_decay_gamma 0.2 --refine_epoch 350 --crop_size 20 --input_size 32 --flip --color_jitter

python3 main.py --dataset stl10 --k 10 --arch resnet18 --rotnet --lr 0.05 --wd 0.0005 --epochs 400 --lr_decay_epochs 300 --lr_decay_gamma 0.2 --refine_epoch 350 --crop_size 64 --input_size 64 --flip --color_jitter

python3 main.py --dataset imagenet10 --k 10 --arch resnet18 --rotnet --lr 0.05 --wd 0.0005 --epochs 400 --lr_decay_epochs 300 --lr_decay_gamma 0.2 --refine_epoch 350 --crop_size 64 --input_size 64 --flip --color_jitter

python3 main.py --dataset tinyimagenet --k 200 --arch resnet18 --rotnet --lr 0.05 --wd 0.0001 --epochs 400 --lr_decay_epochs 300 --lr_decay_gamma 0.2 --refine_epoch 350 --crop_size 40 --input_size 64 --flip --color_jitter
```
### Checkpoints
| Dataset          | ACC               |Download link |
|------------------|-------------------|--------------|
| MNIST            | 99.1              |[Download](https://drive.google.com/file/d/1uqypeLL1ltmvBswXKaYBDjPdiGGzKce2/view?usp=sharing)  |
| CIFAR10          | 84.3              |[Download](https://drive.google.com/file/d/1-EGtlYj1KWe60JibtsoelDeIqwn10WP3/view?usp=sharing)  |
| CIFAR100         | 43.7              |[Download](https://drive.google.com/file/d/1VkTpH22GHz-Xz0QaI4zXlvaEvj8o5-Bk/view?usp=sharing)  |
| STL-10           | 67.2              |[Download](https://drive.google.com/file/d/15PtB2gfP4OWpuIioMUy1Y_C3UdaOC9kA/view?usp=sharing)  |
| ImageNet-10      | 83.4              |[Download](https://drive.google.com/file/d/1UL5PvYyTmCAws8Nskiqx9W5HQbf6JzhZ/view?usp=sharing)  |
| Tiny-ImageNet    | 15.2              |[Download](https://drive.google.com/file/d/18xUThrSl4_fOT8coExrfgD3YvNSX7VgX/view?usp=sharing)  |

### Requirements
- python >= 3.6
- torch >= 1.4.0
- torchvision 0.5.0
- lap 0.4.0
- numpy, scipy, h5py

### Citation
If you find this repository useful in your research, please cite the following paper:

    @inproceedings{shiran2020mmdc,
        title = {Multi-Modal Deep Clustering: Unsupervised Partitioning of Images},
        author = {Guy Shiran and Daphna Weinshall},
        booktitle = {International Conference on Pattern Recognition (ICPR)},
        year = 2020}
