# LSST

This is the official PyTorch implementation of our paper:

> [**Simple and Efficient: A Semisupervised Learning Framework for Remote Sensing Image Semantic Segmentation**](https://ieeexplore.ieee.org/abstract/document/9943552)      
> Xiaoqiang Lu, Licheng Jiao, Fang Liu, Shuyuan Yang, Xu Liu, Zhixi Feng, Lingling Li, Puhua Chen    
> *Accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS) 2022*

# Getting Started
## Install
```
conda create -n lsst python=3.7
pip install -r requirements.txt
```

## Data Preparation

### Pre-trained Model
```
mkdir pretrained
cd pretrained
wget https://download.pytorch.org/models/resnet101-63fe2227.pth
mv resnet101-63fe2227.pth resnet101.pth
cd ..
```
or download via the following link
[ResNet-101](https://download.pytorch.org/models/resnet101-63fe2227.pth)

### Dataset
We have processed the original dataset as mentioned in the paper. You can access the processed dataset directly via the following link.

[GID-15](https://drive.google.com/file/d/1CITO8Bxf8eG-6le6mZDshDgTyi9BO2ot/view?usp=sharing) | [iSAID](https://drive.google.com/file/d/1yBlHpxuWs_X_02O5xAxTpo1tDWcGj6-x/view?usp=sharing) | [DFC22](https://drive.google.com/file/d/1FZoZiWW_19TWdWKUz1-Vn-IlTs0QWtfg/view?usp=sharing) | [MER](https://drive.google.com/file/d/1KN0PSoWlC4BPv5QAZdJKrhpvu9F9vdlm/view?usp=sharing) | [MSL](https://drive.google.com/file/d/1XSvmXB5rsZUi98-rWWwQtONfop7HJw4M/view?usp=sharing) | [Vaihingen](https://drive.google.com/file/d/11aOEO3-Ov3Wcg8o2nBA5GsGBqqRHbjRh/view?usp=sharing)

### File Organization

```
├── ./pretrained
    └── resnet101.pth
    
├── [Your Dataset Path]
    ├── images
    └── labels
```

# Results

<div style="text-align: center;">
<table>
    <tr>
        <td>Dataset</td> 
        <td>Setting</td>
        <td>Method</td> 
        <td>mIoU</td>
        <td>Model</td> 
   </tr>
    <tr>
        <td rowspan="4">GID-15</td>    
        <td rowspan="2">1-8</td>
        <td >baseline</td>
        <td >61.86</td>
        <td >-</td>
    </tr>
    <tr>
        <td >ours</td>
        <td >66.38</td>
        <td > <a href="https://drive.google.com/file/d/1r5c_1jTeO7rXXVjB2sXcQePIGTbsc8f_/view?usp=sharing" target="_blank">Google Drive</a></td>
    </tr>
    <tr>
        <td rowspan="2">1-4</td>
        <td >baseline</td>
        <td >67.90</td>
        <td >-</td>
    </tr>
    <tr>
        <td >ours</td>
        <td >71.28</td>
        <td > <a href="https://drive.google.com/file/d/1Bo8FVBeDBbFKM13GLHnJWVqFMsajz7l-/view?usp=sharing" target="_blank">Google Drive</a></td>
    </tr>
    <tr>
        <td rowspan="4">iSAID</td>    
        <td rowspan="2">100</td>
        <td >baseline</td>
        <td >39.91</td>
        <td >-</td>
    </tr>
    <tr>
        <td >ours</td>
        <td >46.94</td>
        <td > <a href="https://drive.google.com/file/d/1vtBomAk-A2Ssr1F5Sbbir6crhqANd7DV/view?usp=sharing" target="_blank">Google Drive</a></td>
    </tr>
    <tr>
        <td rowspan="2">300</td>
        <td >baseline</td>
        <td >60.47</td>
        <td >-</td>
    </tr>
    <tr>
        <td >ours</td>
        <td >63.40</td>
        <td > <a href="https://drive.google.com/file/d/1qEO7MvEGR0XxPYHCwynWb1vaUMfEkIv4/view?usp=sharing" target="_blank">Google Drive</a></td>
    </tr>
    <tr>
        <td rowspan="4">DFC22</td>    
        <td rowspan="2">1-8</td>
        <td >baseline</td>
        <td >26.97</td>
        <td >-</td>
    </tr>
    <tr>
        <td >ours</td>
        <td >30.94</td>
        <td > <a href="https://drive.google.com/file/d/1bYvGUvs4hhJBUz5spIfbzxgXgDA1spII/view?usp=sharing" target="_blank">Google Drive</a></td>
    </tr>
    <tr>
        <td rowspan="2">1-4</td>
        <td >baseline</td>
        <td >32.67</td>
        <td >-</td>
    </tr>
    <tr>
        <td >ours</td>
        <td >36.40</td>
        <td > <a href="https://drive.google.com/file/d/1p5kwkX6JNUp55TR-Xa49WmuxIoU9RjgL/view?usp=sharing" target="_blank">Google Drive</a></td>
    </tr>
    <tr>
        <td rowspan="4">MER</td>    
        <td rowspan="2">1-8</td>
        <td >baseline</td>
        <td >43.63</td>
        <td >-</td>
    </tr>
    <tr>
        <td >ours</td>
        <td >49.68</td>
        <td > <a href="https://drive.google.com/file/d/1onc0yAHrXD0AGQ_UxJt1emFTw-SazPip/view?usp=sharing" target="_blank">Google Drive</a></td>
    </tr>
    <tr>
        <td rowspan="2">1-4</td>
        <td >baseline</td>
        <td >48.19</td>
        <td >-</td>
    </tr>
    <tr>
        <td >ours</td>
        <td >51.31</td>
        <td > <a href="https://drive.google.com/file/d/1YPKR2RCvCrUr0Hg8y6_GKAzmsMGfVy7k/view?usp=sharing" target="_blank">Google Drive</a></td>
    </tr>
    <tr>
        <td rowspan="4">MSL</td>    
        <td rowspan="2">1-8</td>
        <td >baseline</td>
        <td >50.17</td>
        <td >-</td>
    </tr>
    <tr>
        <td >ours</td>
        <td >54.72</td>
        <td > <a href="https://drive.google.com/file/d/1LocR9_OsN1PT1tdpZa2vHlteeawCekhU/view?usp=sharing" target="_blank">Google Drive</a></td>
    </tr>
    <tr>
        <td rowspan="2">1-4</td>
        <td >baseline</td>
        <td >50.26</td>
        <td >-</td>
    </tr>
    <tr>
        <td >ours</td>
        <td >56.22</td>
        <td > <a href="https://drive.google.com/file/d/1eNmc--jeFwZpth8bYle7tB3VaA5_15sc/view?usp=sharing" target="_blank">Google Drive</a></td>
    </tr>
    <tr>
        <td rowspan="4">Vaihingen</td>    
        <td rowspan="2">1-8</td>
        <td >baseline</td>
        <td >53.30</td>
        <td >-</td>
    </tr>
    <tr>
        <td >ours</td>
        <td >64.09</td>
        <td > <a href="https://drive.google.com/file/d/1rk_UPaksMyD5qNXOcRRvQTTVs1JfAIdM/view?usp=sharing" target="_blank">Google Drive</a></td>
    </tr>
    <tr>
        <td rowspan="2">1-4</td>
        <td >baseline</td>
        <td >59.30</td>
        <td >-</td>
    </tr>
    <tr>
        <td >ours</td>
        <td >65.34</td>
        <td > <a href="https://drive.google.com/file/d/1WvRvgtnehhDS-q_j4ZYVP6m-IDoOVvJB/view?usp=sharing" target="_blank">Google Drive</a></td>
    </tr>
    
    

</table>
</div>


# Training and Testing
## Training
Change DATASET, SPLIT, and DATASET_PATH as you want in train.py, then run:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py
```
## Testing
Change WEIGHTS, and DATASET_PATH as you want in test.py, then run:
```
CUDA_VISIBLE_DEVICES=0,1 python test.py
```

## Acknowledgement

The code is mainly inherited from [ST++](https://github.com/LiheYoung/ST-PlusPlus), Thanks a lot for their great works!

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{lu2022simple,
  title={Simple and Efficient: A Semisupervised Learning Framework for Remote Sensing Image Semantic Segmentation},
  author={Lu, Xiaoqiang and Jiao, Licheng and Liu, Fang and Yang, Shuyuan and Liu, Xu and Feng, Zhixi and Li, Lingling and Chen, Puhua},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2022},
  publisher={IEEE}
}
```
