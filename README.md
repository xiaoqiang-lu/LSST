# LSST

This is the official PyTorch implementation of our paper:

> [**Simple and Efficient: A Semisupervised Learning Framework for Remote Sensing Image Semantic Segmentation**      
> Xiaoqiang Lu, Licheng Jiao, Fang Liu, Shuyuan Yang, Xu Liu, Zhixi Feng, Lingling Li, Puhua Chen    
> *Submitted to IEEE Transactions on Geoscience and Remote Sensing (TGRS) 2022*

# Getting Started

## Data Preparation

### Pre-trained Model

[ResNet-101](https://download.pytorch.org/models/resnet101-63fe2227.pth)

### Dataset
We have processed the original dataset as mentioned in the paper. You can access the processed dataset directly via the following link.

[GID-15]() | [iSAID](https://drive.google.com/file/d/1yBlHpxuWs_X_02O5xAxTpo1tDWcGj6-x/view?usp=sharing) | [DFC22](https://drive.google.com/file/d/1FZoZiWW_19TWdWKUz1-Vn-IlTs0QWtfg/view?usp=sharing) | [MER](https://drive.google.com/file/d/1KN0PSoWlC4BPv5QAZdJKrhpvu9F9vdlm/view?usp=sharing) | [MSL](https://drive.google.com/file/d/1XSvmXB5rsZUi98-rWWwQtONfop7HJw4M/view?usp=sharing) | [Vaihingen](https://drive.google.com/file/d/11aOEO3-Ov3Wcg8o2nBA5GsGBqqRHbjRh/view?usp=sharing)

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
        <td > <a href="" target="_blank">Google Drive</a></td>
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
        <td > <a href="" target="_blank">Google Drive</a></td>
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
        <td > <a href="" target="_blank">Google Drive</a></td>
    </tr>
    <tr>
        <td rowspan="2">300</td>
        <td >baseline</td>
        <td >60.47</td>
        <td >-</td>
    </tr>
    <tr>
        <td >ours</td>
        <td >61.80</td>
        <td > <a href="" target="_blank">Google Drive</a></td>
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
        <td > <a href="" target="_blank">Google Drive</a></td>
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
        <td > <a href="" target="_blank">Google Drive</a></td>
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
        <td > <a href="" target="_blank">Google Drive</a></td>
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
        <td > <a href="" target="_blank">Google Drive</a></td>
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
        <td > <a href="" target="_blank">Google Drive</a></td>
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
        <td > <a href="" target="_blank">Google Drive</a></td>
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
        <td > <a href="" target="_blank">Google Drive</a></td>
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
        <td > <a href="" target="_blank">Google Drive</a></td>
    </tr>
    
    

</table>
</div>


# Training and Testing
## Training

```
CUDA_VISIBLE_DEVICES=0,1 python train.py
```


## Acknowledgement

The code is mainly inherited from [ST++](https://github.com/LiheYoung/ST-PlusPlus), Thanks a lot for their great works!

## Citation

If you find this project useful, please consider citing:

```bibtex

```
