<div align="center">
<h1>
  SliNet: Slicing-Aided Learning for Small Object Detection
  
</h1>

<h4>
    <img width="700" alt="teaser" src="https://github.com/Gemini-wt/ASAHI/blob/e72677bbf86e558eb40edad74e58f684f9caa65e/show/tph%2Bsahi.png">
</h4>
<h4>
  This is the official repository for SliNet.

Project page: https://ieeexplore.ieee.org/document/10460167/
</h4>

[![Paper](https://img.shields.io/badge/Paper-arxiv-white)](https://ieeexplore.ieee.org/document/10460167/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-390/)
</div>


### Installation
```
conda install opencv-python==4.6.0.66
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```


### Setting

1. **Download VisDrone Dataset**  
   Download the <a href="https://github.com/VisDrone/VisDrone-Dataset" target="_blank">VisDrone</a> dataset and modify the file path in the `data/VisDrone.yaml` file accordingly like follows：
 ```
├──path: ./Project/  # dataset root dir
├── train: VisDrone2019-DET-train/images  # train images (relative to 'path')  6471 images
├── val: VisDrone2019-DET-val/images  # val images (relative to 'path')  548 images
└── test: VisDrone2019-DET-test/images  # test images (optional)  1610 images
 ```
2. **Convert Labels to YOLO Format**  
   Run the `VisDrone2YOLO_label.py` script to convert the dataset labels to YOLO format. After conversion, your dataset directory structure should resemble the following:
 ```
VisDrone2019-DET-train
 ├── annotations
 ├── images
 └── labels
VisDrone2019-DET-val
 ├── annotations
 ├── images
 └── labels
```
3. **Download weight file**

   Download our best weights <a href="https://drive.google.com/drive/folders/1YVvkL0f-YzSw2E4UKXgseeRjHs9R6-Q7?usp=sharing" target="_blank">SliNet_12slice</a> and the baseline weights from <a href="https://github.com/cv516Buaa/tph-yolov5" target="_blank">yolov5l-xs-2.pt</a>.

4. **Command for Data Augmentation(Fixed slicing size)**
   
   If you would like access to the preprocessed sliced dataset used in our paper, feel free to contact us via email.
   
   ```
   ├──Update the root path
   python tools/slice_visdrone.py
   ```
5. **Command for Testing**
   ```
   python test.py
   ```
 
6. **Command for Training**
   ```
   python train.py
   ```

7. **Command for Visualization**
   ```
   ├──Update the root path
   python Grad-CAM++.py
   ```

## 📊 Comparison on VisDrone2019-DET-val

Comparison of our method **SliNet** with other state-of-the-art object detection methods on **VisDrone2019-DET-val**.
| Models                                     | mAP  | mAP50 |
|--------------------------------------------|------|--------|
| Cascade R-CNN + NWD         | -    | 38.5   | 
| SAHI                 | 34.9 | 55.1   | 
| QueryDet                  | 33.9 | 56.1   | 
| Focus-and-Detect | 42.0 | 66.1   | 
| PP-YOLOE-l               | 29.2 | 47.3   |
| PP-YOLOE-largesize-l      | 43.3 | 66.7   | 
| TPH-YOLOV5 (baseline) | 44.4 | 63.6   | 
| **SliNet (ours)**                          | **46.4** | **67.1** |



### Interactive Visualization & Inspection

<img width="700"  src="https://github.com/Gemini-wt/ASAHI/blob/6e315492129022d1271910bf391f147ab1095dca/show/0000216_00520_d_0000001.png">
<img width="700"  src="https://github.com/Gemini-wt/ASAHI/blob/6e315492129022d1271910bf391f147ab1095dca/show/0000271_05401_d_0000399.png">
<img width="700"  src="https://github.com/Gemini-wt/ASAHI/blob/6e315492129022d1271910bf391f147ab1095dca/show/0000287_02001_d_0000769.png">
<img width="700"  src="https://github.com/Gemini-wt/ASAHI/blob/6e315492129022d1271910bf391f147ab1095dca/show/0000287_03401_d_0000776.png">
<img width="700"  src="https://github.com/Gemini-wt/ASAHI/blob/6e315492129022d1271910bf391f147ab1095dca/show/3.png">
<img width="700"  src="https://github.com/Gemini-wt/ASAHI/blob/6e315492129022d1271910bf391f147ab1095dca/show/5555.png">


## <div align="center">Citation</div>

If you use this package in your work, please cite it as:
```
@ARTICLE{10460167,
  author={Hao, Chuanyan and Zhang, Hao and Song, Wanru and Liu, Feng and Wu, Enhua},
  journal={IEEE Signal Processing Letters}, 
  title={SliNet: Slicing-Aided Learning for Small Object Detection}, 
  year={2024},
  volume={31},
  number={},
  pages={790-794},
  keywords={Transformers;Feature extraction;Detectors;Training;Head;Computational modeling;YOLO;Object detection;small object detection;sliced inference;VisDrone},
  doi={10.1109/LSP.2024.3373261}}
```
