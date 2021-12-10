#### Part-Guided Relational Transformers for Fine-Grained Visual Recognition

![./figs/pipeline.jpg]()

This code provides an implementation of the paper "Part-Guided Relational Transformers for Fine-Grained Visual Recognition".  In this paper, we propose to solve this issue in one unified framework from two aspects, i.e., constructing feature-level interrelationships, and capturing part-level discriminative features. This framework, namely **PA**rt-guided **R**elational **T**ransformers (**PART**), is proposed to learn the discriminative part features with an automatic part discovery module, and to explore the intrinsic correlations with a feature transformation module by adapting the Transformer models.

#### Running Environment:

PyTorch>=1.3, tqdm, torchvsion, Pillow, cv2.

The code is constructed using multi-GPUs (2 GPUs are recommended), tested under 2 NVIDIA-3090 or 2 NVIDIA-2080TIs.

If on other GPU settings, the hyper parameters including batchsize should be modified to achieve similar results.

#### Easy Start for PART

##### For Training:

1. Download the benchmark dataset and unzip them in your customized path.

   CUB-200-2011 http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

   FGVC-Aircraft https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

   Stanford-Cars http://ai.stanford.edu/~jkrause/cars/car_dataset.html

2. Build the Train/validation partition by yourself or download the files from [here](http://cvteam.net/projects/2021/Gard/dataset_split.zip). 

3. Modify the configuration files in /config/config.py and /config/default.py

4. Dataset config

   4-1 Set the dataset path in /config/config.py  if using CUB dataset

   4-2 Set  the dataset path in /datasets/UnifiedLoader.py if using other datasets

   4-3 Add functions and make your own datasets in  /datasets/UnifiedLoader.py

5. Modify Line 70~76, uncomment used dataset  and comment out other datasets

6. ##### Run train.py for training.

##### For Testing:

1. repeat or confirm the operations in **training** steps 1~5

2. Modify the class_num in /config/config.py

3. Put the Pretrained weights in your path

4. Modify the path to Pretrained weights in /config/config.py

5. Run test.py 



##### TIPs

> We believe the hyperparams are robust for all training datasets. Even learning rate and epochs do not need modification. Simply tuning them may lead to better results but not the main focus of our work.
>
> Moreover, the pretrained models contains many unused branches, blocks, and decoders due to our bad code construction, you do not need to save these part branches for testing. 
>
> If you only have GPUs with less than 8GB memories, you should modify the feature dimensions, part numbers and layers of transformers, which lead to slightly lower performance.

#### Playing with Pretraining

| Datasets             | CUB-200-2011                                                 | Stanford-Cars                                                | FGVC-Aircraft                                                |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Results in paper     | 90.1%                                                        | 95.3%                                                        | 94.6%                                                        |
| Results of this repo | 90.2%                                                        | 95.3%                                                        | 94.7%                                                        |
| Links                | [ResNet-101](https://drive.google.com/file/d/17vnA--_yW__jq1hI0g3JS9pDLcC7rUtP/view?usp=sharing) | [ResNet-101](https://drive.google.com/file/d/1ybaZzXSitOqapchAysLLpqE-NR4odV8U/view?usp=sharing) | [ResNet-101](https://drive.google.com/file/d/1YmCGD0X3fM-wnrFd3Me2u2rPfZvQ_Ai_/view?usp=sharing) |

###### Other related models

ResNet-50 for CUB-200-2011: [Model](https://drive.google.com/file/d/1bi5X0otF214mNizGcjCLbixiE-pHeLCZ/view?usp=sharing)



#### Acknowledgement:

Our code is based on the implementation of PyTorch official libs, [DETR](https://github.com/facebookresearch/detr), and [tiny-baseline](https://github.com/lulujianjie/person-reid-tiny-baseline).

#### Citations:

Please remember to cite us if u find this useful : )

```
@article{zhaoLCT21,
  author    = {Yifan Zhao and
               Jia Li and
               Xiaowu Chen and
               Yonghong Tian},
  title     = {Part-Guided Relational Transformers for Fine-Grained Visual Recognition},
  journal   = {{IEEE} Trans. Image Process.},
  volume    = {30},
  pages     = {9470--9481},
  year      = {2021},
  url       = {https://doi.org/10.1109/TIP.2021.3126490},
  doi       = {10.1109/TIP.2021.3126490},
  timestamp = {Tue, 30 Nov 2021 17:31:13 +0100},
}
```

#### License

Please check our License files.