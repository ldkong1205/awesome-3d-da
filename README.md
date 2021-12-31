# Awesome Domain Adaptation in 3D [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
- This repository is a collection of AWESOME things about **domain adaptation in the 3D world**. Feel free to submit [pull requests](https://github.com/ldkong1205/awesome-3d-da/pulls) and [open issues](https://github.com/ldkong1205/awesome-3d-da/issues) to expand our list.

- Notations:
  - :fire: : Citations > 50
  - :star: : GitHub stars > 100

- Tips: Click the **triangle** in front of the paper titles for more information

- Last updated: August 2021

## Table of Contents

- [Survey](#survey)
- [Benchmark](#benchmark)
- [Conference](#conference)
- [Journal](#journal)
- [arXiv](#arxiv)


## Survey

#### 2021
<details>
<summary>
  "<a href="https://arxiv.org/abs/2106.02377">
  A Survey on Deep Domain Adaptation for LiDAR Perception
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Larissa T. Triess, Mariella Dreissig, Christoph B. Rist, and J. Marius Zöllner</i>
  
  Scalable systems for automated driving have to reliably cope with an open-world setting. This means, the perception systems are exposed to drastic domain shifts, like changes in weather conditions, time-dependent aspects, or geographic regions. Covering all domains with annotated data is impossible because of the endless variations of domains and the time-consuming and expensive annotation process. Furthermore, fast development cycles of the system additionally introduce hardware changes, such as sensor types and vehicle setups, and the required knowledge transfer from simulation. To enable scalable automated driving, it is therefore crucial to address these domain shifts in a robust and efficient manner. Over the last years, a vast amount of different domain adaptation techniques evolved. There already exists a number of survey papers for domain adaptation on camera images, however, a survey for LiDAR perception is absent. Nevertheless, LiDAR is a vital sensor for automated driving that provides detailed 3D scans of the vehicle's surroundings. To stimulate future research, this paper presents a comprehensive review of recent progress in domain adaptation methods and formulates interesting research questions specifically targeted towards LiDAR perception.
  
</p></blockquote>
</details>


#### 2020
<details>
<summary>
  "<a href="https://arxiv.org/abs/2006.04307">
  Are We Hungry for 3D LiDAR Data for Semantic Segmentation? A Survey and Experimental Study
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Biao Gao, Yancheng Pan, Chengkun Li, Sibo Geng, and Huijing Zhao</i>
  
  3D semantic segmentation is a fundamental task for robotic and autonomous driving applications. Recent works have been focused on using deep learning techniques, whereas developing fine-annotated 3D LiDAR datasets is extremely labor intensive and requires professional skills. The performance limitation caused by insufficient datasets is called data hunger problem. This research provides a comprehensive survey and experimental study on the question: are we hungry for 3D LiDAR data for semantic segmentation? The studies are conducted at three levels. First, a broad review to the main 3D LiDAR datasets is conducted, followed by a statistical analysis on three representative datasets to gain an in-depth view on the datasets' size and diversity, which are the critical factors in learning deep models. Second, a systematic review to the state-of-the-art 3D semantic segmentation is conducted, followed by experiments and cross examinations of three representative deep learning methods to find out how the size and diversity of the datasets affect deep models' performance. Finally, a systematic survey to the existing efforts to solve the data hunger problem is conducted on both methodological and dataset's viewpoints, followed by an insightful discussion of remaining problems and open questions To the best of our knowledge, this is the first work to analyze the data hunger problem for 3D semantic segmentation using deep learning techniques that are addressed in the literature review, statistical analysis, and cross-dataset and cross-algorithm experiments. We share findings and discussions, which may lead to potential topics in future works.
  
</p></blockquote>
</details>



## Benchmark

#### 2020
<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html">
  nuScenes: A Multimodal Dataset for Autonomous Driving
  </a>," <i>CVPR</i> :fire: :star:
</summary>
<blockquote><p align="justify">
  by <i>Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom</i>
  
  Robust detection and tracking of objects is crucial for the deployment of autonomous vehicle technology. Image based benchmark datasets have driven development in computer vision tasks such as object detection, tracking and segmentation of agents in the environment. Most autonomous vehicles, however, carry a combination of cameras and range sensors such as lidar and radar. As machine learning based methods for detection and tracking become more prevalent, there is a need to train and evaluate such methods on datasets containing range sensor data along with images. In this work we present nuTonomy scenes (nuScenes), the first dataset to carry the full autonomous vehicle sensor suite: 6 cameras, 5 radars and 1 lidar, all with full 360 degree field of view. nuScenes comprises 1000 scenes, each 20s long and fully annotated with 3D bounding boxes for 23 classes and 8 attributes. It has 7x as many annotations and 100x as many images as the pioneering KITTI dataset. We define novel 3D detection and tracking metrics. We also provide careful dataset analysis as well as baselines for lidar and image based detection and tracking. Data, development kit and more information are available online.
  
  Website: https://www.nuscenes.org
  
  GitHub Repo: https://github.com/nutonomy/nuscenes-devkit
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.html">
  Scalability in Perception for Autonomous Driving: Waymo Open Dataset
  </a>," <i>CVPR</i> :fire:
</summary>
<blockquote><p align="justify">
  by <i>Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov</i>
  
  The research community has increasing interest in autonomous driving research, despite the resource intensity of obtaining representative real world data. Existing self-driving datasets are limited in the scale and variation of the environments they capture, even though generalization within and between operating regions is crucial to the over-all viability of the technology. In an effort to help align the research community's contributions with real-world self-driving problems, we introduce a new large scale, high quality, diverse dataset. Our new dataset consists of 1150 scenes that each span 20 seconds, consisting of well synchronized and calibrated high quality LiDAR and camera data captured across a range of urban and suburban geographies. It is 15x more diverse than the largest camera+LiDAR dataset available based on our proposed diversity metric. We exhaustively annotated this data with 2D (camera image) and 3D (LiDAR) bounding boxes, with consistent identifiers across frames. Finally, we provide strong baselines for 2D as well as 3D detection and tracking tasks. We further study the effects of dataset size and generalization across geographies on 3D detection methods. Find data, code and more up-to-date information at http://www.waymo.com/open.
  
  Website: http://www.waymo.com/open
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2004.06320">
  One Thousand and One Hours: Self-driving Motion Prediction Dataset
  </a>," <i>arXiv</i> :fire:
</summary>
<blockquote><p align="justify">
  by <i>John Houston, Guido Zuidhof, Luca Bergamini, Yawei Ye, Long Chen, Ashesh Jain, Sammy Omari, Vladimir Iglovikov, and Peter Ondruska
</i>
  
  Motivated by the impact of large-scale datasets on ML systems we present the largest self-driving dataset for motion prediction to date, containing over 1,000 hours of data. This was collected by a fleet of 20 autonomous vehicles along a fixed route in Palo Alto, California, over a four-month period. It consists of 170,000 scenes, where each scene is 25 seconds long and captures the perception output of the self-driving system, which encodes the precise positions and motions of nearby vehicles, cyclists, and pedestrians over time. On top of this, the dataset contains a high-definition semantic map with 15,242 labelled elements and a high-definition aerial view over the area. We show that using a dataset of this size dramatically improves performance for key self-driving problems. Combined with the provided software kit, this collection forms the largest and most detailed dataset to date for the development of self-driving machine learning tasks, such as motion forecasting, motion planning and simulation. The full dataset is available at https://level-5.global.
  
  Website: https://level-5.global
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2004.06320">
  A2D2: Audi Autonomous Driving Dataset
  </a>," <i>arXiv</i> :fire:
</summary>
<blockquote><p align="justify">
  by <i>Jakob Geyer, Yohannes Kassahun, Mentar Mahmudi, Xavier Ricou, Rupesh Durgesh, Andrew S. Chung, Lorenz Hauswald, Viet Hoang Pham, Maximilian Mühlegg, Sebastian Dorn, Tiffany Fernandez, Martin Jänicke, Sudesh Mirashi, Chiragkumar Savani, Martin Sturm, Oleksandr Vorobiov, Martin Oelker, Sebastian Garreis, and Peter Schuberth
</i>
  
  Research in machine learning, mobile robotics, and autonomous driving is accelerated by the availability of high quality annotated data. To this end, we release the Audi Autonomous Driving Dataset (A2D2). Our dataset consists of simultaneously recorded images and 3D point clouds, together with 3D bounding boxes, semantic segmentation, instance segmentation, and data extracted from the automotive bus. Our sensor suite consists of six cameras and five LiDAR units, providing full 360 degree coverage. The recorded data is time synchronized and mutually registered. Annotations are for non-sequential frames: 41,277 frames with semantic segmentation image and point cloud labels, of which 12,497 frames also have 3D bounding box annotations for objects within the field of view of the front camera. In addition, we provide 392,556 sequential frames of unannotated sensor data for recordings in three cities in the south of Germany. These sequences contain several loops. Faces and vehicle number plates are blurred due to GDPR legislation and to preserve anonymity. A2D2 is made available under the CC BY-ND 4.0 license, permitting commercial use subject to the terms of the license. Data and further information are available at http://www.a2d2.audi.
  
  Website: https://www.a2d2.audi/a2d2
  
</p></blockquote>
</details>


#### 2019
<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content_ICCV_2019/html/Behley_SemanticKITTI_A_Dataset_for_Semantic_Scene_Understanding_of_LiDAR_Sequences_ICCV_2019_paper.html">
  SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences
  </a>," <i>CVPR</i> :fire:
</summary>
<blockquote><p align="justify">
  by <i>Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke, Cyrill Stachniss, and Jurgen Gall</i>
  
  Semantic scene understanding is important for various applications. In particular, self-driving cars need a fine-grained understanding of the surfaces and objects in their vicinity. Light detection and ranging (LiDAR) provides precise geometric information about the environment and is thus a part of the sensor suites of almost all self-driving cars. Despite the relevance of semantic scene understanding for this application, there is a lack of a large dataset for this task which is based on an automotive LiDAR. In this paper, we introduce a large dataset to propel research on laser-based semantic segmentation. We annotated all sequences of the KITTI Vision Odometry Benchmark and provide dense point-wise annotations for the complete 360-degree field-of-view of the employed automotive LiDAR. We propose three benchmark tasks based on this dataset: (i) semantic segmentation of point clouds using a single scan, (ii) semantic segmentation using multiple past scans, and (iii) semantic scene completion, which requires to anticipate the semantic scene in the future. We provide baseline experiments and show that there is a need for more sophisticated models to efficiently tackle these tasks. Our dataset opens the door for the development of more advanced methods, but also provides plentiful data to investigate new research directions.
  
  Website: http://www.semantic-kitti.org
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Chang_Argoverse_3D_Tracking_and_Forecasting_With_Rich_Maps_CVPR_2019_paper.html">
  Argoverse: 3D Tracking and Forecasting With Rich Maps
  </a>," <i>CVPR</i> :fire:
</summary>
<blockquote><p align="justify">
  by <i>Ming-Fang Chang, John Lambert, Patsorn Sangkloy, Jagjeet Singh, Slawomir Bak, Andrew Hartnett, De Wang, Peter Carr, Simon Lucey, Deva Ramanan, and James Hays</i>
  
  We present Argoverse, a dataset designed to support autonomous vehicle perception tasks including 3D tracking and motion forecasting. Argoverse includes sensor data collected by a fleet of autonomous vehicles in Pittsburgh and Miami as well as 3D tracking annotations, 300k extracted interesting vehicle trajectories, and rich semantic maps. The sensor data consists of 360 degree images from 7 cameras with overlapping fields of view, forward-facing stereo imagery, 3D point clouds from long range LiDAR, and 6-DOF pose. Our 290km of mapped lanes contain rich geometric and semantic metadata which are not currently available in any public dataset. All data is released under a Creative Commons license at Argoverse.org. In baseline experiments, we use map information such as lane direction, driveable area, and ground height to improve the accuracy of 3D object tracking. We use 3D object tracking to mine for more than 300k interesting vehicle trajectories to create a trajectory forecasting benchmark. Motion forecasting experiments ranging in complexity from classical methods (k-NN) to LSTMs demonstrate that using detailed vector maps with lane-level information substantially reduces prediction error. Our tracking and forecasting experiments represent only a superficial exploration of the potential of rich maps in robotic perception. We hope that Argoverse will enable the research community to explore these problems in greater depth.
  
  Website: https://www.argoverse.org
  
</p></blockquote>
</details>


#### 2017
<details>
<summary>
  "<a href="https://robotcar-dataset.robots.ox.ac.uk/images/robotcar_ijrr.pdf">
  1 Year, 1000 KM: The Oxford RobotCar Dataset
  </a>," <i>IJRR</i> :fire:
</summary>
<blockquote><p align="justify">
  by <i>Will Maddern, Geoffrey Pascoe, Chris Linegar, and Paul Newman</i>
  
  We present a challenging new dataset for autonomous driving: the Oxford RobotCar Dataset. Over the period of May 2014 to December 2015 we traversed a route through central Oxford twice a week on average using the Oxford RobotCar platform, an autonomous Nissan LEAF. This resulted in over 1000 km of recorded driving with almost 20 million images collected from 6 cameras mounted to the vehicle, along with LIDAR, GPS and INS ground truth. Data was collected in all weather conditions, including heavy rain, night, direct sunlight and snow. Road and building works over the period of a year significantly changed sections of the route from the beginning to the end of data collection. By frequently traversing the same route over the period of a year we enable research investigating long-term localization and mapping for autonomous vehicles in real-world, dynamic urban environments. The full dataset is available for download at: http://robotcar-dataset.robots.ox.ac.uk
  
</p></blockquote>
</details>



## Conference

#### 2021
<details>
<summary>
  "<a href="https://arxiv.org/abs/2107.14724">
  Sparse-to-Dense Feature Matching: Intra and Inter domain Cross-modal Learning in Domain Adaptation for 3D Semantic Segmentation
  </a>," <i>ICCV</i>
</summary>
<blockquote><p align="justify">
  by <i>Duo Peng, Yinjie Lei, Wen Li, Pingping Zhang, and Yulan Guo</i>
  
  Domain adaptation is critical for success when confronting with the lack of annotations in a new domain. As the huge time consumption of labeling process on 3D point cloud, domain adaptation for 3D semantic segmentation is of great expectation. With the rise of multi-modal datasets, large amount of 2D images are accessible besides 3D point clouds. In light of this, we propose to further leverage 2D data for 3D domain adaptation by intra and inter domain cross modal learning. As for intra-domain cross modal learning, most existing works sample the dense 2D pixel-wise features into the same size with sparse 3D point-wise features, resulting in the abandon of numerous useful 2D features. To address this problem, we propose Dynamic sparse-to-dense Cross Modal Learning (DsCML) to increase the sufficiency of multi-modality information interaction for domain adaptation. For inter-domain cross modal learning, we further advance Cross Modal Adversarial Learning (CMAL) on 2D and 3D data which contains different semantic content aiming to promote high-level modal complementarity. We evaluate our model under various multi-modality domain adaptation settings including day-to-night, country-to-country and dataset-to-dataset, brings large improvements over both uni-modal and multi-modal domain adaptation methods on all settings.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2107.11355">
  Unsupervised Domain Adaptive 3D Detection with Multi-Level Consistency
  </a>," <i>ICCV</i>
</summary>
<blockquote><p align="justify">
  by <i>Zhipeng Luo, Zhongang Cai, Changqing Zhou, Gongjie Zhang, Haiyu Zhao, Shuai Yi, Shijian Lu, Hongsheng Li, Shanghang Zhang, and Ziwei Liu</i>
  
  Deep learning-based 3D object detection has achieved unprecedented success with the advent of large-scale autonomous driving datasets. However, drastic performance degradation remains a critical challenge for cross-domain deployment. In addition, existing 3D domain adaptive detection methods often assume prior access to the target domain annotations, which is rarely feasible in the real world. To address this challenge, we study a more realistic setting, unsupervised 3D domain adaptive detection, which only utilizes source domain annotations. 1) We first comprehensively investigate the major underlying factors of the domain gap in 3D detection. Our key insight is that geometric mismatch is the key factor of domain shift. 2) Then, we propose a novel and unified framework, Multi-Level Consistency Network (MLC-Net), which employs a teacher-student paradigm to generate adaptive and reliable pseudo-targets. MLC-Net exploits point-, instance- and neural statistics-level consistency to facilitate cross-domain transfer. Extensive experiments demonstrate that MLC-Net outperforms existing state-of-the-art methods (including those using additional target domain information) on standard benchmarks. Notably, our approach is detector-agnostic, which achieves consistent gains on both single- and two-stage 3D detectors.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_SRDAN_Scale-Aware_and_Range-Aware_Domain_Adaptation_Network_for_Cross-Dataset_3D_CVPR_2021_paper.html">
  SRDAN: Scale-Aware and Range-Aware Domain Adaptation Network for Cross-Dataset 3D Object Detection
  </a>," <i>CVPR</i>
</summary>
<blockquote><p align="justify">
  by <i>Weichen Zhang, Wen Li, and Dong Xu</i>
  
  Geometric characteristic plays an important role in the representation of an object in 3D point clouds. For example, large objects often contain more points, while small ones contain fewer points. The point clouds of objects near the capture device are denser, while those of distant objects are sparser. These issues bring new challenges to 3D object detection, especially under the domain adaptation scenarios. In this work, we propose a new cross-dataset 3D object detection method named Scale-aware and Range-aware Domain Adaptation Network (SRDAN). We take advantage of the geometric characteristics of 3D data (i.e., size and distance), and propose the scale-aware domain alignment and the range-aware domain alignment strategies to guide the distribution alignment between two domains. For scale-aware domain alignment, we design a 3D voxel-based feature pyramid network to extract multi-scale semantic voxel features, and align the features and instances with similar scales between two domains. For range-aware domain alignment, we introduce a range-guided domain alignment module to align the features of objects according to their distance to the capture device. Extensive experiments under three different scenarios demonstrate the effectiveness of our SRDAN approach, and comprehensive ablation study also validates the importance of geometric characteristics for cross-dataset 3D object detection.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content/CVPR2021/html/Yang_ST3D_Self-Training_for_Unsupervised_Domain_Adaptation_on_3D_Object_Detection_CVPR_2021_paper.html">
  ST3D: Self-Training for Unsupervised Domain Adaptation on 3D Object Detection
  </a>," <i>CVPR</i> :star:
</summary>
<blockquote><p align="justify">
  by <i>Jihan Yang, Shaoshuai Shi, Zhe Wang, Hongsheng Li, and Xiaojuan Qi</i>
  
  We present a new domain adaptive self-training pipeline, named ST3D, for unsupervised domain adaptation on 3D object detection from point clouds. First, we pre-train the 3D detector on the source domain with our proposed random object scaling strategy for mitigating the negative effects of source domain bias. Then, the detector is iteratively improved on the target domain by alternatively conducting two steps, which are the pseudo label updating with the developed quality-aware triplet memory bank and the model training with curriculum data augmentation. These specific designs for 3D object detection enable the detector to be trained with consistent and high-quality pseudo labels and to avoid overfitting to the large number of easy examples in pseudo labeled data. Our ST3D achieves state-of-the-art performance on all evaluated datasets and even surpasses fully supervised results on KITTI 3D object detection benchmark. Code will be available at https://github.com/CVMI-Lab/ST3D.
  
  GitHub Repo: https://github.com/CVMI-Lab/ST3D
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content/CVPR2021/html/Yi_Complete__Label_A_Domain_Adaptation_Approach_to_Semantic_Segmentation_CVPR_2021_paper.html">
  Complete & Label: A Domain Adaptation Approach to Semantic Segmentation of LiDAR Point Clouds
  </a>," <i>CVPR</i>
</summary>
<blockquote><p align="justify">
  by <i>Li Yi, Boqing Gong, and Thomas Funkhouser</i>
  
  We study an unsupervised domain adaptation problem for the semantic labeling of 3D point clouds, with a particular focus on domain discrepancies induced by different LiDAR sensors. Based on the observation that sparse 3D point clouds are sampled from 3D surfaces, we take a Complete and Label approach to recover the underlying surfaces before passing them to a segmentation network. Specifically, we design a Sparse Voxel Completion Network (SVCN) to complete the 3D surfaces of a sparse point cloud. Unlike semantic labels, to obtain training pairs for SVCN requires no manual labeling. We also introduce local adversarial learning to model the surface prior. The recovered 3D surfaces serve as a canonical domain, from which semantic labels can transfer across different LiDAR sensors. Experiments and ablation studies with our new benchmark for cross-domain semantic labeling of LiDAR data show that the proposed approach provides 6.3-37.6% better performance than previous domain adaptation methods.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content/WACV2021/html/Achituve_Self-Supervised_Learning_for_Domain_Adaptation_on_Point_Clouds_WACV_2021_paper.html">
  Self-Supervised Learning for Domain Adaptation on Point Clouds
  </a>," <i>WACV</i>
</summary>
<blockquote><p align="justify">
  by <i>Idan Achituve, Haggai Maron, and Gal Chechik</i>
  
  Self-supervised learning (SSL) is a technique for learning useful representations from unlabeled data. It has been applied effectively to domain adaptation (DA) on images and videos. It is still unknown if and how it can be leveraged for domain adaptation in 3D perception problems. Here we describe the first study of SSL for DA on point clouds. We introduce a new family of pretext tasks, Deformation Reconstruction, inspired by the deformations encountered in sim-to-real transformations. In addition, we propose a novel training procedure for labeled point cloud data motivated by the MixUp method called Point cloud Mixup (PCM). Evaluations on domain adaptations datasets for classification and segmentation, demonstrate a large improvement over existing and baseline methods.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2009.03456">
  ePointDA: An End-to-End Simulation-to-Real Domain Adaptation Framework for LiDAR Point Cloud Segmentation
  </a>," <i>AAAI</i>
</summary>
<blockquote><p align="justify">
  by <i>Sicheng Zhao, Yezhen Wang, Bo Li, Bichen Wu, Yang Gao, Pengfei Xu, Trevor Darrell, and Kurt Keutzer</i>
  
  Due to its robust and precise distance measurements, LiDAR plays an important role in scene understanding for autonomous driving. Training deep neural networks (DNNs) on LiDAR data requires large-scale point-wise annotations, which are time-consuming and expensive to obtain. Instead, simulation-to-real domain adaptation (SRDA) trains a DNN using unlimited synthetic data with automatically generated labels and transfers the learned model to real scenarios. Existing SRDA methods for LiDAR point cloud segmentation mainly employ a multi-stage pipeline and focus on feature-level alignment. They require prior knowledge of real-world statistics and ignore the pixel-level dropout noise gap and the spatial feature gap between different domains. In this paper, we propose a novel end-to-end framework, named ePointDA, to address the above issues. Specifically, ePointDA consists of three modules: self-supervised dropout noise rendering, statistics-invariant and spatially-adaptive feature alignment, and transferable segmentation learning. The joint optimization enables ePointDA to bridge the domain shift at the pixel-level by explicitly rendering dropout noise for synthetic LiDAR and at the feature-level by spatially aligning the features between different domains, without requiring the real-world statistics. Extensive experiments adapting from synthetic GTA-LiDAR to real KITTI and SemanticKITTI demonstrate the superiority of ePointDA for LiDAR point cloud segmentation.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://ojs.aaai.org/index.php/AAAI/article/view/16278">
  Dynamic to Static LiDAR Scan Reconstruction Using Adversarially Trained Auto Encoder
  </a>," <i>AAAI</i>
</summary>
<blockquote><p align="justify">
  by <i>Prashant Kumar, Sabyasachi Sahoo, Vanshil Shah, Vineetha Kondameedi, Abhinav Jain, Akshaj Verma, Chiranjib Bhattacharyya, and Vinay Vishwanath</i>
  
  Accurate reconstruction of static environments from LiDAR scans of scenes containing dynamic objects, which we refer to as Dynamic to Static Translation (DST), is an important area of research in Autonomous Navigation. This problem has been recently explored for visual SLAM, but to the best of our knowledge no work has been attempted to address DST for LiDAR scans. The problem is of critical importance due to wide-spread adoption of LiDAR in Autonomous Vehicles. We show that state-of the art methods developed for the visual domain when adapted for LiDAR scans perform poorly. We develop DSLR, a deep generative model which learns a mapping between dynamic scan to its static counterpart through an adversarially trained autoencoder. Our model yields the first solution for DST on LiDAR that generates static scans without using explicit segmentation labels. DSLR cannot always be applied to real world data due to lack of paired dynamic-static scans. Using Unsupervised Domain Adaptation, we propose DSLR-UDA for transfer to real world data and experimentally show that this performs well in real world settings. Additionally, if segmentation information is available, we extend DSLR to DSLR-Seg to further improve the reconstruction quality. DSLR gives the state of the art performance on simulated and real-world datasets and also shows at least 4× improvement. We show that DSLR, unlike the existing baselines, is a practically viable model with its reconstruction quality within the tolerable limits for tasks pertaining to autonomous navigation like SLAM in dynamic environments.
  
</p></blockquote>
</details>


#### 2020
<details>
<summary>
  "<a href="https://link.springer.com/chapter/10.1007%2F978-3-030-58545-7_2">
  Monocular 3D Object Detection via Feature Domain Adaptation
  </a>," <i>ECCV</i>
</summary>
<blockquote><p align="justify">
  by <i>Xiaoqing Ye, Liang Du, Yifeng Shi, Yingying Li, Xiao Tan, Jianfeng Feng, Errui Ding, and Shilei Wen</i>
  
  Monocular 3D object detection is a challenging task due to unreliable depth, resulting in a distinct performance gap between monocular and LiDAR-based approaches. In this paper, we propose a novel domain adaptation based monocular 3D object detection framework named DA-3Ddet, which adapts the feature from unsound image-based pseudo-LiDAR domain to the accurate real LiDAR domain for performance boosting. In order to solve the overlooked problem of inconsistency between the foreground mask of pseudo and real LiDAR caused by inaccurately estimated depth, we also introduce a context-aware foreground segmentation module which helps to involve relevant points for foreground masking. Extensive experiments on KITTI dataset demonstrate that our simple yet effective framework outperforms other state-of-the-arts by a large margin.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Jaritz_xMUDA_Cross-Modal_Unsupervised_Domain_Adaptation_for_3D_Semantic_Segmentation_CVPR_2020_paper.html">
  xMUDA: Cross-Modal Unsupervised Domain Adaptation for 3D Semantic Segmentation
  </a>," <i>CVPR</i> :star:
</summary>
<blockquote><p align="justify">
  by <i>Maximilian Jaritz, Tuan-Hung Vu, Raoul de Charette, Emilie Wirbel, and Patrick Perez</i>
  
  Unsupervised Domain Adaptation (UDA) is crucial to tackle the lack of annotations in a new domain. There are many multi-modal datasets, but most UDA approaches are uni-modal. In this work, we explore how to learn from multi-modality and propose cross-modal UDA (xMUDA) where we assume the presence of 2D images and 3D point clouds for 3D semantic segmentation. This is challenging as the two input spaces are heterogeneous and can be impacted differently by domain shift. In xMUDA, modalities learn from each other through mutual mimicking, disentangled from the segmentation objective, to prevent the stronger modality from adopting false predictions from the weaker one. We evaluate on new UDA scenarios including day-to-night, country-to-country and dataset-to-dataset, leveraging recent autonomous driving datasets. xMUDA brings large improvements over uni-modal UDA on all tested scenarios, and is complementary to state-of-the-art UDA techniques. Code is available at https://github.com/valeoai/xmuda.
  
  GitHub Repo: https://github.com/valeoai/xmuda
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Baek_Weakly-Supervised_Domain_Adaptation_via_GAN_and_Mesh_Model_for_Estimating_CVPR_2020_paper.html">
  Weakly-Supervised Domain Adaptation via GAN and Mesh Model for Estimating 3D Hand Poses Interacting Objects
  </a>," <i>CVPR</i>
</summary>
<blockquote><p align="justify">
  by <i>Seungryul Baek, Kwang In Kim, and Tae-Kyun Kim</i>
  
  Despite recent successes in hand pose estimation, there yet remain challenges on RGB-based 3D hand pose estimation (HPE) under hand-object interaction (HOI) scenarios where severe occlusions and cluttered backgrounds exhibit. Recent RGB HOI benchmarks have been collected either in real or synthetic domain, however, the size of datasets is far from enough to deal with diverse objects combined with hand poses, and 3D pose annotations of real samples are lacking, especially for occluded cases. In this work, we propose a novel end-to-end trainable pipeline that adapts the hand-object domain to the single hand-only domain, while learning for HPE. The domain adaption occurs in image space via 2D pixel-level guidance by Generative Adversarial Network (GAN) and 3D mesh guidance by mesh renderer (MR). Via the domain adaption in image space, not only 3D HPE accuracy is improved, but also HOI input images are translated to segmented and de-occluded hand-only images. The proposed method takes advantages of both the guidances: GAN accurately aligns hands, while MR effectively fills in occluded pixels. The experiments using Dexter-Object, Ego-Dexter and HO3D datasets show that our method significantly outperforms state-of-the-arts trained by hand-only data and is comparable to those supervised by HOI data. Note our method is trained primarily by hand-only images with pose labels, and HOI images without pose labels.
  
</p></blockquote>
</details>

<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Train_in_Germany_Test_in_the_USA_Making_3D_Object_CVPR_2020_paper.html">
  Train in Germany, Test in the USA: Making 3D Object Detectors Generalize
  </a>," <i>CVPR</i>
</summary>
<blockquote><p align="justify">
  by <i>Yan Wang, Xiangyu Chen, Yurong You, Li Erran Li, Bharath Hariharan, Mark Campbell, Kilian Q. Weinberger, and Wei-Lun Chao</i>
  
  In the domain of autonomous driving, deep learning has substantially improved the 3D object detection accuracy for LiDAR and stereo camera data alike. While deep networks are great at generalization, they are also notorious to overfit to all kinds of spurious artifacts, such as brightness, car sizes and models, that may appear consistently throughout the data. In fact, most datasets for autonomous driving are collected within a narrow subset of cities within one country, typically under similar weather conditions. In this paper we consider the task of adapting 3D object detectors from one dataset to another. We observe that naively, this appears to be a very challenging task, resulting in drastic drops in accuracy levels. We provide extensive experiments to investigate the true adaptation challenges and arrive at a surprising conclusion: the primary adaptation hurdle to overcome are differences in car sizes across geographic areas. A simple correction based on the average car size yields a strong correction of the adaptation gap. Our proposed method is simple and easily incorporated into most 3D object detection frameworks. It provides a first baseline for 3D object detection adaptation across countries, and gives hope that the underlying problem may be more within grasp than one may have hoped to believe. Our code is available at https://github.com/cxy1997/3D_adapt_auto_driving.
  
  GitHub Repo: https://github.com/cxy1997/3D_adapt_auto_driving
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="http://ras.papercept.net/images/temp/IROS/files/0060.pdf">
  Domain Transfer for Semantic Segmentation of LiDAR Data using Deep Neural Networks
  </a>," <i>IROS</i>
</summary>
<blockquote><p align="justify">
  by <i>Ferdinand Langer, Andres Milioto, Alexandre Haag, Jens Behley, and Cyrill Stachniss</i>
  
  Inferring semantic information towards an understanding of the surrounding environment is crucial for autonomous vehicles to drive safely. Deep learning-based segmentation methods can infer semantic information directly from laser range data, even in the absence of other sensor modalities such as cameras. In this paper, we address improving the generalization capabilities of such deep learning models to range data that was captured using a different sensor and in situations where no labeled data is available for the new sensor setup. Our approach assists the domain transfer of a LiDAR-only semantic segmentation model to a different sensor and environment exploiting existing geometric mapping systems. To this end, we fuse sequential scans in the source dataset into a dense mesh and render semi-synthetic scans that match those of the target sensor setup. Unlike simulation, this approach provides a real-to-real transfer of geometric information and delivers additionally more accurate remission information. We implemented and thoroughly tested our approach by transferring semantic scans between two different real-world datasets with different sensor setups. Our experiments show that we can improve the segmentation performance substantially with zero manual re-labeling. This approach solves the number one feature request since we released our semantic segmentation library LiDAR-bonnetal.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/1712.05765">
  SF-UDA3D: Source-Free Unsupervised Domain Adaptation for LiDAR-Based 3D Object Detection
  </a>," <i>3DV</i>
</summary>
<blockquote><p align="justify">
  by <i>Cristiano Saltori, Stéphane Lathuiliére, Nicu Sebe, Elisa Ricci, and Fabio Galasso</i>
  
  3D object detectors based only on LiDAR point clouds hold the state-of-the-art on modern street-view benchmarks. However, LiDAR-based detectors poorly generalize across domains due to domain shift. In the case of LiDAR, in fact, domain shift is not only due to changes in the environment and in the object appearances, as for visual data from RGB cameras, but is also related to the geometry of the point clouds (e.g., point density variations). This paper proposes SF-UDA3D, the first Source-Free Unsupervised Domain Adaptation (SF-UDA) framework to domain-adapt the state-of-the-art PointRCNN 3D detector to target domains for which we have no annotations (unsupervised), neither we hold images nor annotations of the source domain (source-free). SF-UDA3D is novel on both aspects. Our approach is based on pseudo-annotations, reversible scale-transformations and motion coherency. SF-UDA3D outperforms both previous domain adaptation techniques based on features alignment and state-of-the-art 3D object detection methods which additionally use few-shot target annotations or target annotation statistics. This is demonstrated by extensive experiments on two large-scale datasets, i.e., KITTI and nuScenes.
  
  GitHub Repo: https://github.com/saltoricristiano/SF-UDA-3DV
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://ieeexplore.ieee.org/abstract/document/9294540">
  Unsupervised Evaluation of Lidar Domain Adaptation
  </a>," <i>ITSC</i>
</summary>
<blockquote><p align="justify">
  by <i>Christian Hubschneider, Simon Roesler, and J. Marius Zöllner</i>
  
  In this work, we investigate the potential of latent representations generated by Variational Autoencoders (VAE) to analyze and distinguish between real and synthetic data. Although the details of the domain adaptation task are not the focus of this work, we use the example of simulated lidar data adapted by a generative model to match real lidar data. To assess the resulting adapted data, we evaluate the potential of latent representations learned by a VAE. During training, the VAE aims to reduce the input data to a fixed-dimensional feature vector, while also enforcing stochastic independence between the latent variables. These properties can be used to define pseudometrics to make statements about generative models that perform domain adaptation tasks. The variational autoencoder is trained on real target data only and is subsequently used to generate distributions of feature vectors for data coming from different data sources such as simulations or the output of Generative Adversarial Networks.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content/ACCV2020/html/Lopez-Rodriguez_Project_to_Adapt_Domain_Adaptation_for_Depth_Completion_from_Noisy_ACCV_2020_paper.html">
  Project to Adapt: Domain Adaptation for Depth Completion from Noisy and Sparse Sensor Data
  </a>," <i>ACCV</i>
</summary>
<blockquote><p align="justify">
  by <i>Adrian Lopez-Rodriguez, Benjamin Busam, and Krystian Mikolajczyk</i>
  
  Depth completion aims to predict a dense depth map from a sparse depth input. The acquisition of dense ground truth annotations for depth completion settings can be difficult and, at the same time, a significant domain gap between real LiDAR measurements and synthetic data has prevented from successful training of models in virtual settings. We propose a domain adaptation approach for sparse-to-dense depth completion that is trained from synthetic data, without annotations in the real domain or additional sensors. Our approach simulates the real sensor noise in an RGB + LiDAR set-up, and consists of three modules: simulating the real LiDAR input in the synthetic domain via projections, filtering the real noisy LiDAR for supervision and adapting the synthetic RGB image using a CycleGAN approach. We extensively evaluate these modules against the state-of-the-art in the KITTI depth completion benchmark, showing significant improvements.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content/ACCV2020/html/Wang_L2R_GAN_LiDAR-to-Radar_Translation_ACCV_2020_paper.html">
  L2R GAN: LiDAR-to-Radar Translation
  </a>," <i>ACCV</i>
</summary>
<blockquote><p align="justify">
  by <i>Leichen Wang, Bastian Goldluecke, and Carsten Anklam</i>
  
  The lack of annotated public radar datasets causes difficulties for research in environmental perception from radar observations. In this paper, we propose a novel neural network based framework which we call L2R GAN to generate the radar spectrum of natural scenes from a given LiDAR point cloud.We adapt ideas from existing image-to-image translation GAN frameworks, which we investigate as a baseline for translating radar spectra image from a given LiDAR bird's eye view (BEV). However, for our application, we identify several shortcomings of existing approaches. As a remedy, we learn radar data generation with an occupancy-grid-mask as a guidance, and further design a set of local region generators and discriminator networks. This allows our L2R GAN to combine the advantages of global image features and local region detail, and not only learn the cross-modal relations between LiDAR and radar in large scale, but also refine details in small scale. Qualitative and quantitative comparison show that L2R GAN outperforms previous GAN architectures with respect to details by a large margin. A L2R-GAN-based GUI also allows users to define and generate radar data of special emergency scenarios to test corresponding ADAS applications such as Pedestrian Collision Warning (PCW).
  
</p></blockquote>
</details>


#### 2019
<details>
<summary>
  "<a href="https://arxiv.org/abs/1911.02744">
  Domain-Adaptive Single-View 3D Reconstruction
  </a>," <i>ICCV</i>
</summary>
<blockquote><p align="justify">
  by <i>Pedro O. Pinheiro, Negar Rostamzadeh, and Sungjin Ahn</i>
  
  Single-view 3D shape reconstruction is an important but challenging problem, mainly for two reasons. First, as shape annotation is very expensive to acquire, current methods rely on synthetic data, in which ground-truth 3D annotation is easy to obtain. However, this results in domain adaptation problem when applied to natural images. The second challenge is that there are multiple shapes that can explain a given 2D image. In this paper, we propose a framework to improve over these challenges using adversarial training. On one hand, we impose domain confusion between natural and synthetic image representations to reduce the distribution gap. On the other hand, we impose the reconstruction to be `realistic' by forcing it to lie on a (learned) manifold of realistic object shapes. Our experiments show that these constraints improve performance by a large margin over baseline reconstruction models. We achieve results competitive with the state of the art with a much simpler architecture.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content_ICCVW_2019/html/CVRSUAD/Weng_Monocular_3D_Object_Detection_with_Pseudo-LiDAR_Point_Cloud_ICCVW_2019_paper.html">
  Monocular 3D Object Detection with Pseudo-LiDAR Point Cloud
  </a>," <i>ICCVW</i> :fire: :star:
</summary>
<blockquote><p align="justify">
  by <i>Xinshuo Weng and Kris Kitani</i>
  
  Monocular 3D scene understanding tasks, such as object size estimation, heading angle estimation and 3D localization, is challenging. Successful modern-day methods for 3D scene understanding require the use of a 3D sensor. On the other hand, single image-based methods have significantly worse performance. In this work, we aim at bridging the performance gap between 3D sensing and 2D sensing for 3D object detection by enhancing LiDAR-based algorithms to work with single image input. Specifically, we perform monocular depth estimation and lift the input image to a point cloud representation, which we call pseudo-LiDAR point cloud. Then we can train a LiDAR-based 3D detection network with our pseudo-LiDAR end-to-end. Following the pipeline of two-stage 3D detection algorithms, we detect 2D object proposals in the input image and extract a point cloud frustum from the pseudo-LiDAR for each proposal. Then an oriented 3D bounding box is detected for each frustum. To handle the large amount of noise in the pseudo-LiDAR, we propose two innovations: (1) use a 2D-3D bounding box consistency constraint, adjusting the predicted 3D bounding box to have a high overlap with its corresponding 2D proposal after projecting onto the image; (2) use the instance mask instead of the bounding box as the representation of 2D proposals, in order to reduce the number of points not belonging to the object in the point cloud frustum. Through our evaluation on the KITTI benchmark, we achieve the top-ranked performance on both bird's eye view and 3D object detection among all monocular methods, effectively quadrupling the performance over previous state-of-the-art. Our code is available at https://github.com/xinshuoweng/Mono3D_PLiDAR.
  
  GitHub Repo: https://github.com/xinshuoweng/Mono3D_PLiDAR
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content_ICCVW_2019/html/ADW/Wang_Range_Adaptation_for_3D_Object_Detection_in_LiDAR_ICCVW_2019_paper.html">
  Range Adaptation for 3D Object Detection in LiDAR
  </a>," <i>ICCVW</i>
</summary>
<blockquote><p align="justify">
  by <i>Ze Wang, Sihao Ding, Ying Li, Minming Zhao, Sohini Roychowdhury, Andreas Wallin, Guillermo Sapiro, and Qiang Qiu</i>
  
  LiDAR-based 3D object detection plays a crucial role in modern autonomous driving systems. LiDAR data often exhibit severe changes in properties across different observation ranges. In this paper, we explore cross-range adaptation for 3D object detection using LiDAR, i.e., far-range observations are adapted to near-range. This way, far-range detection is optimized for similar performance to near-range one. We adopt a bird-eyes view (BEV) detection framework to perform the proposed model adaptation. Our model adaptation consists of an adversarial global adaptation, and a fine-grained local adaptation. The proposed cross-range adaptation framework is validated on three state-of-the-art LiDAR based object detection networks, and we consistently observe performance improvement on the far-range objects, without adding any auxiliary parameters to the model. To the best of our knowledge, this paper is the first attempt to study cross-range LiDAR adaptation for object detection in point clouds. To demonstrate the generality of the proposed adaptation framework, experiments on more challenging cross-device adaptation are further conducted, and a new LiDAR dataset with high-quality annotated point clouds is released to promote future research.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://openaccess.thecvf.com/content_ICCVW_2019/html/TASK-CV/Saleh_Domain_Adaptation_for_Vehicle_Detection_from_Birds_Eye_View_LiDAR_ICCVW_2019_paper.html">
  Domain Adaptation for Vehicle Detection from Bird's Eye View LiDAR Point Cloud Data
  </a>," <i>ICCVW</i>
</summary>
<blockquote><p align="justify">
  by <i>Khaled Saleh, Ahmed Abobakr, Mohammed Attia, Julie Iskander, Darius Nahavandi, Mohammed Hossny, and Saeid Nahvandi</i>
  
  Point cloud data from 3D LiDAR sensors are one of the most crucial sensor modalities for versatile safety-critical applications such as self-driving vehicles. Since the annotations of point cloud data is an expensive and time-consuming process, therefore recently the utilisation of simulated environments and 3D LiDAR sensors for this task started to get some popularity. However, the generated synthetic point cloud data are still missing the artefacts usually exist in point cloud data from real 3D LiDAR sensors. Thus, in this work, we are proposing a domain adaptation framework for bridging this gap between synthetic and real point cloud data. Our proposed framework is based on the deep cycle-consistent generative adversarial networks (CycleGAN) architecture. We have evaluated the performance of our proposed framework on the task of vehicle detection from a bird's eye view (BEV) point cloud images coming from real 3D LiDAR sensors. The framework has shown competitive results with an improvement of more than 7% in average precision score over other baseline approaches when tested on real BEV point cloud images.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/1911.02744">
  PointDAN: A Multi-Scale 3D Domain Adaption Network for Point Cloud Representation
  </a>," <i>NeurIPS</i> :star:
</summary>
<blockquote><p align="justify">
  by <i>Can Qin, Haoxuan You, Lichen Wang, C.-C. Jay Kuo, and Yun Fu</i>
  
  Domain Adaptation (DA) approaches achieved significant improvements in a wide range of machine learning and computer vision tasks (i.e., classification, detection, and segmentation). However, as far as we are aware, there are few methods yet to achieve domain adaptation directly on 3D point cloud data. The unique challenge of point cloud data lies in its abundant spatial geometric information, and the semantics of the whole object is contributed by including regional geometric structures. Specifically, most general-purpose DA methods that struggle for global feature alignment and ignore local geometric information are not suitable for 3D domain alignment. In this paper, we propose a novel 3D Domain Adaptation Network for point cloud data (PointDAN). PointDAN jointly aligns the global and local features in multi-level. For local alignment, we propose Self-Adaptive (SA) node module with an adjusted receptive field to model the discriminative local structures for aligning domains. To represent hierarchically scaled features, node-attention module is further introduced to weight the relationship of SA nodes across objects and domains. For global alignment, an adversarial-training strategy is employed to learn and align global features across domains. Since there is no common evaluation benchmark for 3D point cloud DA scenario, we build a general benchmark (i.e., PointDA-10) extracted from three popular 3D object/scene datasets (i.e., ModelNet, ShapeNet and ScanNet) for cross-domain 3D objects classification fashion. Extensive experiments on PointDA-10 illustrate the superiority of our model over the state-of-the-art general-purpose DA methods.
  
  GitHub Repo: https://github.com/canqin001/PointDAN
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/1809.08495">
  SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud
  </a>," <i>ICRA</i> :fire: :star:
</summary>
<blockquote><p align="justify">
  by <i>Bichen Wu, Xuanyu Zhou, Sicheng Zhao, Xiangyu Yue, and Kurt Keutzer</i>
  
  Earlier work demonstrates the promise of deep-learning-based approaches for point cloud segmentation; however, these approaches need to be improved to be practically useful. To this end, we introduce a new model SqueezeSegV2 that is more robust to dropout noise in LiDAR point clouds. With improved model structure, training loss, batch normalization and additional input channel, SqueezeSegV2 achieves significant accuracy improvement when trained on real data. Training models for point cloud segmentation requires large amounts of labeled point-cloud data, which is expensive to obtain. To sidestep the cost of collection and annotation, simulators such as GTA-V can be used to create unlimited amounts of labeled, synthetic data. However, due to domain shift, models trained on synthetic data often do not generalize well to the real world. We address this problem with a domain-adaptation training pipeline consisting of three major components: 1) learned intensity rendering, 2) geodesic correlation alignment, and 3) progressive domain calibration. When trained on real data, our new model exhibits segmentation accuracy improvements of 6.0-8.6% over the original SqueezeSeg. When training our new model on synthetic data using the proposed domain adaptation pipeline, we nearly double test accuracy on real-world data, from 29.0% to 57.4%. Our source code and synthetic dataset will be open-sourced.
  
  GitHub Repo: https://github.com/xuanyuzhou98/SqueezeSegV2
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="http://zju-capg.org/unsupervised_domain_adaptation/main.pdf">
  Cross-Sensor Deep Domain Adaptation for LiDAR Detection and Segmentation
  </a>," <i>IV</i>
</summary>
<blockquote><p align="justify">
  by <i>Christoph B. Rist, Markus Enzweiler, and Dariu M. Gavrila</i>
  
  A considerable amount of annotated training data is necessary to achieve state-of-the-art performance in perception tasks using point clouds. Unlike RGB-images, LiDAR point clouds captured with different sensors or varied mounting positions exhibit a significant shift in their input data distribution. This can impede transfer of trained feature extractors between datasets as it degrades performance vastly. We analyze the transferability of point cloud features between two different LiDAR sensor set-ups (32 and 64 vertical scanning planes with different geometry). We propose a supervised training methodology to learn transferable features in a pre-training step on LiDAR datasets that are heterogeneous in their data and label domains. In extensive experiments on object detection and semantic segmentation in a multi-task setup we analyze the performance of our network architecture under the impact of a change in the input data domain. We show that our pre-training approach effectively increases performance for both target tasks at once without having an actual multi-task dataset available for pre-training.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="http://zju-capg.org/unsupervised_domain_adaptation/main.pdf">
  Unsupervised Domain Adaptation for 3D Human Pose Estimation
  </a>," <i>ACM MM</i>
</summary>
<blockquote><p align="justify">
  by <i>Xiheng Zhang, Yongkang Wong, Mohan S. Kankanhalli, and Weidong Geng</i>
  
  Training an accurate 3D human pose estimator often requires a large amount of 3D ground-truth data which is inefficient and costly to collect. Previous methods have either resorted to weakly supervised methods to reduce the demand of ground-truth data for training, or using synthetically-generated but photo-realistic samples to enlarge the training data pool. Nevertheless, the former methods mainly require either additional supervision, such as unpaired 3D ground-truth data, or the camera parameters in multiview settings. On the other hand, the latter methods require accurately textured models, illumination configurations and background which need careful engineering. To address these problems, we propose a domain adaptation framework with unsupervised knowledge transfer, which aims at leveraging the knowledge in multi-modality data of the easy-to-get synthetic depth datasets to better train a pose estimator on the real-world datasets. Specifically, the framework first trains two pose estimators on synthetically-generated depth images and human body segmentation masks with full supervision, while jointly learning a human body segmentation module from the predicted 2D poses. Subsequently, the learned pose estimator and the segmentation module are applied to the real-world dataset to unsupervisedly learn a new RGB image based 2D/3D human pose estimator. Here, the knowledge encoded in the supervised learning modules are used to regularize a pose estimator without ground-truth annotations. Comprehensive experiments demonstrate significant improvements over weakly supervised methods when no ground-truth annotations are available. Further experiments with ground-truth annotations show that the proposed framework can outperform state-of-the-art fully supervised methods. In addition, we conducted ablation studies to examine the impact of each loss term, as well as with different amount of supervisions signal.
  
</p></blockquote>
</details>


#### 2018
<details>
<summary>
  "<a href="https://arxiv.org/abs/1712.05765">
  Unsupervised Domain Adaptation for 3D Keypoint Estimation via View Consistency
  </a>," <i>ECCV</i>
</summary>
<blockquote><p align="justify">
  by <i>Xingyi Zhou, Arjun Karpur, Chuang Gan, Linjie Luo, and Qixing Huang</i>
  
  In this paper, we introduce a novel unsupervised domain adaptation technique for the task of 3D keypoint prediction from a single depth scan or image. Our key idea is to utilize the fact that predictions from different views of the same or similar objects should be consistent with each other. Such view consistency can provide effective regularization for keypoint prediction on unlabeled instances. In addition, we introduce a geometric alignment term to regularize predictions in the target domain. The resulting loss function can be effectively optimized via alternating minimization. We demonstrate the effectiveness of our approach on real datasets and present experimental results showing that our approach is superior to state-of-the-art general-purpose domain adaptation techniques.
  
  GitHub Repo: https://github.com/xingyizhou/3DKeypoints-DA
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/1712.05765">
  Semi-Supervised Adversarial Learning to Generate Photorealistic Face Images of New Identities from 3D Morphable Model
  </a>," <i>ECCV</i>
</summary>
<blockquote><p align="justify">
  by <i>Baris Gecer, Binod Bhattarai, Josef Kittler, and Tae-Kyun Kim</i>
  
  We propose a novel end-to-end semi-supervised adversarial framework to generate photorealistic face images of new identities with a wide range of expressions, poses, and illuminations conditioned by synthetic images sampled from a 3D morphable model. Previous adversarial style-transfer methods either supervise their networks with a large volume of paired data or train highly under-constrained two-way generative networks in an unsupervised fashion. We propose a semi-supervised adversarial learning framework to constrain the two-way networks by a small number of paired real and synthetic images, along with a large volume of unpaired data. A set-based loss is also proposed to preserve identity coherence of generated images. Qualitative results show that generated face images of new identities contain pose, lighting and expression diversity. They are also highly constrained by the synthetic input images while adding photorealism and retaining identity information. We combine face images generated by the proposed method with a real data set to train face recognition algorithms and evaluate the model quantitatively on two challenging data sets: LFW and IJB-A. The generated images by our framework consistently improve the performance of deep face recognition networks trained with the Oxford VGG Face dataset, and achieve comparable results to the state-of-the-art.

</p></blockquote>
</details>



## Journal

#### 2021
<details>
<summary>
  "<a href="https://ieeexplore.ieee.org/abstract/document/9483674">
  Cross-Dataset Point Cloud Recognition Using Deep-Shallow Domain Adaptation Network
  </a>," <i>IEEE Transactions on Image Processing</i>
</summary>
<blockquote><p align="justify">
  by <i>Feiyu Wang, Wen Li, and Dong Xu</i>
  
  In this work, we propose a novel two-view domain adaptation network named Deep-Shallow Domain Adaptation Network (DSDAN) for 3D point cloud recognition. Different from the traditional 2D image recognition task, the valuable texture information is often absent in point cloud data, making point cloud recognition a challenging task, especially in the cross-dataset scenario where the training and test data exhibit a considerable distribution mismatch. In our DSDAN method, we tackle the challenging cross-dataset 3D point cloud recognition task from two aspects. On one hand, we propose a two-view learning framework, such that we can effectively leverage multiple feature representations to improve the recognition performance. To this end, we propose a simple and efficient Bag-of-Points feature method, as a complementary view to the deep representation. Moreover, we also propose a cross view consistency loss to boost the two-view learning framework. On the other hand, we further propose a two-level adaptation strategy to effectively address the domain distribution mismatch issue. Specifically, we apply a feature-level distribution alignment module for each view, and also propose an instance-level adaptation approach to select highly confident pseudo-labeled target samples for adapting the model to the target domain, based on which a co-training scheme is used to integrate the learning and adaptation process on the two views. Extensive experiments on the benchmark dataset show that our newly proposed DSDAN method outperforms the existing state-of-the-art methods for the cross-dataset point cloud recognition task.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2103.14198">
  Unsupervised Subcategory Domain Adaptive Network for 3D Object Detection in LiDAR
  </a>," <i>Electronics</i>
</summary>
<blockquote><p align="justify">
  by <i>Zhiyu Wang, Li Wang, Liang Xiao, and Bin Dai</i>
  
  Three-dimensional object detection based on the LiDAR point cloud plays an important role in autonomous driving. The point cloud distribution of the object varies greatly at different distances, observation angles, and occlusion levels. Besides, different types of LiDARs have different settings of projection angles, thus producing an entirely different point cloud distribution. Pre-trained models on the dataset with annotations may degrade on other datasets. In this paper, we propose a method for object detection using an unsupervised adaptive network, which does not require additional annotation data of the target domain. Our object detection adaptive network consists of a general object detection network, a global feature adaptation network, and a special subcategory instance adaptation network. We divide the source domain data into different subcategories and use a multi-label discriminator to assign labels dynamically to the target domain data. We evaluated our approach on the KITTI object benchmark and proved that the proposed unsupervised adaptive method could achieve a remarkable improvement in the adaptation capabilities.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2103.14198">
  Unsupervised Learning of Lidar Features for Use ina Probabilistic Trajectory Estimator
  </a>," <i>IEEE Robotics and Automation Letters</i>
</summary>
<blockquote><p align="justify">
  by <i>David J. Yoon, Haowei Zhang, Mona Gridseth, Hugues Thomas, and Timothy D. Barfoot</i>
  
  We present unsupervised parameter learning in a Gaussian variational inference setting that combines classic trajectory estimation for mobile robots with deep learning for rich sensor data, all under a single learning objective. The framework is an extension of an existing system identification method that optimizes for the observed data likelihood, which we improve with modern advances in batch trajectory estimation and deep learning. Though the framework is general to any form of parameter learning and sensor modality, we demonstrate application to feature and uncertainty learning with a deep network for 3D lidar odometry. Our framework learns from only the on-board lidar data, and does not require any form of groundtruth supervision. We demonstrate that our lidar odometry performs better than existing methods that learn the full estimator with a deep network, and comparable to state-of-the-art ICP-based methods on the KITTI odometry dataset. We additionally show results on lidar data from the Oxford RobotCar dataset.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://www.sciencedirect.com/science/article/abs/pii/S0924271621001131">
  Adversarial Unsupervised Domain Adaptation for 3D Semantic Segmentation with Multi-Modal Learning
  </a>," <i>ISPRS Journal of Photogrammetry and Remote Sensing</i>
</summary>
<blockquote><p align="justify">
  by <i>Wei Liu, Zhiming Luo, Yuanzheng Cai, Ying Yu, Yang Ke, José Marcato Junior, Wesley Nunes Gonçalves, and Jonathan Li</i>
  
  Semantic segmentation in 3D point-clouds plays an essential role in various applications, such as autonomous driving, robot control, and mapping. In general, a segmentation model trained on one source domain suffers a severe decline in performance when applied to a different target domain due to the cross-domain discrepancy. Various Unsupervised Domain Adaptation (UDA) approaches have been proposed to tackle this issue. However, most are only for uni-modal data and do not explore how to learn from the multi-modality data containing 2D images and 3D point clouds. We propose an Adversarial Unsupervised Domain Adaptation (AUDA) based 3D semantic segmentation framework for achieving this goal. The proposed AUDA can leverage the complementary information between 2D images and 3D point clouds by cross-modal learning and adversarial learning. On the other hand, there is a highly imbalanced data distribution in real scenarios. We further develop a simple and effective threshold-moving technique during the final inference stage to mitigate this issue. Finally, we conduct experiments on three unsupervised domain adaptation scenarios, ie., Country-to-Country (USA →Singapore), Day-to-Night, and Dataset-to-Dataset (A2D2 →SemanticKITTI). The experimental results demonstrate the effectiveness of proposed method that can significantly improve segmentation performance for rare classes. Code and trained models are available at https://github.com/weiliu-ai/auda.
  
  GitHub Repo: https://github.com/weiliu-ai/auda
  
</p></blockquote>
</details>


#### 2020
<details>
<summary>
  "<a href="https://www.sciencedirect.com/science/article/abs/pii/S0924271620302744">
  Unsupervised Scene Adaptation for Semantic Segmentation of Urban Mobile Laser Scanning Point Clouds
  </a>," <i>ISPRS Journal of Photogrammetry and Remote Sensing</i>
</summary>
<blockquote><p align="justify">
  by <i>Haifeng Luo, Kourosh Khoshelham, Lina Fang, and Chongcheng Chen</i>
  
  Semantic segmentation is a fundamental task in understanding urban mobile laser scanning (MLS) point clouds. Recently, deep learning-based methods have become prominent for semantic segmentation of MLS point clouds, and many recent works have achieved state-of-the-art performance on open benchmarks. However, due to differences of objects across different scenes such as different height of buildings and different forms of the same road-side objects, the existing open benchmarks (namely source scenes) are often significantly different from the actual application datasets (namely target scenes). This results in underperformance of semantic segmentation networks trained using source scenes when applied to target scenes. In this paper, we propose a novel method to perform unsupervised scene adaptation for semantic segmentation of urban MLS point clouds. Firstly, we show the scene transfer phenomena in urban MLS point clouds. Then, we propose a new pointwise attentive transformation module (PW-ATM) to adaptively perform the data alignment. Next, a maximum classifier discrepancy-based (MCD-based) adversarial learning framework is adopted to further achieve feature alignment. Finally, an end-to-end alignment deep network architecture is designed for the unsupervised scene adaptation semantic segmentation of urban MLS point clouds. To experimentally evaluate the performance of our proposed approach, two large-scale labeled source scenes and two different target scenes were used for the training. Moreover, four actual application scenes are used for the testing. The experimental results indicated that our approach can effectively achieve scene adaptation for semantic segmentation of urban MLS point clouds.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://www.sciencedirect.com/science/article/pii/S0924271620301301">
  A Multiclass TrAdaBoost Transfer Learning Algorithm for the Classification of Mobile LiDAR Data
  </a>," <i>ISPRS Journal of Photogrammetry and Remote Sensing</i>
</summary>
<blockquote><p align="justify">
  by <i>Hanxian He, Kourosh Khoshelham, and Clive Fraser</i>
  
  A major challenge in the application of state-of-the-art deep learning methods to the classification of mobile lidar data is the lack of sufficient training samples for different object categories. The transfer learning technique based on pre-trained networks, which is widely used in deep learning for image classification, is not directly applicable to point clouds, because pre-trained networks trained by a large number of samples from multiple sources are not available. To solve this problem, we design a framework incorporating a state-of-the-art deep learning network, i.e. VoxNet, and propose an extended Multiclass TrAdaBoost algorithm, which can be trained with complementary training samples from other source datasets to improve the classification accuracy in the target domain. In this framework, we first train the VoxNet model with the combined dataset and extract the feature vectors from the fully connected layer, and then use these to train the Multiclass TrAdaBoost. Experimental results show that the proposed method achieves both improvement in the overall accuracy and a more balanced performance in each category.
  
</p></blockquote>
</details>



## arXiv

#### 2021
<details>
<summary>
  "<a href="https://arxiv.org/abs/2111.15242">
  ConDA: Unsupervised Domain Adaptation for LiDAR Segmentation via Regularized Domain Concatenation
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Lingdong Kong, Niamul Quader, and Venice Erin Liong</i>
  
  Transferring knowledge learned from the labeled source domain to the raw target domain for unsupervised domain adaptation (UDA) is essential to the scalable deployment of an autonomous driving system. State-of-the-art approaches in UDA often employ a key concept: utilize joint supervision signals from both the source domain (with ground-truth) and the target domain (with pseudo-labels) for self-training. In this work, we improve and extend on this aspect. We present ConDA, a concatenation-based domain adaptation framework for LiDAR semantic segmentation that: (1) constructs an intermediate domain consisting of fine-grained interchange signals from both source and target domains without destabilizing the semantic coherency of objects and background around the ego-vehicle; and (2) utilizes the intermediate domain for self-training. Additionally, to improve both the network training on the source domain and self-training on the intermediate domain, we propose an anti-aliasing regularizer and an entropy aggregator to reduce the detrimental effects of aliasing artifacts and noisy target predictions. Through extensive experiments, we demonstrate that ConDA is significantly more effective in mitigating the domain gap compared to prior arts.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2107.09783">
  Unsupervised Domain Adaptation in LiDAR Semantic Segmentation with Self-Supervision and Gated Adapters
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Mrigank Rochan, Shubhra Aich, Eduardo R. Corral-Soto, Amir Nabatchian, and Bingbing Liu</i>
  
  In this paper, we focus on a less explored, but more realistic and complex problem of domain adaptation in LiDAR semantic segmentation. There is a significant drop in performance of an existing segmentation model when training (source domain) and testing (target domain) data originate from different LiDAR sensors. To overcome this shortcoming, we propose an unsupervised domain adaptation framework that leverages unlabeled target domain data for self-supervision, coupled with an unpaired mask transfer strategy to mitigate the impact of domain shifts. Furthermore, we introduce gated adapter modules with a small number of parameters into the network to account for target domain-specific information. Experiments adapting from both real-to-real and synthetic-to-real LiDAR semantic segmentation benchmarks demonstrate the significant improvement over prior arts.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2107.05399">
  SynLiDAR: Learning From Synthetic LiDAR Sequential Point Cloud for Semantic Segmentation
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Aoran Xiao, Jiaxing Huang, Dayan Guan, Fangneng Zhan, and Shijian Lu</i>
  
  Transfer learning from synthetic to real data has been proved an effective way of mitigating data annotation constraints in various computer vision tasks. However, the developments focused on 2D images but lag far behind for 3D point clouds due to the lack of large-scale high-quality synthetic point cloud data and effective transfer methods. We address this issue by collecting SynLiDAR, a synthetic LiDAR point cloud dataset that contains large-scale point-wise annotated point cloud with accurate geometric shapes and comprehensive semantic classes, and designing PCT-Net, a point cloud translation network that aims to narrow down the gap with real-world point cloud data. For SynLiDAR, we leverage graphic tools and professionals who construct multiple realistic virtual environments with rich scene types and layouts where annotated LiDAR points can be generated automatically. On top of that, PCT-Net disentangles synthetic-to-real gaps into an appearance component and a sparsity component and translates SynLiDAR by aligning the two components with real-world data separately. Extensive experiments over multiple data augmentation and semi-supervised semantic segmentation tasks show very positive outcomes - including SynLiDAR can either train better models or reduce real-world annotated data without sacrificing performance, and PCT-Net translated data further improve model performance consistently.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2106.11239">
  Domain and Modality Gaps for LiDAR-Based Person Detection on Mobile Robots
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Dan Jia, Alexander Hermans, and Bastian Leibe
</i>
  
  Person detection is a crucial task for mobile robots navigating in human-populated environments and LiDAR sensors are promising for this task, given their accurate depth measurements and large field of view. This paper studies existing LiDAR-based person detectors with a particular focus on mobile robot scenarios (e.g. service robot or social robot), where persons are observed more frequently and in much closer ranges, compared to the driving scenarios. We conduct a series of experiments, using the recently released JackRabbot dataset and the state-of-the-art detectors based on 3D or 2D LiDAR sensors (CenterPoint and DR-SPAAM respectively). These experiments revolve around the domain gap between driving and mobile robot scenarios, as well as the modality gap between 3D and 2D LiDAR sensors. For the domain gap, we aim to understand if detectors pretrained on driving datasets can achieve good performance on the mobile robot scenarios, for which there are currently no trained models readily available. For the modality gap, we compare detectors that use 3D or 2D LiDAR, from various aspects, including performance, runtime, localization accuracy, robustness to range and crowdedness. The results from our experiments provide practical insights into LiDAR-based person detection and facilitate informed decisions for relevant mobile robot designs and applications.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2105.12774">
  DSLR: Dynamic to Static LiDAR Scan Reconstruction Using Adversarially Trained Autoencoder
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Prashant Kumar, Sabyasachi Sahoo, Vanshil Shah, Vineetha Kondameedi, Abhinav Jain, Akshaj Verma, Chiranjib Bhattacharyya, and Vinay Viswanathan
</i>
  
  Accurate reconstruction of static environments from LiDAR scans of scenes containing dynamic objects, which we refer to as Dynamic to Static Translation (DST), is an important area of research in Autonomous Navigation. This problem has been recently explored for visual SLAM, but to the best of our knowledge no work has been attempted to address DST for LiDAR scans. The problem is of critical importance due to wide-spread adoption of LiDAR in Autonomous Vehicles. We show that state-of the art methods developed for the visual domain when adapted for LiDAR scans perform poorly. We develop DSLR, a deep generative model which learns a mapping between dynamic scan to its static counterpart through an adversarially trained autoencoder. Our model yields the first solution for DST on LiDAR that generates static scans without using explicit segmentation labels. DSLR cannot always be applied to real world data due to lack of paired dynamic-static scans. Using Unsupervised Domain Adaptation, we propose DSLR-UDA for transfer to real world data and experimentally show that this performs well in real world settings. Additionally, if segmentation information is available, we extend DSLR to DSLR-Seg to further improve the reconstruction quality. DSLR gives the state of the art performance on simulated and real-world datasets and also shows at least 4x improvement. We show that DSLR, unlike the existing baselines, is a practically viable model with its reconstruction quality within the tolerable limits for tasks pertaining to autonomous navigation like SLAM in dynamic environments.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2104.11021">
  Cycle and Semantic Consistent Adversarial Domain Adaptation for Reducing Simulation-to-Real Domain Shift in LiDAR Bird's Eye View
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Alejandro Barrera, Jorge Beltrán, Carlos Guindel, Jose Antonio Iglesias, and Fernando García
</i>
  
  The performance of object detection methods based on LiDAR information is heavily impacted by the availability of training data, usually limited to certain laser devices. As a result, the use of synthetic data is becoming popular when training neural network models, as both sensor specifications and driving scenarios can be generated ad-hoc. However, bridging the gap between virtual and real environments is still an open challenge, as current simulators cannot completely mimic real LiDAR operation. To tackle this issue, domain adaptation strategies are usually applied, obtaining remarkable results on vehicle detection when applied to range view (RV) and bird's eye view (BEV) projections while failing for smaller road agents. In this paper, we present a BEV domain adaptation method based on CycleGAN that uses prior semantic classification in order to preserve the information of small objects of interest during the domain adaptation process. The quality of the generated BEVs has been evaluated using a state-of-the-art 3D object detection framework at KITTI 3D Object Detection Benchmark. The obtained results show the advantages of the proposed method over the existing alternatives.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2104.05164">
  A Learnable Self-Supervised Task for Unsupervised Domain Adaptation on Point Clouds
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Xiaoyuan Luo, Shaolei Liu, Kexue Fu, Manning Wang, and Zhijian Song
</i>
  
  Deep neural networks have achieved promising performance in supervised point cloud applications, but manual annotation is extremely expensive and time-consuming in supervised learning schemes. Unsupervised domain adaptation (UDA) addresses this problem by training a model with only labeled data in the source domain but making the model generalize well in the target domain. Existing studies show that self-supervised learning using both source and target domain data can help improve the adaptability of trained models, but they all rely on hand-crafted designs of the self-supervised tasks. In this paper, we propose a learnable self-supervised task and integrate it into a self-supervision-based point cloud UDA architecture. Specifically, we propose a learnable nonlinear transformation that transforms a part of a point cloud to generate abundant and complicated point clouds while retaining the original semantic information, and the proposed self-supervised task is to reconstruct the original point cloud from the transformed ones. In the UDA architecture, an encoder is shared between the networks for the self-supervised task and the main task of point cloud classification or segmentation, so that the encoder can be trained to extract features suitable for both the source and the target domain data. Experiments on PointDA-10 and PointSegDA datasets show that the proposed method achieves new state-of-the-art performance on both classification and segmentation tasks of point cloud UDA. Code will be made publicly available.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2103.14198">
  Exploiting Playbacks in Unsupervised Domain Adaptation for 3D Object Detection
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Yurong You, Carlos Andres Diaz-Ruiz, Yan Wang, Wei-Lun Chao, Bharath Hariharan, Mark Campbell, and Kilian Q Weinberger</i>
  
  Self-driving cars must detect other vehicles and pedestrians in 3D to plan safe routes and avoid collisions. State-of-the-art 3D object detectors, based on deep learning, have shown promising accuracy but are prone to over-fit to domain idiosyncrasies, making them fail in new environments -- a serious problem if autonomous vehicles are meant to operate freely. In this paper, we propose a novel learning approach that drastically reduces this gap by fine-tuning the detector on pseudo-labels in the target domain, which our method generates while the vehicle is parked, based on replays of previously recorded driving sequences. In these replays, objects are tracked over time, and detections are interpolated and extrapolated -- crucially, leveraging future information to catch hard cases. We show, on five autonomous driving datasets, that fine-tuning the object detector on these pseudo-labels substantially reduces the domain gap to new driving environments, yielding drastic improvements in accuracy and detection reliability.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2103.02093">
  Pseudo-labeling for Scalable 3D Object Detection
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Benjamin Caine, Rebecca Roelofs, Vijay Vasudevan, Jiquan Ngiam, Yuning Chai, Zhifeng Chen, and Jonathon Shlens</i>
  
  To safely deploy autonomous vehicles, onboard perception systems must work reliably at high accuracy across a diverse set of environments and geographies. One of the most common techniques to improve the efficacy of such systems in new domains involves collecting large labeled datasets, but such datasets can be extremely costly to obtain, especially if each new deployment geography requires additional data with expensive 3D bounding box annotations. We demonstrate that pseudo-labeling for 3D object detection is an effective way to exploit less expensive and more widely available unlabeled data, and can lead to performance gains across various architectures, data augmentation strategies, and sizes of the labeled dataset. Overall, we show that better teacher models lead to better student models, and that we can distill expensive teachers into efficient, simple students.
Specifically, we demonstrate that pseudo-label-trained student models can outperform supervised models trained on 3-10 times the amount of labeled examples. Using PointPillars [24], a two-year-old architecture, as our student model, we are able to achieve state of the art accuracy simply by leveraging large quantities of pseudo-labeled data. Lastly, we show that these student models generalize better than supervised models to a new domain in which we only have unlabeled data, making pseudo-label training an effective form of unsupervised domain adaptation.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2102.07373">
  Generation For Adaption: A GAN-Based Approach for 3D Domain Adaption with Point Cloud Data
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Junxuan Huang, Junsong Yuan, and Chunming Qiao</i>
  
  Recent deep networks have achieved good performance on a variety of 3d points classification tasks. However, these models often face challenges in "wild tasks".There are considerable differences between the labeled training/source data collected by one Lidar and unseen test/target data collected by a different Lidar. Unsupervised domain adaptation (UDA) seeks to overcome such a problem without target domain labels.Instead of aligning features between source data and target data,we propose a method that use a Generative adversarial network to generate synthetic data from the source domain so that the output is close to the target domain.Experiments show that our approach performs better than other state-of-the-art UDA methods in three popular 3D object/scene datasets (i.e., ModelNet, ShapeNet and ScanNet) for cross-domain 3D objects classification.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2101.07253">
  Cross-modal Learning for Domain Adaptation in 3D Semantic Segmentation
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Maximilian Jaritz, Tuan-Hung Vu, Raoul de Charette, Émilie Wirbel, and Patrick Pérez
</i>
  
  Domain adaptation is an important task to enable learning when labels are scarce. While most works focus only on the image modality, there are many important multi-modal datasets. In order to leverage multi-modality for domain adaptation, we propose cross-modal learning, where we enforce consistency between the predictions of two modalities via mutual mimicking. We constrain our network to make correct predictions on labeled data and consistent predictions across modalities on unlabeled target-domain data. Experiments in unsupervised and semi-supervised domain adaptation settings prove the effectiveness of this novel domain adaptation strategy. Specifically, we evaluate on the task of 3D semantic segmentation using the image and point cloud modality. We leverage recent autonomous driving datasets to produce a wide variety of domain adaptation scenarios including changes in scene layout, lighting, sensor setup and weather, as well as the synthetic-to-real setup. Our method significantly improves over previous uni-modal adaptation baselines on all adaption scenarios. Code will be made available.
  
</p></blockquote>
</details>


#### 2020
<details>
<summary>
  "<a href="https://arxiv.org/abs/2012.05018">
  A Registration-Aided Domain Adaptation Network for 3D Point Cloud Based Place Recognition
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Zhijian Qiao, Hanjiang Hu, Weiang Shi, Siyuan Chen, Zhe Liu, and Hesheng Wang</i>
  
  In the field of large-scale SLAM for autonomous driving and mobile robotics, 3D point cloud based place recognition has aroused significant research interest due to its robustness to changing environments with drastic daytime and weather variance. However, it is time-consuming and effort-costly to obtain high-quality point cloud data for place recognition model training and ground truth for registration in the real world. To this end, a novel registration-aided 3D domain adaptation network for point cloud based place recognition is proposed. A structure-aware registration network is introduced to help to learn features with geometric information and a 6-DoFs pose between two point clouds with partial overlap can be estimated. The model is trained through a synthetic virtual LiDAR dataset through GTA-V with diverse weather and daytime conditions and domain adaptation is implemented to the real-world domain by aligning the global features. Our results outperform state-of-the-art 3D place recognition baselines or achieve comparable on the real-world Oxford RobotCar dataset with the visualization of registration on the virtual dataset.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2010.12239">
  Domain Adaptation in LiDAR Semantic Segmentation
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Inigo Alonso, Luis Riazuelo. Luis Montesano, and Ana C. Murillo</i>
  
  LiDAR semantic segmentation provides 3D semantic information about the environment, an essential cue for intelligent systems during their decision making processes. Deep neural networks are achieving state-of-the-art results on large public benchmarks on this task. Unfortunately, finding models that generalize well or adapt to additional domains, where data distribution is different, remains a major challenge. This work addresses the problem of unsupervised domain adaptation for LiDAR semantic segmentation models. Our approach combines novel ideas on top of the current state-of-the-art approaches and yields new state-of-the-art results. We propose simple but effective strategies to reduce the domain shift by aligning the data distribution on the input space. Besides, we propose a learning-based approach that aligns the distribution of the semantic classes of the target domain to the source domain. The presented ablation study shows how each part contributes to the final performance. Our strategy is shown to outperform previous approaches for domain adaptation with comparisons run on three different domains.
  
</p></blockquote>
</details>


<details>
<summary>
  "<a href="https://arxiv.org/abs/2003.01174">
  LiDARNet: A Boundary-Aware Domain Adaptation Model for Point Cloud Semantic Segmentation
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Peng Jiang and Srikanth Saripalli</i>
  
  We present a boundary-aware domain adaptation model for LiDAR scan full-scene semantic segmentation (LiDARNet). Our model can extract both the domain private features and the domain shared features with a two-branch structure. We embedded Gated-SCNN into the segmentor component of LiDARNet to learn boundary information while learning to predict full-scene semantic segmentation labels. Moreover, we further reduce the domain gap by inducing the model to learn a mapping between two domains using the domain shared and private features. Additionally, we introduce a new dataset (SemanticUSL\footnote{The access address of SemanticUSL:\url{this https URL}}) for domain adaptation for LiDAR point cloud semantic segmentation. The dataset has the same data format and ontology as SemanticKITTI. We conducted experiments on real-world datasets SemanticKITTI, SemanticPOSS, and SemanticUSL, which have differences in channel distributions, reflectivity distributions, diversity of scenes, and sensors setup. Using our approach, we can get a single projection-based LiDAR full-scene semantic segmentation model working on both domains. Our model can keep almost the same performance on the source domain after adaptation and get an 8\%-22\% mIoU performance increase in the target domain.
  
</p></blockquote>
</details>


#### 2019
<details>
<summary>
  "<a href="https://arxiv.org/abs/1911.10575">
  Unsupervised Neural Sensor Models for Synthetic LiDAR Data Augmentation
  </a>," <i>arXiv</i>
</summary>
<blockquote><p align="justify">
  by <i>Ahmad El Sallab, Ibrahim Sobh, Mohamed Zahran, and Mohamed Shawky</i>
  
  Data scarcity is a bottleneck to machine learning-based perception modules, usually tackled by augmenting real data with synthetic data from simulators. Realistic models of the vehicle perception sensors are hard to formulate in closed form, and at the same time, they require the existence of paired data to be learned. In this work, we propose two unsupervised neural sensor models based on unpaired domain translations with CycleGANs and Neural Style Transfer techniques. We employ CARLA as the simulation environment to obtain simulated LiDAR point clouds, together with their annotations for data augmentation, and we use KITTI dataset as the real LiDAR dataset from which we learn the realistic sensor model mapping. Moreover, we provide a framework for data augmentation and evaluation of the developed sensor models, through extrinsic object detection task evaluation using YOLO network adapted to provide oriented bounding boxes for LiDAR Bird-eye-View projected point clouds. Evaluation is performed on unseen real LiDAR frames from KITTI dataset, with different amounts of simulated data augmentation using the two proposed approaches, showing improvement of 6% mAP for the object detection task, in favor of the augmenting LiDAR point clouds adapted with the proposed neural sensor models over the raw simulated LiDAR.
  
</p></blockquote>
</details>




