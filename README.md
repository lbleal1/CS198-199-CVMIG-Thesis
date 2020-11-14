# Road Segmentation from Multi-Modal Imagery of Ilocos Norte using Mask-RCNN
## Fulfillment for Undergraduate Thesis (CS 198 and 199)
Project Duration: September 2019 - August 2020 

<img align="right" src="https://github.com/lbleal1/CS198-199-CVMIG-Thesis/blob/master/assets/cvmig_announcements.png" width="450">

**Abstract**

Road segmentation contributes significantly to different fields ranging from urban planning to disaster response and hazard mapping. Considered as a complex task due to the high visual variations in the data captured, this is usually manifested with photometric distortions by current survey technologies. International efforts such as DeepGlobe2018 in 2018 and SpaceNet2019 in 2019 have been introduced to utilize image processing and deep learning methods in solving this problem. However, area localization is still an issue in the existing models in that the introduction of a new dataset with different sensor specifications, capture parameters, and sampled area topographies have yielded low accuracies. As a supplementary research to the current systems for the analysis of Philippine geographic data, this paper focuses on the segmentation of roads from combinations of LiDAR and Google Satellite imagery of the Philippines using Mask R-CNN. We investigate different annotation schemes, augmentations, scale evaluations, anchor scales, dataset combinations, contrasts, data preprocessing methods, and detection backbones. Our model achieved a dice score of 74.07% and an IoU score of 65.61% in Ilocos Norte having mixed topography and was tested further in other regions having varying topographies.

[Presentation slides](https://github.com/lbleal1/CS198-199-CVMIG-Thesis/blob/master/assets/%5BCS%20199-Espino-Leal%5D%20Final%20Thesis%20Presentation.pdf) 
<br>
**Errata on slides:** <br>
Visualization on Bounding Boxes - the training set should only be the one visualized. This errata does not alter the resuts since the decisions regarding the anchor boxes were done using the training data only. 

Note:
This project uses [Matterport's implementation of Mask R-CNN](https://github.com/matterport/Mask_RCNN)
