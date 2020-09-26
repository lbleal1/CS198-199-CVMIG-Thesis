# Road Segmentation from Multi-Modal Imagery of Ilocos Norte using Mask-RCNN
## Fulfillment for Undergraduate Thesis (CS 198 and 199)
Project Duration: September 2019 - August 2020 

Advisers: Riza Rae Pineda and Kristofer delas Penas

Abstract:

Road segmentation contributes significantly to different fields ranging from urban planning to disaster response and hazard mapping. Considered as a complex task due to the high visual variations in the data captured, this is usually manifested with photometric distortions by current survey technologies. International efforts such as DeepGlobe2018 in 2018 and SpaceNet2019 in 2019 have been introduced to utilize image processing and deep learning methods in solving this problem. However, area localization is still an issue in the existing models in that the introduction of a new dataset with different sensor specifications, capture parameters, and sampled area topographies have yielded low accuracies. As a supplementary research to the current systems for the analysis of Philippine geographic data, this paper focuses on the segmentation of roads from combinations of LiDAR and Google Satellite imagery of the Philippines using Mask R-CNN. We investigate different annotation schemes, augmentations, scale evaluations, anchor scales, dataset combinations, contrasts, data preprocessing methods, and detection backbones. Our model achieved a dice score of 74.07% and an IoU score of 65.61% in Ilocos Norte having mixed topography and was tested further in other regions having varying topographies.

[Presentation slides:]() 

Note:
This project uses Matterport's implementation of Mask R-CNN
