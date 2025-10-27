
# iBRF

This repository contains the code and supplementary files for the proposed iBRF (Improved Balanced Random Forest) classifier. 

## Dependencies

This project uses the following libraries:

[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.1-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![imbalanced-learn](https://img.shields.io/badge/imblearn-0.11.0-blue?logo=python)](https://imbalanced-learn.org/)



# Paper

The paper on this work has been published in IEEE Xplore.

Title - iBRF: Improved Balanced Random Forest Classifier

DOI: https://doi.org/10.23919/FRUCT61870.2024.10516372
## Synopsis

This paper proposes a modification to the original BRF classifier for enhanced prediction performance. 

In the original algorithm, the Random Undersampling (RUS) technique is utilized to balance the bootstrap samples. However, randomly eliminating too many samples from the data leads to significant data loss, resulting in a major decline in performance.

This paper proposed a novel sampling approach that, when incorporated into the framework of the RF classifier, achieves better and more generalized prediction performance. The proposed algorithm outperforms the original BRF classifier. 


Original BRF classifier implementation in the imblearn library: https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html
## Screenshots

![App Screenshot](https://github.com/newaz-aa/iBRF/blob/main/ibrf_4.png)


## Note
This repository currently contains the earliest version of the proposed iBRF framework. A more advanced version is currently under development.


LemaÃŽtre, G., Nogueira, F. and Aridas, C.K., 2017. Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets in machine learning. Journal of machine learning research, 18(17), pp.1-5. 

## BibTex Citation

```
@INPROCEEDINGS{10516372,
  author={Newaz, Asif and Mohosheu, Md. Salman and Noman, Md. Abdullah Al and Jabid, Taskeed},
  booktitle={2024 35th Conference of Open Innovations Association (FRUCT)}, 
  title={iBRF: Improved Balanced Random Forest Classifier}, 
  year={2024},
  volume={},
  number={},
  pages={501-508},
  keywords={Technological innovation;Data preprocessing;Benchmark testing;Prediction algorithms;Data models;Classification algorithms;Ensemble learning},
  doi={10.23919/FRUCT61870.2024.10516372}}
```
