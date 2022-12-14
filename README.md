# CptS 437 Machine Learning Course Project

## TABLE OF CONTENTS
- [CptS 437 Machine Learning Course Project](#cpts-437-machine-learning-course-project)
  - [TABLE OF CONTENTS](#table-of-contents)
  - [1. Course Information](#1-course-information)
    - [Course Prerequisites:](#course-prerequisites)
  - [2. Team Information](#2-team-information)
  - [3. Project Information](#3-project-information)
    - [Problem](#problem)
    - [Data](#data)
    - [Solution](#solution)
    - [Challenges & Findings](#challenges-and-findings)
  - [4. Links & References](#4links-and-references)

---

## 1. Course Information

- Semester: Fall 2023
- Professor: Dianne Cook, djcook@wsu.edu
    - Office Hour: [Tue Thu 10:30-11:00am](https://wsu.zoom.us/j/93838519178?pwd=cTJoSGtJVU9RL29FRlU2YjhEMHU3UT09)
- TA: Ramesh Sah, ramesh.sah@wsu.edu
    - Office Hour: [Mon Tue 2:30-04:00pm](https://wsu.zoom.us/j/8991996138 )

### Course Prerequisites:

- Linear Algebra
    - vector & matrix norms
    - linear independence

- Multivariable Calculus
    - derivatives
    - chain rule
    - gradient descent

- Probability $P$ & Statistics
    - discrete and continuous $P$ distributions
    - sum rule, product rule
    - marginal $P$ distributions
    - conditional $P$ distributions
    - joint $P$ distributions
    - independence & conditional independence
    - Bayes theorem
    - variance & co-variance
    - expectation


## 2. Team Information

- Charles Nguyen, charles.norden@wsu.edu
    - Major: Computer Science, B.Sci.
- Denise Tanumihardja, denise.tanumihardja@wsu.edu
    - Major: NeuroScience, B.Sci.


## 3. Project Information

### Problem

Use a neural network to perform a binary classfication on some fMRI data of some neurological condition. We want a predictive model that can distinguish between *healthy* and *diseased*.

### Data

- https://neurovault.org/collections/1015/

Description of the dataset: the data is directly resulted from the study on [Functional MRI of emotional memory in adolescent depression](https://www.sciencedirect.com/science/article/pii/S1878929315001322). The dataset contains fMRI of 56 MDD patients aged 11-17 years. These participants are matched against a group of 30 healthy control participants. The study in fact dropped a portion of the participants due to not meeting the study's criteria. However, there fMRI are still collected and collated into the dataset. In total there are about 84 distinct participant samples.


### Solution

We originally thought of building a convolutional network from scratch, but then quickly realized that it would require a tremendous amount of work beyond our team of two. Furthermore, while working on the data cleaning part, we realized that our data is a 3-dimensional volumetric object, and thus requires us to write convolution functions to beable to apply filters on the data. Realizing this, we researched for template models and found [one](https://keras.io/examples/vision/3D_image_classification/) using convolutional network for 3D image classification.

Adopting this template, for this project, we decided to make the target label of classification to be `unmeddep`, shorthand for *Unmedicated Depressed*. This is the condition the researchers were looking for in the participants. We separated the data into their own labels, primarily correlation by age, groupmean, and SMFQ.

The model can be triggered to start training via the command line with the appropriate keywords:

- `age`
- `groupmean`
- `smfq`

```sh
# at the root of the repo, run:
python ./src/main.py <one of three keywords>
```
### Challenges & Findings

We didn't realize how difficult it was to look for a dataset appropriate for our goal of neuroimage classication. The vast majority of datasets are very small, ranging between 7-50 images. Ours contains 295 images in total. However, the actual distinct participants number only to 84. We understand that this is the inherent reality of most data out there.

Due to the small data, we are aware that it is difficult to say if the model produced effective outcomes for classification. We observed that on `age` correlation data, the model converged after roughly 15-17 epochs. On the other hand, for `groupmean` and `smfq`, the model converged very quickly after only 10 epochs. This shows a contrary to the findings from the actual study, where there is a siginficant different between the MDD patients and the controls.

We also have a [Colab notebook](https://colab.research.google.com/drive/1H_pVmqi7TkZQ0G455Blea1DQl8SH5zkz?usp=sharing) avaialble with incomplete code. It took twice as long for the model to train on the Colab notebook so we decided to migrate the code to a local repo. All data and source code can be found in this repo.


## 4. Links & References

- [Functional MRI of emotional memory in adolescent depression](https://www.sciencedirect.com/science/article/pii/S1878929315001322)
- [3D Image Classication from CT Scans](https://keras.io/examples/vision/3D_image_classification/)
