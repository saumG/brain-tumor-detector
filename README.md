# brain-tumor-detector
Developing a Brian Tumor Detection app that uses deep learning to classify brain tumors


## What is a brain tumor?
Accumulation, mass, or growth of abnormal cells in the brain. The tumor can be benign or malignant. Malignant brain tumors are often rare, roughly 1-2% of all cancer types. 

## Overview 
The aim is to build a multi-class classification based convolutional neural network to classify 3 different types of brain tumors as well as normal cases (ones without tumors).
The categories are glioma, meningioma, pituitary, and no tumor. 

This project will use the Brain Tumor MRI Dataset from Kaggle.

### Glioma
Account for around 1/3 of all brain tumors. Gliomas orignate in the glial cells that surround the brain's neurons. A glioma can hinder brain function and potentially be life-threatening based on its location or growth. There are three types of glial cells that produce these tumors:

1. Astrocytomas, including astrocytoma, anaplastic astrocytoma and glioblastoma
2. Ependymomas, including anaplastic ependymoma, myxopapillary ependymoma and subependymoma
3. Oligodendrogliomas, including oligodendroglioma, anaplastic oligodendroglioma and anaplastic oligoastrocytoma

### Meningioma
A primary central nervous system tumor, begginning in the brain or the spinal cord. It forms on the three layers of membranes known as meninges. Meningiomas are slow-growing and around 90% are benign, often meaning no symptoms and they often require no immediate treatment. However, the growth of a benign tumor could result in serious complications. 


### Pituitary 
Forms in the pituitary gland which can cause irregular hormone levels in the body. They are mostly benign growths (adenomas) which do not spread to any other parts of the body. Although benign, they could still lead to major health problems due to their proximity with the brain. Pituitary cancers are very rare. 


## Image Pre-processing
1. Find the biggest contour
2. Find the extreme points 
3. Crop the image