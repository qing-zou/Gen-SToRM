# Gen-SToRM
The source code for Dynamic imaging using a deep generative SToRM (Gen-SToRM) model

## Reference papers
Dynamic imaging using a deep generative SToRM (Gen-SToRM) model by Q. Zou, A. Ahmed, P. Nagpal, S. Kruger and M. Jacob in IEEE Transactions on Medical Imaging

Link: https://arxiv.org/pdf/2102.00034.pdf


#### What this code do:
In the above paper, we propose a generative SToRM model for the reconstruction of free-breathing and ungated cardiac MRI. We assume that the images in the time series are some non-linear map of the points that are lying in the low dimensional latent space. We use a CNN to build the non-linear map as the structure of CNN can implicitly offer spitial regularization [1]. We also add two regularization terms to further constrain the solution: the network regularization and the latent vector regularization. In the paper, we also propose a progressive training-in-time approach to speed up the reconstruction process. Note that this proposed scheme needs only the highly undersampled k-t space data. So this scheme is unsupervised, meaning that no fully sampled training data are needed.

The code then did the jobs that are mentioned in the above paragraph.


Main file: gen_storm_main.py
