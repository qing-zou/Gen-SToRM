# Gen-SToRM
The source code for Dynamic imaging using a deep generative SToRM (Gen-SToRM) model

## Reference papers
Dynamic imaging using a deep generative SToRM (Gen-SToRM) model by Q. Zou, A. Ahmed, P. Nagpal, S. Kruger and M. Jacob in IEEE Transactions on Medical Imaging

Link: https://arxiv.org/pdf/2102.00034.pdf


#### What this code do:
In the above paper, we propose a generative SToRM model for the reconstruction of free-breathing and ungated cardiac MRI. We assume that the images in the time series are some non-linear map of the points that are lying in the low dimensional latent space. We use a CNN to build the non-linear map as the structure of CNN can implicitly offer spitial regularization [1]. We also add two regularization terms to further constrain the solution: the network regularization and the latent vector regularization. In the paper, we also propose a progressive training-in-time approach to speed up the reconstruction process. Note that this proposed scheme needs only the highly undersampled k-t space data. So this scheme is unsupervised, meaning that no fully sampled training data are needed.

The code then did the jobs that are mentioned in the above paragraph.

#### The cost function:
The following figure shows the basic idea of the work.
![image](https://user-images.githubusercontent.com/36931917/114895176-e4ab6f00-9dd4-11eb-91c7-5dc3ad214f8b.png)

We feed a set of latent vectors into the generator (G_theta) and the generator is then able to generate the images in the time series. We then perform the non-uniform Fourier Transform (NUFFT) to get the k-t space data of the generated images and compare them with the collected k-t space data. Based on which, we have the following cost function and the code solves for the cost function:
![image](https://user-images.githubusercontent.com/36931917/114894290-20920480-9dd4-11eb-93ce-58ef3e4a33ed.png)

A_i here means the non-uniform Fourier transform. b is the acquired undersampled k-t space data. In this work, we use the golden angle spiral trajectories to acquired the k-t space data.


Main file: gen_storm_main.py
