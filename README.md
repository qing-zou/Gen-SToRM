# Gen-SToRM
The source code for Dynamic imaging using a deep generative SToRM (Gen-SToRM) model

## Reference paper
Dynamic imaging using a deep generative SToRM (Gen-SToRM) model by Q. Zou, A. Ahmed, P. Nagpal, S. Kruger and M. Jacob in IEEE Transactions on Medical Imaging

Link: https://arxiv.org/pdf/2102.00034.pdf


### What this code do:
In the above paper, we propose a generative SToRM model for the reconstruction of free-breathing and ungated cardiac MRI. We assume that the images in the time series are some non-linear map of the points that are lying in the low dimensional latent space. We use a CNN to build the non-linear map as the structure of CNN can implicitly offer spitial regularization [1]. We also add two regularization terms to further constrain the solution: the network regularization and the latent vector regularization. In the paper, we also propose a progressive training-in-time approach to speed up the reconstruction process. Note that this proposed scheme needs only the highly undersampled k-t space data. So this scheme is unsupervised, meaning that no fully sampled training data are needed.

The code then did the jobs that are mentioned in the above paragraph.

### The idea of the work and the cost function:
The following figure shows the basic idea of the work.
![image](https://user-images.githubusercontent.com/36931917/114895176-e4ab6f00-9dd4-11eb-91c7-5dc3ad214f8b.png)

We feed a set of latent vectors into the generator (G_theta) and the generator is then able to generate the images in the time series. We then perform the non-uniform Fourier Transform (NUFFT) to get the k-t space data of the generated images and compare them with the collected k-t space data. Based on which, we have the following cost function and the code solves for the cost function:
![image](https://user-images.githubusercontent.com/36931917/114894290-20920480-9dd4-11eb-93ce-58ef3e4a33ed.png)

A_i here means the non-uniform Fourier transform. b is the acquired undersampled k-t space data. In this work, we use the golden angle spiral trajectories to acquired the k-t space data.

### The progressive training-in-time approach
Since the data in this work are collected using the spiral trajectories and each frame will have different spirals and hence we need different NUFFT operators for different frames. Therefore the computational complexity will be very high to directly solving the cost function. Hence we propose a progressive training-in-time approach to speed up the process. The following figure shows the idea of this approach. The detailed description of this approch can be found in the paper above.
![image](https://user-images.githubusercontent.com/36931917/114896036-99459080-9dd5-11eb-89c1-cec952c4015f.png)

### Main benefits of Gen-SToRM:
1. Unlike the traditional CNN based approach, Gen-SToRM does not need extensive fully sampled training data.
2. Gen-SToRM is able to estimate the phases directly from the undersampled k-t space data and hence no navigator is needed.
3. The memory footprint of Gen-SToRM is dependent on the number of network paprameters and the very low dimensional latenet vector and we do not need to store all the images in the time series comparing to the traditional manifold methods for free-breathing and ungated cardiac MRI.

### Dependencies:
The code relies on the NUFFT operator and we used the torchkbnufft repository to have the NUFFT operator. But note that the torchkbnufft version used in this repository is not the most recent one. You can download the .zip file for the torchkbnufft under this repository. To use it, you can just unzip it and put it in your working folder.
Note that the torchkbnufft version that we used in this repository does not support complex number. However, the most recent version (https://github.com/mmuckley/torchkbnufft) is able to support complex number. But you need to make sure that your PyTorch version is above 1.5.0 so that complex type data can be supported.

### Dataset:
We have released one dataset used in this paper. You can download the full dataset from the following link:

https://drive.google.com/file/d/1whgyHXcuY5JKCoa5-EZ9PzFjJt43rPq1/view?usp=sharing

This dataset contains 8 image slices. The k-t space data, the kspace trajectories and the density compensation functions are included in the file.

The dataset is **NOT public** and has copyright. If you want to use this dataset in your research, please contact us at zou-qing@uiowa.edu

### Before running the code: A word about the generator
There are several commnon ways to bulid the generator, even though all named as CNN generator. In this work, we tried two ways for buliding the generator:
**1.** Using ConvTranspose2d
**2.** Using Conv2d + Upsampling

Both the two ways have their own advantages and disadvantages.

For buliding the generator using ConvTranspose2d, one common complain would be the checkerboard artifacts (https://distill.pub/2016/deconv-checkerboard/). There are actually ways to avoid the checkerboard artifacts using ConvTranspose2d, the simplest ways are trying different initializations and/or running more epoches. But the advantage of ConvTranspose2d is that it can produce good results quickly and it needs less GPU memory. For example, in our setting, using ConvTranspose2d just needs 16G GPU to get good results.

For buliding the generator using Conv2d, it can completely overcome the checkerboard artifacts. However, it may take longer time to get good results. Also, the GPU requirement is higher, we have to use a 32G GPU card to get comparable results as using ConvTranspose2d. Also, the "mode" used in the upsampling layer will affect the results a lot. In this work, we used the nearest neighbor interpolation for upsampling. One can also try to set "mode=bilinear" to see the results or other interpolation methods to do the upsampling. The results are sensitive to the interpolation methods. One may try different interpolation methods to see which one will give you the best results. When one tries the linearly interpolting, don't forget to set "lign_corners = True".

### Run the code:
For the ConvTranspose2d case, run the code, we just need to run the main file: gen_storm_main.py

The code is written in PyTorch. Also, one needs to make sure that a working GPU is installed. The requirement for the GPU is 16GB. If you have a smaller GPU, you can reduce the number of the frames being processed to fit the GPU.

For the Conv2d + Upsampling case, to run the algorithm, we just need to run the main file: gen_storm_upsamp.py

In this setting, one needs to make sure that a working GPU is installed. The requirement for the GPU is 32GB. If you have a smaller GPU, you can reduce the number of the frames being processed to fit the GPU.

### Files description:
**dataAndOperators.py** is used to prepare the data and define some necessary operator. The data in this work is acquired using the spiral trajectories and we bin every 6 spirals to get one frame. We also did the coil combination and coil sensitivity map estimation in this file. If your data is acquired in a different acquisition scheme, you can change this file to fit your data acquisition scheme. The necessary operators such us NUFFT are defined in this file as well.


**generator.py** is used the build the generator using the CNN (ConvTranspose2d).

**generator_upsamp.py** is used the build the generator using the CNN (Conv2d + Upsampling).

**latentVariable.py** is used to process the latent vectors.

**optimize_generator.py** is used to solve the cost function. We use the Adam optimization to solve the cost function in this work.

The **esprit folder** contains the ESPRIT function which is used to estimate the coil sensitivity map.

We provided one examplar reconstruction for one slice using the above dataset in the **example folder**.

### Final notes:
The code is provided to support reproducible research. 

The dataset is **NOT public** and has copyright. If you want to use this dataset in your research, please contact us at zou-qing@uiowa.edu

If the code is giving any error or some files are missing then you may open an issue or directly email me at zou-qing@uiowa.edu
