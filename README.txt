Here are some tips to run the ADMM-HFNet:

1. If you already have the datasets like CAVE, Harvard and University of Houston, you can just run the DataReader.py to prepare the data before training
    according to your data path. 

    For other datasets or different data settings (spatial resolution, spectral resolution, sampling rate and etc.), you need to
    add the dataset and modify the codes slightly.
    
    Considering the limited space, we only provide the well-trained model for CAVE and you can directly test the performance by execute the "test" function in ADMM-HFNet.py. 
    If you need other well-trained models, please contact us.

2. After preparing the data, you can run the ADMM-HFNet.py to start training. The only parameter you should to choose is the data index which represents
   the data you want to train.

3. When you finish the training process, a fusion model is obtained. You can execute the "test" function in ADMM-HFNet.py to verify the fusion performance.


If you want to use the ADMM-HFNet as one of your comparison methods, please kindly cite the article : Shen D, Liu J, Wu Z, et al. Admm-hfnet: A matrix decomposition based deep approach for
hyperspectral image fusion[J]. IEEE Transactions on Geoscience and Remote Sensing, 2021, early access, doi:10.1109/TGRS.2021.3112181.

If you have other problems about the article or the codes, please contact us by this email: sdb_2012@163.com. Thanks for your support.

