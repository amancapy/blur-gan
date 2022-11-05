# blur-gan
**"GANs to fix blurry images**

what's fundamentally different here from the traditional implementation (you will notice that the code here derives largely from Tensorflow's official tutorial on GANs) is that instead of scaling L1 by a pre-set lambda, lambda is scaled gradually during training, which I have discovered results in a much lower tendency to collapse or to fail to converge.

timeline:

01/10: gan training done, it seems the average of the geometric and arithmetic averages of three picked generations of the model is a good result. ~~plan is to now apply this to larger images by splicing them into small squares then stitching them back together again post-gan.~~

02/10 discovery, it seems some images in the dataset were inexplicably blurred much more heavily than intended, which might have hurt the training process but is also probably the reason why some results are arbitrarily subpar.
