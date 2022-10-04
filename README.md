# blur-gan
**"fixing" blurry images with a GAN**

results are decent-ish surprisingly.

01/10: gan training done, it seems the average of the geometric and arithmetic averages of three picked generations of the model is a good result. some outputs are still somehow blurry but I'm calling it a day. plan is to now apply this to larger images by splicing them into small squares then stitching them back together again post-gan.

02/10 new discovery, it seems some images in the dataset were inexplicably blurred much more heavily than intended, which might have hurt the training process but is also probably the reason why some results are arbitrarily subpar.
