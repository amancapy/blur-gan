# blur-gan
**GAN ensemble to fix blurry images**

What's fundamentally different here from the traditional implementation (you will notice that the code here derives largely from Tensorflow's official tutorial on GANs) is that instead of scaling L1 by a pre-set lambda, lambda is scaled gradually during training, which I have found results in a much lower tendency to collapse or to fail to converge even with very small batch sizes. Though this may not be ground-breaking, the models converged well to the task.

![image](https://github.com/amancapy/blur-gan/assets/111729660/b6ec0651-e608-4704-98af-93af8edb0577)
