# MNIST generative model

Install `pylego` from the parent directory to run this code.

To reproduce the MNIST generative model results from the paper, use the following commands.

To compare with VAE:
```
python main.py --name experiment_name_here --batch_size 100 --normal_latent --layers 1
```

To compare with Sinkhorn divergence:
```
python main.py --name experiment_name_here --batch_size 200 --nonormal_latent --layers 2
```

