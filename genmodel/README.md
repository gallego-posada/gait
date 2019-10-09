# MNIST generative model

Install `pylego` from the parent directory to run this code.

To reproduce the MNIST generative model results from the paper, use the following commands.

To compare with VAE:
```
python main.py --name experiment_name_here --batch_size 100 --layers 1
```

To compare with Sinkhorn divergence:
```
python main.py --name experiment_name_here --batch_size 200 --nonormal_latent --layers 2
```

To generate Fashion MNIST manifold:
```
python main.py --name experiment_name_here --data fmnist --batch_size 200 --layers 2
```

To generate Fashion MNIST samples:
```
python main.py --name experiment_name_here --data fmnist --batch_size 500 --layers 2 --z_size 100
```
