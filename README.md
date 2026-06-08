# Variational AutoEncoder
> _Architecture_
<center>
<img src='https://github.com/Martinmbiro/variational-autoencoder/raw/refs/heads/main/plots/vae_lean.PNG' alt='vae_sketch' width=800/>
</center>

Hello again 👋
+ A Variational AutoEncoder (VAE) is a foundational generative model introduced by [Diederik Kingma and Max Welling in 2013](https://arxiv.org/abs/1312.6114). It bridges the gap between traditional autoencoders (which compress data) and probabilistic generative models (which create new data). The goal of a VAE is to ouput new data in the form of random variations of the training data $x$
+ This project is a step-by-step implementation, training, and analysis of a vanilla VAE utilizing [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). Despite this being a simple project, I implement early-stopping as a regularization technique, model checkpointing, a learing rate scheduler to improving convergence, and warm-up to stabilize training during initial batches and prevent optimization shocks
+ Find the notes and annotated code in the Jupyter Notebook - `01. VAE from scratch.ipynb`. Feel free to open and explore in Google Colab

## Key concepts 📝
Here's a few things to note with regard to Variational AutoEncoders:
#### **1. Encoder output**
For a given input image, $x$, the encoder outputs two seperate vectors of equal length:
>+ **Mean vector**, `mu` ($μ$) - Center points where image features are most likely to cluster
>+ **Log-Variance vector**, `logvar` ($σ^2$) - Spread / uncertainty around the mean
> Mathematically, this distribution is assumed to be a Gaussian / Normal distribution

#### **2. The sampling step**
Once the encoder has defined this unique distribution for the image, we need to pass a latent vector $z$ to the decoder. To do this, we sample a random point $z$ from this newly defined distribution

#### **3. Reparameterization trick**
Since direct random sampling is non-differentiable (meaning backpropagation cannot pass through a random node), VAEs use the reparameterization trick: $z = μ + σ * ϵ$, where  
> + $μ$ - the mean vetor
> + $σ$ - the standard deviation vector
> + $ϵ$ - a random noise vector sampled from a normal disribution, $N(0, 1)$, meaning a normal distribution with a mean, $μ$ of $0$ and standard deviaion, $σ$ of $1$

#### **4. The Evidence Lower Bound (ELBO)**
The ELBO is the loss function that forces the VAE to balance the act of reconstructing the image as closely as possible to the input while keeping the latent space organized. Hence, the ELBO is made of two terms (Reconstruction term + Regularization term):
> + **Reconstruction term** - To ensure the output image is as close as possible to the input image
>> + Calculated using [Binary Cross Entropy (BCE)](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html) loss or [Mean Squared Error](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html) (MSE)
> + **Regularization term** - To ensure that the latent space distribution is as close as possible to a Gaussian distribution, $N(0, 1)$. Calculated by computing the **KL Divergence** between the encoder distribution and prior distribution which are both Gaussians. Hence, the math simplifies into the formula:
>> + $D_{KL} = -\frac{1}{2} (1 + log(σ^2) - μ^2 - σ^2)$

#### **4. Expressing** $σ^2$ **in terms of** `logvar`
Since we'll be predicting $log(σ^2)$ as `logvar` in our network, it makes sence to express $σ^2$ in the regularization term in terms of `logvar`  
Mathematically:
+ `logvar` = $log(σ^2)$
+ Introducing Exponential on both sides,
> + $e^{log(σ^2)} = e^{logvar}$
+ The $e$ and $log$ on the left cancel out, hence:
> + $σ^2 = e^{logvar}$

#### **5. Expressing** $σ$ **in terms of** `logvar`
Expressing standard deviation $σ$, in the reparameterization trick in terms of `logvar`  
Mathematically:
+ $σ = \sqrt{σ^2} = ({σ^2})^\frac{1}{2} = ({σ^2})^{0.5} = (e^{logvar})^{0.5}$
+ From laws of indices,
> + $(e^{logvar})^{0.5} = e^{0.5 \cdot logvar}$
+ Therefore,
> + Standard deviation, `std`, $σ = e^{0.5 \cdot logvar}$



## Tools ⚒️
**The tools used for this repository are:**
1. [Google Colab](https://colab.google/) - A hosted _Jupyter Notebook_ service by Google.
2. [`torch`](https://pytorch.org/) - (PyTorch) - An open-source **Deep Learning** library
3. [`lightning`](https://lightning.ai/docs/pytorch/stable/) - (PyTorch Lightning) - An open-source framework that serves as a high-level interface for PyTorch
4. [`matplotlib`](https://matplotlib.org/) - A comprehensive library for making static, animated, and interactive visualizations in Python
5. [`plotly`](https://plotly.com/python/) - An interactive, open-source, and browser-based graphing library
6. [`sklearn`](https://scikit-learn.org/stable/getting_started.html) - An open-source Python toolset that provides algorithms for machine learning, data preprocessing, and model evaluation
7. [`numpy`](https://numpy.org/) - The fundamental package for scientific computing with Python
8. [`pandas`](https://pandas.pydata.org/) - An open-source tool for data analysis and manipulation in Python
9. [`torchinfo`](https://github.com/TylerYep/torchinfo) - A library for inspecting neural network layer structures and network parameter details
10. [`transformers`](https://huggingface.co/docs/transformers/en/index) - An open-source Python toolkit designed to simplify the use of state-of-the-art machine learning models

> _Notable modules from [The Python Standard Library](https://docs.python.org/3/library/index.html) include [`pathlib`](https://docs.python.org/3/library/pathlib.html) and [`os`](https://docs.python.org/3/library/os.html)_

## Results 📈
+ _Structure of Neural Network_
> <img src='https://github.com/Martinmbiro/variational-autoencoder/raw/refs/heads/main/plots/Screenshot%202026-06-07%20211742.png' alt='neural_network_structure' width=550/>

+ _Training Logs_
> <img src='https://github.com/Martinmbiro/variational-autoencoder/raw/refs/heads/main/plots/Screenshot%202026-06-07%20211550.png' alt='training_logs' width=450 />

+ _Training Loss [Reconstruction Loss (BCE Loss) + KL Divergence]_
> <img src='https://github.com/Martinmbiro/variational-autoencoder/raw/refs/heads/main/plots/training_loss.png' alt='training_loss' width=650 />

+ _Input vs reconstructed input_
> <img src="https://github.com/Martinmbiro/variational-autoencoder/raw/refs/heads/main/plots/x_vs_x_recon.png" alt="x vs x'" width=650/>

+ _Sampled latent space_
> <img src="https://github.com/Martinmbiro/variational-autoencoder/raw/refs/heads/main/plots/latent_space_sampled.png" alt="sampled_latent_space" width=650/>

+ _Latent space in 2D & 3D_

|      |       |
| :--- | :---: |
| <img src="https://github.com/Martinmbiro/variational-autoencoder/raw/refs/heads/main/plots/latent_space_2d.png" alt="latent_space_2d" width=390/> | <img src="https://github.com/Martinmbiro/variational-autoencoder/raw/refs/heads/main/plots/latent_space_3d.png" alt="latent_space_3d" width=400/> |
<br>

## Roadmap 🎌
<center>
<img src='https://github.com/Martinmbiro/variational-autoencoder/raw/refs/heads/main/plots/roadmap.png' alt='auto_encoder_sketch' width=800/>
</center>
<p align='center'><i>This repository is part of a roadmap for understanding Generative Computer Vision</i></p>

## References 📚
+ [Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. _arXiv preprint arXiv:1312.6114._](https://arxiv.org/abs/1312.6114)
+ [Variational Autoencoders Explained From Scratch | Principles of Diffusion Models](https://www.youtube.com/watch?v=Yyir7-OAnlA) - by [Vizuara AI Labs](https://vizuara.ai/)
+ [Variational Autoencoders explained from scratch](https://www.vizuaranewsletter.com/p/variational-autoencoders-explained) - by [Vizuara AI Labs](https://vizuara.ai/)
+ [Variational Autoencoder (VAE) from scratch | Intuition + Coding](https://www.youtube.com/watch?v=VUwAGLM6K_8&t=5551s) - by [Vizuara AI Labs](https://vizuara.ai/)
+ [What is a variational autoencoder?](https://www.ibm.com/think/topics/variational-autoencoder) - by [IBM](https://www.youtube.com/@IBMTechnology)




