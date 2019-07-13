import torch
import torch.nn.functional as F


def loss_fn(original_seq, recon_seq, f_mean, f_logvar, z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar):
    """
    Loss function consists of 3 parts:
    1. the reconstruction term that is the MSE loss between the generated and the original images;
    2. the KL divergence of f;
    3. and the sum over the KL divergence of each z_t,
    with the sum divided by batch_size

    Loss = {mse + KL of f + sum(KL of z_t)} / batch_size
    Prior of f is a spherical zero mean unit variance Gaussian,
    and the prior of each z_t is a Gaussian whose mean and variance are given by the LSTM.
    """
    batch_size = original_seq.size(0)
    mse = F.mse_loss(recon_seq, original_seq, reduction='sum')
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean, 2) - torch.exp(f_logvar))
    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar + (
        (z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)

    return (mse + kld_f + kld_z) / batch_size, mse / batch_size, kld_f / batch_size, kld_z / batch_size
