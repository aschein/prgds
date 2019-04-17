*4/17/19*

Thinking about results. I've run the matrix experiments using a bunch of different variants of PrGDS:

PrGDS-v1 (version 1) uses shrinkage weights drawn from the usual gamma process:
 
      nu_k ~ Gam(gamma_0 / K, beta)

  And, it only uses them to shrink the first time step of latent Poisson states:
  
      h_1k ~ Pois(nu_k ...)

  Meanwhile, the data is shrunk using a completely different set of weights:
  
      y_tk ~ Pois(lambda_k ...)

PrGDS-v2 (version 2) uses shrinkage weights drawn from a Poisson-randomized gamma process:
      
      gamma_0, beta ~ Gam(...)
      g_k ~ Pois(gamma_0 / K)
      nu_k ~ Gam(eps + g_k, beta)

And, it uses them to shrink both the first latent state and the data:

    h_1k ~ Pois(nu_k ...)
    y_tk ~ Pois(lambda_k ...)

I've run experiments using both v1 and v2. For both, I've tried with:
* Dirichlet versus gamma priors over the factor matrix in the Poisson likelihood
* eps=0 versus eps=1 for the Poisson-randomized gamma prior over theta_tk

And, for v2 I've additionally tried with:
* eps=0 versus eps=1 for the prior over nu_k

Results observations:
* The forecasting performance is **terrible** when using gamma priors over the factor matrix; this is especially true for v2.
* The Dirichlet models for v2 are generally doing the best.

We want to show that developing this whole model class was worth it. One of the main sells is that using Poisson-randomized dynamics yields a richer model class---i.e., one that allows for gamma priors over factors in the Poisson likelihood. However, remember, we can make this point for **just the gamma priors over \lambda_k**

