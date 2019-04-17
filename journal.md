*4/17/19*

Thinking about results. I've run the matrix experiments using a bunch of different variants of PrGDS:

PrGDS-v1 (version 1) uses shrinkage weights drawn from the usual gamma process:
 
      nu_k ~ Gam(gamma_0 / K, beta)

  And, it only uses them to shrink the first time step of latent Poisson states:
  
      h1_k ~ Pois(nu_k)

  Meanwhile, the data is shrunk using a completely different set of weights:
  
      y_k ~ Pois(lambda_k ...)

 
  
