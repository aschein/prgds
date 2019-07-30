# R1

1. Contributions: Please list three things this paper contributes (e.g., theoretical, methodological, algorithmic, empirical contributions; bridging fields; or providing an important critical analysis). For each contribution, briefly state the level of significance (i.e., how much impact will this work have on researchers and practitioners in the future?). If you cannot think of three things, please explain why. Not all good papers will have three contributions.

1) A new dynamic model for count data which generalises the Poisson Gamma Dynamical System).
2) A MCMC sampler with some novel update steps.
3) Illustrates the method on a range of problems.

2. Detailed comments: Please provide a thorough review of the submission, including its originality, quality, clarity, and significance. Hover over the "?" next to this prompt to see a brief description of these metrics.

This is a nicely written paper. The additions lead to a sampler which is simpler to implement. The method is shown to perform much better than the PGDS model in some data sets. Here are a few comments:
1) the transition for \theta_t (summing over h_k^{(t)}) is mixture of gamma distribution (in contrast to the PGDS model which assumes a single gamma distribution). It would be interesting to see what effects this has on the properties of the model. There is talk of "bursty" behaviour but I didn't see this effect being clearly explored in the paper.
2) Why is \epsilon_0^{(g)}=1 used as a canonical choice when this parameter is non-zero. Why? Would it make sense to put a prior on this parameter and learn it from the data.
3) The MCMC sampler is simpler to implement but I didn't see any discussion of the timings of the different methods. It would be interesting to know if there were differences in the computational times.
4) The sparse PrGDS performs a lot better than the non-sparse version on the GDELT, ICEWS (as matrices), NeurIPS and DBLP data sets but the performance is similar for the other data sets. It would be interesting to know what aspect of the data favour the sparse version (other, presumably, the DGP being "sparse"). Why is there a difference in performance for the ICEWS data when looking matrices but not tensor.
5) The PrGDS outperforms the PGDS for the GDELT and ICEWS (as matrices) but gives similar performance on other data sets. Is there some way to know from the data when the PrGDS will perform much better than PGDS?

Originality: This is an extension of the PGDS model but I think that this is an important development since it allows the introduction of sparsity which leads to much better predictive performance for some problems.

Quality: A lot of work has obviously gone in to this paper. The construction of the Gibbs sampler is neat, the method is well-explained and there are a range of comparisons. My one main criticism is that the competing benefits of the two methods could be better explored.

Clarity: The paper is well-written and clear.

Significance: I think that this is a nice complementary idea to the PGDS which add some extra sparsity into the mix. This clearly leads to better predictive performance in some data sets.

3. Please provide an "overall score" for this submission.

7: A good submission; an accept. I vote for accepting this submission, although I would not be upset if it were rejected.

4. Please provide a "confidence score" for your assessment of this submission.

5: You are absolutely certain about your assessment. You are very familiar with the related work.

5. Improvements: What would the authors have to do for you to increase your score?

Provide a deeper discussion of the performance of the different methods.

# R2

Questions
1. Contributions: Please list three things this paper contributes (e.g., theoretical, methodological, algorithmic, empirical contributions; bridging fields; or providing an important critical analysis). For each contribution, briefly state the level of significance (i.e., how much impact will this work have on researchers and practitioners in the future?). If you cannot think of three things, please explain why. Not all good papers will have three contributions.
The paper makes the following contributions:
1) It presents a new Allocative Poisson Factorization model for sequentially observed count data (tensors) that avoids augmentation via chaining of poisson and gamma latent states.
2) It offers a Gibbs sampler that exploits the closed-form conditionals offered.
3) It demonstrates the capabilities of PrGDS in real world experiments against prior art.
Finally, the new model has a natural inductive bias due to the prior hyper-parameter construction that trade-offs sparsity and smoothness.
2. Detailed comments: Please provide a thorough review of the submission, including its originality, quality, clarity, and significance. Hover over the "?" next to this prompt to see a brief description of these metrics.
The contribution builds on the recent work on Poisson Gamma Dynamical Systems by Schein et al. and its deep variants albeit departing from the standard PGDS formulation and required augmentation scheme by introducing alternating chains of Poisson and Gamma latent state variables. This expanded construction offers closed-form conditionals via the relatively unknown Bessel distribution and the authors defined SCH distribution whose MGF is derived.

The paper offers significant novelty and extends the state of the art in the area. A nice characteristic of the contribution is that although the latent structure is expanded (in relation to PGDS) in the model (and hence its complexity and any identifiability issues), the induced sparsity in the latent states and the closed form conditionals from the Poisson-gamma chaining simplifies inference and helps (?) with identifiability.

The paper is well written, the experiments are convincing and demonstrate improvement over the PGDS and a simple dynamic baseline (GP-DPFA). I appreciate the accompanying code for both PGDS and PrGDS and the expanded material in the appendix.

My only minor concerns are:
- Identifiability: There is little mention (if any) of this in this line of work and more complex constructions of latent state chaining will suffer further on this aspect. Is the model identifiable and to what extend? Does the induced sparsity help? Is that shown anywhere?And how does that relate to the observations made in Sec. 4 of the appendix?
- Computational complexity: please provide this in comparison to PGDS
- MCMC: Any evidence of convergence of the chain?any rates?
- MCMC: Since you have a Gibbs sampler you should be able to also derive an SVI or EM variant? 
- hyper-parameters: why is \epsilon_0^{(\lambda)} kept at 1 and would a 0 induce further sparsity on the component-weights?
- "perplexity": This is mentioned from the start but only defined in experiments section. A pointer to that would help in the start. More importantly: Is this the right measure to be using for a probabilistic model like this? wouldn't you be interested in more Bayesian measures like the coverage you are offering? See for example: 

Leininger, T. J., Gelfand, A. E., et al. (2017). Bayesian inference and model assessment for spatial point pat- terns using posterior predictive samples. Bayesian Analysis, 12(1):1–30.
3. Please provide an "overall score" for this submission.
7: A good submission; an accept. I vote for accepting this submission, although I would not be upset if it were rejected.
4. Please provide a "confidence score" for your assessment of this submission.
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
5. Improvements: What would the authors have to do for you to increase your score?
I have provided some detailed questions and (implied?) recommendations above - any answers on these would help me better understand some of the limitations and benefits of the model and inference scheme.
I would like to see some bigO notation, discussion or demonstration on identifiability, something supporting or replacing perplexity and MCMC answers.

# R3

Questions
1. Contributions: Please list three things this paper contributes (e.g., theoretical, methodological, algorithmic, empirical contributions; bridging fields; or providing an important critical analysis). For each contribution, briefly state the level of significance (i.e., how much impact will this work have on researchers and practitioners in the future?). If you cannot think of three things, please explain why. Not all good papers will have three contributions.
1, This paper introduces a poisson-gamma-poisson motif on the Poisson Gamma Dynamic Systems to achieve a tractable sampler (via semi-conjugate way) and more expressive model (via less restriction on prior settings).

2, The presentation is well organised and it is a pleasant reading.
2. Detailed comments: Please provide a thorough review of the submission, including its originality, quality, clarity, and significance. Hover over the "?" next to this prompt to see a brief description of these metrics.
This paper uses a new trick on the Poisson Gamma Dynamic Systems to achieve tractable and more expressive models. Experimental results verify the advantages of the newly proposed model and look to be promising.

The poisson-gamma-poisson motif proposed in this paper contains substantial originality. In my rough understanding, this technique can be readily applied to other models (e.g. Gamma Belief Networks, maybe Dirichlet Belief Networks) and circumvent the complex data augmentation techniques usually required. Thus, this paper will have impact on the community. 

Since this trick introduces an "auxiliary" discrete variable to the standard PGDS, I am interested in the sampling efficiency with comparison to the standard PGDS. I think should be ok, just to confirm.

I like this paper, especially the poisson-gamma-poisson trick. Maybe the only pity is that this trick only applies on the rate parameter of Gamma distribution. 

3. Please provide an "overall score" for this submission.
7: A good submission; an accept. I vote for accepting this submission, although I would not be upset if it were rejected.
4. Please provide a "confidence score" for your assessment of this submission.
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
5. Improvements: What would the authors have to do for you to increase your score?
Maybe an exploration on the sampling efficiency as it introduces "auxiliary" variables. 

The current form looks good to me. I think 7 is the maximum I can give for this theme, as this trick applies only to the rate parameter of the Gamma distribution.
