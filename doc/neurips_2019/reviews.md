#R1

Questions
1. Contributions: Please list three things this paper contributes (e.g., theoretical, methodological, algorithmic, empirical contributions; bridging fields; or providing an important critical analysis). For each contribution, briefly state the level of significance (i.e., how much impact will this work have on researchers and practitioners in the future?). If you cannot think of three things, please explain why. Not all good papers will have three contributions.
1. A new dynamic model for count data which generalises the Poisson Gamma Dynamical System).
2. A MCMC sampler with some novel update steps.
3. Illustrates the method on a range of problems.
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
