###########################################################

# random sample of size n, from the qExponential(q,b) distribution
function rqexp(n,q,b)
   us = rand(Uniform(0,1),n)
   invals = (ones(n) - (ones(n) - us).^((q-1)/q))/(b*(q-1))
   return(invals)
end

###########################################################

# pdf of the qExponential(q,b) distribution
function dqexp(xs,q,b)
   ys = q*b*(ones(length(xs))-(q-1)*b*xs).^(1/(q-1))
   return(ys)
end

###########################################################
# Monte Carlo repetitions of the qExponential \beta-parameter and MSE calculation (Figure 3)
# Estimation methods: Shannon - Renyi (linear constraints) - Renyi (escort constraints)
# ss: sample size, nreps: number of Monte Carlo repetitions, q,b: parameters of the qExponential

function monteCarlo_betas(ss, nreps, q, b)
   qhat = zeros(nreps)
   bhat_sh = zeros(nreps)
   bhat_r1 = zeros(nreps)
   bhat_r2 = zeros(nreps)
   betas = zeros(nreps,3)
   for i in 1:nreps
      global data
      # generate qexp sample of size ss
      data = rqexp(ss,q,b)
      ##################################################################
      g(x) = 1/length(data)*sum(data ./ (1 .+ data ./ x)) * (1. + (length(data) / sum(log.(1 .+ data / x))) ) - x
      strue = 1/(b * (1-q))
      sigmahat = find_zero(g, rand(Uniform(strue-ee,strue+ee)))
      # calculate estimators
      thetahat = length(data)/sum(log.(1 .+ data / sigmahat))
      qhat[i] = thetahat / (thetahat + 1.)
      ##
      bhat_sh[i] = 1. / (sigmahat * (1. - qhat[i]))
      bhat_r1[i] = 1/((2q-1)*mean(data)) #linear constraint estimator
      bhat_r2[i] = 1/(q*mean(data)) #escort constraint estimator
   end
   betas[:,1] = bhat_sh
   betas[:,2] = bhat_r1
   betas[:,3] = bhat_r2
   return(betas)
end

###########################################################

# Monte Carlo repetitions of the qExponential \beta-parameter and MSE calculation (Figure 5)
# Estimation methods: Shannon - Renyi (linear constraints) - Renyi (escort constraints)
# ss: sample size, nreps: number of Monte Carlo repetitions, q,b: parameters of the qExponential
function monteCarloMSE(ss, nreps, q, b)
   # initialization
   qhat = zeros(nreps)
   bhat_sh = zeros(nreps)
   bhat_r1 = zeros(nreps)
   bhat_r2 = zeros(nreps)
   for i in 1:nreps
      global data
      # generate qexp sample of size ss
      data = rqexp(ss,q,b)
      ##################################################################
      g(x) = 1/length(data)*sum(data ./ (1 .+ data ./ x)) * (1. + (length(data) / sum(log.(1 .+ data / x))) ) - x
      strue = 1/(b * (1-q))
      sigmahat = find_zero(g, rand(Uniform(strue-ee,strue+ee)))
      # calculate estimators
      thetahat = length(data)/sum(log.(1 .+ data / sigmahat))
      qhat[i] = thetahat / (thetahat + 1.)
      ##
      bhat_sh[i] = 1. / (sigmahat * (1. - qhat[i])) # shannon estimator (coincides with MLE)
      bhat_r1[i] = 1/((2q-1)*mean(data)) # Renyi linear constraint estimator
      bhat_r2[i] = 1/(q*mean(data)) # Renyi escort constraint estimator
   end
   mser = zeros(4)
   # calculate MC MS errors
   mser[1] = var(qhat) + (mean(qhat)-q)^2
   mser[2] = var(bhat_sh) + (mean(bhat_sh)-b)^2
   mser[3] = var(bhat_r1) + (mean(bhat_r1)-b)^2
   mser[4] = var(bhat_r2) + (mean(bhat_r2)-b)^2
   return(mser)
end

###########################################################
