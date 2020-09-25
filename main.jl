#
using Plots, Distributions, DataFrames, DelimitedFiles, ColorSchemes, LaTeXStrings, Roots, NLsolve, Random
#
include("toolbox.jl")
#
Random.seed!(12345)
global ee = 1e-8
## generate random sample from the qExponential distribution
xs = [0.1:0.001:1;]
plot(xs,dqexp(xs,0.9,10.))
#
data = rqexp(50,0.9,10.)
#
histogram(data,normalize=true,color="beige",bins=500,label="data")
xs = [minimum(data):0.001:maximum(data);]
plot!(xs,dqexp(xs,0.9,10.),linewidth=2,xlim=[0,1],label=L"\rho_1^{\dagger}(x)",color="red")
#
savefig("qexpdata.pdf") # Figure 1
#
########################################################################
## Calculate \beta-estimators via the Shannon and Renyi MaxEnt procedures

nsims = 200 # number of simulations
qs = LinRange(0.51, 0.9, nsims) # grid of q values
betas = LinRange(1., 100., nsims) # grid of \beta values
# initialization
bestimse_sh = zeros(nsims, nsims)
bestimse_r1 = zeros(nsims, nsims)
bestimse_r2 = zeros(nsims, nsims)
qestimse = zeros(nsims, nsims)
strue = zeros(nsims, nsims)
ss, nreps = 1000, 1000 # sample size and Monte Carlo repetitions


# 
@time for i in 1:nsims
   display("$i/$nsims")
   for j in 1:nsims
      tt = sqrt.(monteCarloMSE(ss, nreps, qs[i], betas[j])) # root of the MSE
      qestimse[i,j] = tt[1]
      bestimse_sh[i,j] = tt[2]
      bestimse_r1[i,j] = tt[3]
      bestimse_r2[i,j] = tt[4]
   end
end
###
# save results
# writedlm("q_estimations.txt",qestimse)
# writedlm("b_estimations_sh.txt",bestimse_sh)
# writedlm("b_estimations_linear.txt",bestimse_r1)
# writedlm("b_estimations_escort.txt",bestimse_r2)
###

# Calculate relative RMS errors
for i in 1:length(bestimse_sh[:,1])
   bestimse_sh[:,i] = bestimse_sh[:,i] / betas[i]
   bestimse_r1[:,i] = bestimse_r1[:,i] / betas[i]
   bestimse_r2[:,i] = bestimse_r2[:,i] / betas[i]
end

## Plot results
p1 = heatmap(qs[1:end],betas[1:end],transpose(log10.(bestimse_r1[1:end,1:end])),c=cgrad(ColorSchemes.jet1.colors),
   xlabel=L"q",ylabel=L"\beta",title=(L"\log{\hat{MSE}}"))
#
p2 = heatmap(qs[1:end],betas[1:end],transpose(log10.(bestimse_r2[1:end,1:end])),c=cgrad(ColorSchemes.jet1.colors),
   xlabel=L"q",ylabel=L"\beta",line_z=[-1.5,0.5])
#
p3 = heatmap(qs[1:end],betas[1:end],transpose(log10.(bestimse_sh[1:end,1:end])),c=cgrad(ColorSchemes.jet1.colors),
   xlabel=L"q",ylabel=L"\beta",line_z=[-1.5,0.5])
#
plot(p1,p2,p3,layout=(3,1))
##
savefig("colormaps.png") # Figure 5

#########################################################################################################################
#########################################################################################################################
