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
Random.seed!(4545)
global ee = 1e-09

qs = [0.55, 0.65, 0.75];#LinRange(0.55, 0.9, nsims)
betatrue = 10.;
qestimse = zeros(length(qs));
ss, nreps = 10000, 10000; # sample size and Monte Carlo repetitions
bestimse_sh = zeros(nreps,length(qs));
bestimse_r1 = zeros(nreps,length(qs));
bestimse_r2 = zeros(nreps,length(qs));

@time for i in 1:length(qs)
   display("$i/3")
   tt = monteCarlo_betas(ss, nreps, qs[i], betatrue);
   bestimse_sh[:,i] = tt[:,1];
   bestimse_r1[:,i] = tt[:,2];
   bestimse_r2[:,i] = tt[:,3];
end
##############
# writedlm("bestimseNe04_nsims10000_sh.txt",bestimse_sh)
# writedlm("bestimseNe04_nsims10000_r1.txt",bestimse_r1)
# writedlm("bestimseNe04_nsims10000_r2.txt",bestimse_r2)
####
# bestimse_sh = readdlm("bestimseNe03_nsims10000_sh.txt");
# bestimse_r1 = readdlm("bestimseNe03_nsims10000_r1.txt");
# bestimse_r2 = readdlm("bestimseNe03_nsims10000_r2.txt");
#
# bestimse_sh2 = readdlm("bestimseNe04_nsims10000_sh.txt");
# bestimse_r12 = readdlm("bestimseNe04_nsims10000_r1.txt");
# bestimse_r22 = readdlm("bestimseNe04_nsims10000_r2.txt");
####
pyplot()

a = kde_lscv(bestimse_sh[:,1]);
xvals, densityvals = a.x, a.density
p11 = plot(xvals, densityvals,label="Shannon",linewidth=2,title=L"(a)\,\, Q=0.55, \, n=10^3",titlefontsize=10,legendfontsize=7,legend=:topright,ylab="Density",xlim=[0,25],ylim=[0,1.5])
a = kde_lscv(bestimse_r1[:,1]);
xvals, densityvals = a.x, a.density
p11 = plot!(xvals, densityvals,label="Renyi (lin.)",linewidth=2)
a = kde_lscv(bestimse_r2[:,1]);
xvals, densityvals = a.x, a.density
p11 = plot!(xvals, densityvals,label="Renyi (esc.)",linewidth=2)
vline!([betatrue],label="",c="black",line=:dash)
#
a = kde_lscv(bestimse_sh[:,2]);
xvals, densityvals = a.x, a.density
p12 = plot(xvals, densityvals,linewidth=2,title=L"(b)\,\, Q=0.65, \, n=10^3",titlefontsize=10,legendfontsize=7,label="",xlim=[0,13],ylim=[0,2.5])
a = kde_lscv(bestimse_r1[:,2]);
xvals, densityvals = a.x, a.density
p12 = plot!(xvals, densityvals,linewidth=2,label="")
a = kde_lscv(bestimse_r2[:,2]);
xvals, densityvals = a.x, a.density
p12 = plot!(xvals, densityvals,linewidth=2,label="")
vline!([betatrue],label="",c="black",line=:dash)
#
a = kde_lscv(bestimse_sh[:,3]);
xvals, densityvals = a.x, a.density
p13 = plot(xvals, densityvals,linewidth=2,title=L"(c)\,\, Q=0.75,\, n=10^3",titlefontsize=10,legendfontsize=7,label="",xlim=[0,13],ylim=[0,3.5])
a = kde_lscv(bestimse_r1[:,3]);
xvals, densityvals = a.x, a.density
p13 = plot!(xvals, densityvals,linewidth=2,label="")
a = kde_lscv(bestimse_r2[:,3]);
xvals, densityvals = a.x, a.density
p13 = plot!(xvals, densityvals,linewidth=2,label="")
vline!([betatrue],label="",c="black",line=:dash)
#
a = kde_lscv(bestimse_sh2[:,1]);
xvals, densityvals = a.x, a.density
p21 = plot(xvals, densityvals,label="Shannon",linewidth=2,title=L"(a)\,\, Q=0.55, \, n=10^4",titlefontsize=10,legendfontsize=7,legend=:topright,ylab="Density",xlim=[0,25],ylim=[0,1.5])
a = kde_lscv(bestimse_r12[:,1]);
xvals, densityvals = a.x, a.density
p21 = plot!(xvals, densityvals,label="Renyi (lin.)",linewidth=2)
a = kde_lscv(bestimse_r22[:,1]);
xvals, densityvals = a.x, a.density
p21 = plot!(xvals, densityvals,label="Renyi (esc.)",linewidth=2)
vline!([betatrue],label="",c="black",line=:dash)
#
a = kde_lscv(bestimse_sh2[:,2]);
xvals, densityvals = a.x, a.density
p22 = plot(xvals, densityvals,linewidth=2,title=L"(b)\,\, Q=0.65, \, n=10^4",titlefontsize=10,legendfontsize=7,label="",xlab=L"\hat{\Lambda}",xlim=[0,13],ylim=[0,2.5])
a = kde_lscv(bestimse_r12[:,2]);
xvals, densityvals = a.x, a.density
p22 = plot!(xvals, densityvals,linewidth=2,label="")
a = kde_lscv(bestimse_r22[:,2]);
xvals, densityvals = a.x, a.density
p22 = plot!(xvals, densityvals,linewidth=2,label="")
vline!([betatrue],label="",c="black",line=:dash)
#
a = kde_lscv(bestimse_sh2[:,3]);
xvals, densityvals = a.x, a.density
p23 = plot(xvals, densityvals,linewidth=2,title=L"(c)\,\, Q=0.75,\, n=10^4",titlefontsize=10,legendfontsize=7,label="",xlim=[0,13],ylim=[0,3.5])
a = kde_lscv(bestimse_r12[:,3]);
xvals, densityvals = a.x, a.density
p23 = plot!(xvals, densityvals,linewidth=2,label="")
a = kde_lscv(bestimse_r22[:,3]);
xvals, densityvals = a.x, a.density
p23 = plot!(xvals, densityvals,linewidth=2,label="")
vline!([betatrue],label="",c="black",line=:dash)
#
plot(p11,p12,p13,p21,p22,p23,layout=(2,3)) # Figure 3
plot!(size=(1000,500))
# savefig("kdes_qs_ns2.pdf")

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
