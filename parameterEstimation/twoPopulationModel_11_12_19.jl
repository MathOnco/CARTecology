using BlackBoxOptim
using DifferentialEquations
using RecursiveArrayTools # for VectorOfArray
using Optim
using PenaltyFunctions
using Plots
using LaTeXStrings
using DelimitedFiles
using LineSearches
using Printf
using Dates
using Random
# using Statistics

function CARmodel(du,u,p,t)
   
   (rM,kM,rMsmall,tau,rN) = p
   (N,M) = u

    N = max(N,0)
    M = max(M,0)

    if (N+M)/kM < 0 || (N+M)/kN < 0
        @show(N+M,kM)
    end

    # Time sensitive growth rate of CAR memory compartment
    rMfunc = max((rM - rMsmall)/(1 + exp(-(tau - t))) + rMsmall,0.0)

    # ODEs
    du[1] = -rN*N*log((N+M)/kN)
    du[2] = -rMfunc*M*log((N+M)/kM)

end

function my_loss_function(sol,params)
    if maximum(sol.t) < t[end] + 1
        # @warn "max time is $(maximum(sol.t))"
        return 1e10
    end

    tot_loss = 0.0

    (rM,kM,rMsmall,tau,rN) = params;

    # Variances computed 
    weights = [2225.4;612.6;12.4177;0.5695;0.8687]

    for n in 1:length(t)
        tot_loss += lambda0*(sol(t[n])[2] - CARmemorydata[n])^2/weights[n]
    end

    # for n in 1:length(t)
    #     tot_loss += lambda0*(sol(t[n])[2] - CARmemorydata[n])^2
    # end

    tot_loss += lambda1*max(0,kM - kN)
    tot_loss += lambda2*max(0,sum(u0) - kM)
    tot_loss += lambda1*max(0,rMsmall - rM)

    global maxCAR = 0.0
    global tmax = 0.0
    global maxWildtype = 0.0

    # Get max soln array
    for i in 1:length(sol.u)
        solu = sol.u[i]
        solCAR = solu[2]
        solWildtype = solu[1]

        # Update maxCAR and time of maxCAR
        if solCAR > maxCAR
            tmax = sol.t[i]
            maxCAR = solCAR
        end
        if solWildtype > maxWildtype
            maxWildtype = solWildtype
        end
    end

    # Time of peak should be around 7 days
    tot_loss += lambda3*(tmax - t[1])^2

    # Max CAR should be at 7 days
    tot_loss += lambda4*(maxCAR - maxCARmemorydata)^2

    for n in 1:length(twildtype)
        tot_loss += lambda5*(sol(twildtype[n])[1] - Wildtypedata[n])^2
    end

    # @show tot_loss
    
   return tot_loss

end

function cost_function2(params, earlyterminate=false)

    prob = ODEProblem(CARmodel,u0,tspan,params)
    cb = nothing
    sol = solve(prob,Rodas4(autodiff=false),maxiters=1e11,force_dtmin=true,callback=cb,abstol=1e-9,reltol=1e-6)

    if sol.retcode != :Success
        @show sol.retcode
        @show params
    end

    my_loss_function(sol,params)

end

function localmethod(Nparams,lower,upper)

    initparams = rand(Nparams).*(upper.-lower) .+ lower
    result = optimize(cost_function2, lower, upper,initparams,Fminbox(BFGS(linesearch=LineSearches.HagerZhang(linesearchmax=500))),opts)

    return result

end

function globalmethod(Nparams,bounds)

    result = bboptimize(cost_function2;SearchRange = bounds, MaxSteps = 2e4,Method=:de_rand_1_bin,TraceMode=:silent)

    return result

end

function plotsol(CARmodel, result)
    p_opt = result
    prob = ODEProblem(CARmodel,u0,tspan,p_opt)
    cb = nothing
    sol = solve(prob,Rosenbrock23(),maxiters=1e9,force_dtmin=true,callback=cb,abstol=1e-12,reltol=1e-9)

    fig = plot(sol,label=["Wildtype","CAR memory"],yaxis=:log,legend=:bottomleft)
    xlims!((0.0,200.0))
    ylims!((1e-8,1.1*max(maxCAR,maxCARmemorydata,maximum(Wildtypedata))))
    scatter!(fig,t,CARmemorydata,label=L"CAR_{data}")
    scatter!(fig,twildtype,Wildtypedata,label=L"Normal_{data}")
    return fig
end

# Parameters
kN = 500.0
# rN = 0.16
runlocal = true
ploton = true
useOriginalData = true

# Optimization parameters
lambda0 = 1e3       # Weight of CAR data
lambda1 = 1000.0     # rho > C + D
lambda2 = 0.0      # Weight of tumor burden
lambda3 = 0.0       # Correspond peak data time to peak sim time
lambda4 = 0.0       # Correspond max(CAR) to max(simCAR)
lambda5 = 1e-1      # Weight of the wildtype data
Nparams = 5
# Parameters: [rMmax kM rMmin tau rN] (b = 1, logistic, b = 0 Gompertz)
lower = zeros(Nparams)
lower[1] = 0.05;
lower[2] = 10.0
lower[4] = 5.0
lower[5] = 0.02;
# lower[4] = 0.01
# lower[4] = 0.2
# lower[5] = 0.05;
upper = [1.0;0.95*kN;1.0;50.0;1.0] #kN*ones(Nparams)
bounds = repeat([(0.0,20.0)],Nparams)
bounds[2] = (0.0,kN)
opts = Optim.Options(iterations=Int(1e4),show_trace=false);

#=
Q1 data
    #day    median  +/-
    7	10.8086	2.289
    14	3.3263	0.7044
    28	0.8178	0.1676
    90	0.0537	0.011
    180	0.0235	4.8224e-3


Median data
    #day	median	+/-
    7	28.1908	5.3653
    14	12.8407	2.4595
    28	2.188	0.4479
    90	0.4203	0.0862
    180	0.2697	0.057

Q3 data
    #day	Q3	+/-
    7	74.4459	14.2809
    14	36.7151	7.5299
    28	5.5715	1.1786
    90	1.0717	0.241
    180	1.2808	0.2707
=#

#= It is assumed that 1% of circulating T cells are in peripheral blood at any
given time. We will amend this assumption, by stipulating that this holds only
for T cells which have neutral selection. CAR T cells in the absence of tumor
should have an increasingly high negative selection and so we assume that most
of them will not arrest in the lymph nodes over time. Therefore, we assume a
logistic function that governs the migration rate from the lymph nodes given by
h(t) = 1 - 1/(1 + 49*exp(-b*t)). Thus at t = 0, h(0) = 49/50 = 0.98. That is, 98% of the CAR is in the lymph nodes and tissues. As t->infty, h(t) -> 0, that is all the CAR is now circulating.
=#

# Data
t = [7;14;28;90;180]
# t = [7;14;28]

nT = length(t)

Q1CAR = [10.81;3.33;0.82;0.054;0.0235]
medianCAR = [28.19;12.84;2.19;0.42;0.27]
Q3CAR = [74.4459;36.7151;5.5715;1.07;1.28]

CARcombined = hcat([Q1CAR,medianCAR,Q3CAR]...)

if !useOriginalData
    # We create the 243 possible combinations of Q1-Q3
    global counter = 0;
    CARdata = Array{Float64}(undef,(3^nT,nT))
    if nT == 5
        for i = 1 : 3
            for j = 1 : 3
                for k = 1 : 3
                    for l = 1 : 3
                        for m = 1 : 3
                            global counter += 1
                            CARdata[counter,:] = [CARcombined[1,i],CARcombined[2,j],CARcombined[3,k],CARcombined[4,l],CARcombined[5,m]]
                        end
                    end
                end
            end
        end
    elseif nT == 3
        for i = 1 : 3
            for j = 1 : 3
                for k = 1 : 3
                    global counter += 1
                    CARdata[counter,:] = [CARcombined[1,i],CARcombined[2,j],CARcombined[3,k]]
                end
            end
        end
    end
    trial = shuffle(1:3^nT)

else
    CARdata = (Q1CAR,medianCAR,Q3CAR)
    trial = shuffle(1:3)
end

memoryFreq = [0.1;0.5;0.9]

# Median wildtype data
# twildtype = [5;7;14;28;90;180]
# Wildtypedata = 10^3*[0.08;0.16;0.28;0.48;0.49;0.47]

twildtype = [7;14;28;90;180]
WildtypedataParent = 10^3*[0.16;0.28;0.48;0.49;0.47]

tspan = (0.0,200.0)

# Initial wildtype and CAR data
w0 = 6.0
CAR0 = 0.36

if !useOriginalData
    global results = Array{Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64}}(undef,3^nT*length(memoryFreq))
else
    global results = Array{Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64}}(undef,3*length(memoryFreq))
end

if ENV["USER"] == "gregorykimmel"
    directory = "/Users/gregorykimmel/Dropbox/_Projects/01_CART/code/parameterEstimation/twoPopulationModelData/"
    nLocalRuns = 20
# On Moffitt cluster or anywhere else we take present directory for storage
else
    directory = string(pwd(),"/data/");
    nLocalRuns = 50
end
filename = "outputData"
extension = ".txt"

todaysDate = Dates.today()

for n = 1 : 1000
    global FILE = string(directory,filename,"_",string(n),"_",todaysDate,extension)
    global paramFILE = string(directory,filename,"_params_",string(n),"_",todaysDate,extension)
    if ~isfile(FILE)
        @printf("%s file created!\n",FILE)
        break
    end
end
myfile = open(FILE, "w")

myparamfile = open(paramFILE,"w")

@printf("Running %d local searches...\nCAR memory\ttumor data\tu0\tparams\tmin val\n--------------------------------------------------------------\n",nLocalRuns)

write(myfile,"#CAR memory\ttumor data\tu0\tparams\tmin val\n#--------------------------------------------------------------------------------------------------------------------------------------------\n")

write(myparamfile,"#(rM,kM,rMsmall,tau,rN)\n")

flush(myfile)
flush(myparamfile)

# We run the optimizer over three possible memory fractions and 3^5 = 243
# time points
global counter = 0;
for (indexf,f) in enumerate(memoryFreq)
    for m = 1 : length(trial)

        if useOriginalData
            global CARmemorydata = f*CARdata[m]
        else
            global CARmemorydata = f*CARdata[m,:]
        end
        global maxCARmemorydata = maximum(CARmemorydata)

        global u0 = [w0;f*CAR0]

        global Wildtypedata = WildtypedataParent - CARdata[m]

        # global CARmemorydata /= maximum(CARmemorydata)
        # global Wildtypedata /= maximum(Wildtypedata)

        bestmin = 1e10
        bestparams = lower

        if runlocal
            for n = 1 : nLocalRuns
                result = localmethod(Nparams,lower,upper)
                if result.minimum < bestmin
                    bestmin = result.minimum
                    bestparams = result.minimizer
                end
            end
        else
            result = globalmethod(Nparams,bounds)
            if best_fitness(result) < bestmin
                bestmin = best_fitness(result)
                bestparams = best_candidate(result)
            end
        end

        println(CARmemorydata,'\t',u0,'\t',bestparams,'\t',bestmin)
        global counter += 1
        results[counter] = (CARmemorydata,u0,bestparams,bestmin)

        writedlm(myparamfile,bestparams')
        flush(myparamfile)

        if ploton
            fig = plotsol(CARmodel, bestparams)
            filename = string(directory,"TwoPopulation_",string(counter),"_",todaysDate,".pdf")
            savefig(fig,filename)
        end

    end
end

writedlm(myfile,results)

close(myfile)

# writedlm("/Users/gregorykimmel/Dropbox/01_CART/data/matrixMinVals_IncludeWildtype.txt",minvals,'\t')

# ploton = true
# if ploton

#     fig = plotsol(CARmodel, bestparams)
#     filename = string("/Users/gregorykimmel/Dropbox/01_CART/data/testfile25.pdf")
#     savefig(fig,filename)

# end