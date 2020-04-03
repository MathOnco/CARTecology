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
   
    # Define parameters from previous fit (of 2pop model)
    rN = 0.16; kN = 500.0; rB = 0.065

    if abs(fmem-0.1) < 0.01
        rMmax = 0.37; rMmin = 0.033;
        kM = 131.68; tau = 16.6;
    elseif abs(fmem-0.5) < 0.01
        rMmax = 0.414; rMmin = 0.0214;
        kM = 159.54; tau = 18.78;
    else
        rMmax = 0.32; rMmin = 0;
        kM = 250.0; tau = 35.22;
    end

    # Grab populations
    (N,M,E,B) = u

    # Grab the parameters
    # Parameters: (mu,rEff0,Gamma_E,Gamma_B)
    (mu,rEff0,Gamma_E,Gamma_B) = p

    # Net growth rate of CAR effector population
    rM = max((rMmax - rMmin)/(1 + exp(-(tau - t))) + rMmin,0.0)

    # Asymmetric differentiation rate is tumor-size dependent
    # rEff = rEff0*(1.0 + exp(-alpha_1*B)*(1.0 - exp(-alpha_2*B)))
    # rEff = rEff0*(1.0 + alpha_1*exp(-((B - alpha_3)/alpha_2)^2))
    rEff = rEff0
    # rEff = rEff0;
    
    du[1] = -rN*N*log((N+M)/kN)
    du[2] = -rM*M*log((N+M)/kM)
    du[3] = rEff*M - Gamma_E*E*B - mu*E
    du[4] = rB*B - Gamma_B*E*B

end

function my_loss_function(sol,params)
    if maximum(sol.t) < t[end] + 1
        @warn "max time is $(maximum(sol.t))"
        return 1e10
    end

    tot_loss = 0.0

    (mu,rEff0,Gamma_E,Gamma_B) = params;

    # Approximation of the variance by looking at quartiles
    weights = [2225.4;612.6;12.4177;0.5695;0.8687]

    # Loop over the time series data and compare to our solution
    for n in 1:length(t)
        tot_loss += (sol(t[n])[3] - CAReffectordata[n])^2/weights[n]
    end
    tot_loss *= lambda0

    # @show Tumordata

    # Time of disease progression (we assume occurs when Tumor[T] = Tumor[0])
    try
        tumorAtDay30 = sol(30)[4]
        tumorAtDay90 = sol(90)[4]
        tumorAtDay180 = sol(180)[4]

        if Tumordata[1] == PD
            if  tumorAtDay30 < PD
                tot_loss += lambda1*(Tumordata[1] - tumorAtDay30)
            end
        else
            tot_loss += lambda1*abs(Tumordata[1] - tumorAtDay30)
        end

        if Tumordata[2] == PD
            if  tumorAtDay90 < PD
                tot_loss += lambda1*(Tumordata[2] - tumorAtDay90)
            end
        else
            tot_loss += lambda1*abs(Tumordata[2] - tumorAtDay90)
        end

        if Tumordata[3] == PD
            if  tumorAtDay180 < PD
                tot_loss += lambda1*(Tumordata[3] - tumorAtDay180)
            end
        else
            tot_loss += lambda1*abs(Tumordata[3] - tumorAtDay180)
        end
    catch
        @warn "time 30 or later not found"
        tot_loss += 1e10
    end

    tot_loss += lambda2*sum(params)

    # Force the Effector population to grow initially?
    EffBirth = rEff0
    EffDeath = u0[3]/u0[2]*(Gamma_E*u0[4] + mu)
    tot_loss += 1e2*max(EffDeath - EffBirth,0)

    # @show params,tot_loss

    # @show lambda0, lambda1, lambda2
    
   return tot_loss
end

function cost_function2(params, earlyterminate=false)

    # Stop simulation if progression
    cb = nothing
    prob = ODEProblem(CARmodel,u0,tspan,params)
    sol = solve(prob,Rodas4(autodiff=false),maxiters=1e9,force_dtmin=true,callback=cb,abstol=1e-9,reltol=1e-6)

    if sol.retcode != :Success
        @show sol.retcode
        @show params
    end

    my_loss_function(sol,params)

end

function localmethod(Nparams,lower,upper)

    initparams = rand(Nparams).*(upper.-lower) .+ lower
    result = optimize(cost_function2, lower, upper,initparams,Fminbox(LBFGS(linesearch=LineSearches.HagerZhang(linesearchmax=500))),opts)

    #BFGS(linesearch=LineSearches.HagerZhang(linesearchmax=500))

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

    fig = plot(sol.t,sol[3,:],label=["CAR effector"],yaxis=:log,legend=:bottomleft)
    xlims!((0.0,200.0))
    ylims!((1e-8,1.1*maximum(CAReffectordata)))
    scatter!(fig,t,CAReffectordata,label=L"CAR_{data}")
    # scatter!(fig,twildtype,Wildtypedata,label=L"Normal_{data}")
    return fig
end

# Options
runlocal = true
ploton = true

useOriginalData = true

# Optimization parameters
# lambda0 = 1e0       # Weight of CAR data
# lambda1 = 1e-2       # Weight of tumor burden data
# lambda2 = 1e0;        # L1 Regularization

lambdaVals = [0.1;1.0;10.0]

lambdaSet = collect([10.0,x2,x3] for x2 in lambdaVals, x3 in lambdaVals)
lambdaSet = lambdaSet[:]

Nparams = 4
# Parameters: (mu,rEff0,Gamma_E,Gamma_B)
lower = zeros(Nparams).+1e-12
upper = 10.0*ones(Nparams)
# upper[1] = 50.0;
# lower[2] = 0.01
# upper[2] = 0.5;
# upper[3] = 150.0;
bounds = repeat([(1e-13,10.0)],Nparams)
bounds[2] = (1e-13,1.0)
opts = Optim.Options(show_trace=false); #iterations=Int(1e4)

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

#= It is assumed that 2% of circulating T cells are in peripheral blood at any
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

# We create the 243 possible combinations of Q1-Q3

global counter = 0;

if !useOriginalData
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

effectorFreq = [0.9;0.5;0.1]

# Median wildtype data
twildtype = [5;7;14;28;90;180]
Wildtypedata = 10^3*[0.08;0.16;0.28;0.48;0.49;0.47]

tspan = (0.0,200.0)

# Initial wildtype and CAR data
w0 = 6.0    # 6.0*5*10^6/0.01
CAR0 = 0.36 # 0.36*5*10^6/0.01
B0 = 200.0 #(52.69)^(1.5)*pi/6*10^9  # (52.69)^(3/2)*pi/6*10^9

global maxTumorSize = 2.0*B0

# Initial tumor data and patient types
(CR,SD,PD) = [0.0,1.0,2.0]*B0

# Update at time points 30,90,180
TumorDataVec = ([CR;PD;PD],[PD;PD;PD],[CR;CR;CR])

directory = string(pwd(),"/data/");
nLocalRuns = 50
filename = "outputData"
extension = ".txt"

for n = 1 : 1000
    global FILE = string(directory,filename,"_",string(n),"_",Dates.today(),extension)
    global paramFILE = string(directory,filename,"_params_",string(n),"_",Dates.today(),extension)
    if ~isfile(FILE)
        @printf("%s file created!\n",FILE)
        break
    end
end
myfile = open(FILE, "w")

myparamfile = open(paramFILE,"w")

@printf("Running %d local searches...\nlambdas\tCAR effector\ttumor data\tu0\tparams\tmin val\n--------------------------------------------------------------\n",nLocalRuns)

write(myfile,"#lambdas\tCAR effector\ttumor data\tu0\tparams\tmin val\n#--------------------------------------------------------------------------------------------------------------------------------------------\n")

write(myparamfile,"#(mu,rEff0,Gamma_E,Gamma_B)\n")

flush(myfile)
flush(myparamfile)

if !useOriginalData
    global results = Array{Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64}}(undef,3^nT*length(effectorFreq))
else
    global results = Array{Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1},Array{Float64,1},Array{Float64,1},Float64}}(undef,length(TumorDataVec)*length(effectorFreq)*length(trial)*length(lambdaSet))
end

# We run the optimizer over three possible effector fractions and 3^5 = 243
# time points
global counter = 0
for (indexT,TumorVal) in enumerate(TumorDataVec)
    for (indexF,f) in enumerate(effectorFreq)
        for m = 1 : length(trial)
            for (indexLam,lambdas) in enumerate(lambdaSet)

                global CAReffectordata = f*CARdata[trial[m]]

                global (lambda0,lambda1,lambda2) = lambdas

                global u0 = [w0;(1.0-f)*CAR0;f*CAR0;B0]

                # Grab memory freq
                global fmem = 1.0-f

                bestmin = 1e10
                bestparams = lower

                global Tumordata = TumorVal

                if runlocal
                    for n = 1 : nLocalRuns
                        result = localmethod(Nparams,lower,upper)
                        if result.minimum < bestmin
                            bestmin = result.minimum
                            bestparams = result.minimizer
                        end
                        # println(CAReffectordata,'\t',Tumordata,'\t',u0,'\t',result.minimizer,'\t',result.minimum)
                    end
                else
                    result = globalmethod(Nparams,bounds)
                    if best_fitness(result) < bestmin
                        bestmin = best_fitness(result)
                        bestparams = best_candidate(result)
                    end
                end

                println(lambdas,CAReffectordata,'\t',Tumordata,'\t',u0,'\t',bestparams,'\t',bestmin)
                global counter += 1
                results[counter] = (lambdas,CAReffectordata,Tumordata,u0,bestparams,bestmin)

                writedlm(myparamfile,bestparams')
                flush(myparamfile)

                if ploton
                    fig = plotsol(CARmodel, bestparams)
                    filename = string(directory,"fourPopulationNoTumorDependence_",string(counter),"_",Dates.today(),".pdf")
                    savefig(fig,filename)
                end
            end
        end

    end
end

writedlm(myfile,results)

close(myfile)
close(myparamfile)

ploton = true
if ploton

    fig = plotsol(CARmodel, bestparams)
    filename = string(directory,"savedFig.pdf")
    savefig(fig,filename)

end