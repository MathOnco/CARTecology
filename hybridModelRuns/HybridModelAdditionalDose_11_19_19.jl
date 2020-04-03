using Plots
using ProgressMeter
using DelimitedFiles
using Distributed
using Printf
using DifferentialEquations
using Dates

function cumdist(x)

    y = Vector{Float64}(UndefInitializer(),length(x))
    for n = 1 : length(x)
        y[n] = sum(x[1:n])
    end

    y = y./y[end]

    return y
end

function F(u,params)

    (N,M,E,B) = u
    (rN,rMmax,rMmin,tau,kN,kM,mu,rB,rEff0,alpha_1,alpha_2,Gamma_E,Gamma_B) = params

    tumorBirth        = rB*B
    tumorDeath        = Gamma_B*E*B 

    # @show (t[end],M,B)

    [tumorBirth,tumorDeath]
end

function CARmodel(du,u,p,t)

    (N,M,E,B) = u

    B0 = x0[4]
    
    # Grab the parameters
    # (mu,rB,rEff0,alpha_1,alpha_2,Gamma_E,Gamma_B) = p
    (rN,rMmax,rMmin,tau,kN,kM,mu,rB,rEff0,alpha_1,alpha_2,Gamma_E,Gamma_B) = p

    # Net growth rate of CAR effector population
    if t<tdoses[1]
        rM = max((rMmax - rMmin)/(1 + exp(-(tau - t))) + rMmin,0.0)
    else
        rM = max((rMmax - rMmin)/(1 + exp(-(tau+tdoses[1] - t))) + rMmin,0.0)
    end

    # Asymmetric differentiation rate is tumor-size dependent
    rEff = rEff0*(1.0 + alpha_1*min(B/B0,alpha_2))

    # Ensure numerical error doesn't cause negative pop sizes
    # if (N<0 || M<0 || E<0 || B<0)
    #     @show N,M,E,B
    # end
    N = max(N,0)
    M = max(M,0)
    E = max(E,0)
    B = max(B,0)
    
    du[1] = -rN*N*log((N+M)/kN)
    du[2] = -rM*M*log((N+M)/kM)
    du[3] = rEff*M - Gamma_E*E*B - mu*E

    # If we are stochastic we will assume the population does not change
    # in the specified interval
    if isDiscrete
        du[4] = 0
    else
        du[4] = rB*B - Gamma_B*E*B
    end

end

function continuousRegime(tspan,currentPop,p,tdoses=[])

    callbackON = true
    if callbackON
        # Check to see if the tumor has entered the stochastic regime
        condition(u,t,integrator) = stochasticThreshold-u[4]
        affect!(integrator) = terminate!(integrator)
        cb1 = ContinuousCallback(condition,affect!,nothing,abstol=1e-11)

        # Check to see when the tumor exceeds some c*original size. We define 
        # this as progression.
        condition2(u,t,integrator) = u[4]-tumorSizeThreshold*x0[4]
        affect2!(integrator) = terminate!(integrator)
        cb2 = ContinuousCallback(condition2,affect2!)
        condition3(u,t,integrator) = any(tdoses.==t)
        function affect3!(integrator)
            if chemodeplete
                integrator.u[1] = integrator.u[1]/100
                integrator.u[2] = integrator.u[2]/100 + x0[2]
                integrator.u[3] = integrator.u[3]/100 + x0[3]
            else
                integrator.u[2] = integrator.u[2] + x0[2]
                integrator.u[3] = integrator.u[3] + x0[3]
            end
        end
        cb3 = DiscreteCallback(condition3,affect3!)
        cbset = CallbackSet(cb1,cb2,cb3)
    else
        cb = nothing
    end
    prob = ODEProblem(CARmodel,currentPop,tspan,p)
    sol = solve(prob,Rodas4(autodiff=false),maxiters=1e11,force_dtmin=true,callback=cbset,tstops=tdoses,abstol=1e-9,reltol=1e-6)

    if (sol.retcode != :Success && sol.retcode != :Terminated)
        @show sol.retcode
        @show p
        @show sol.u[end],sol.t[end]-sol.t[end-1],isDiscrete
    end

    solu = Array(sol)

    return (sol.t,solu)

end

function nextDiscreteEvent(udiscrete,p,isDiscrete)

    rates = F(udiscrete,p)

    rateTotal = 0.0

    # If tumor is stochastic
    if isDiscrete
        rateTotal += sum(rates)
        discreteRates = rates
    end

    # If rateTotal = 0.0 (No stochastic event will occur)
    if rateTotal == 0.0
        return (tfinal,[0;0;0;0])
    end

    # Time till next event
    tau = -log(rand())/rateTotal

    # roll
    roll = rand()

    #= Find which event has occurred:
        1) Birth of tumor cell
        2) Death of tumor cell 
    =#

    index = findall(x->(x>roll),cumdist(discreteRates))[1]

    # # Update population if the event is in discrete regime
    # udiscrete += nu[index,:]

    return (tau,nu[index,:])

end

function runHybridModel(x0,initparams,sigma)

    # Stochimetric matrix which determines outcome of the events 
    global nu = [[0 0 0 1];[0 0 0 -1]]

    # Perturb the params fed by sigma variation
    global params = max.(initparams.*(1 .+ sigma*randn(length(initparams))),0.0)

    # Force parameters that go negative to their initial input
    if any(params.<0.0)
        params[params.<0.0] = initparams[params.<0.0]
    end

    # Initialize final time (set to 200 days)
    global tfinal = 1000.0  # Final time

    # Threshold at which stochastic fluctuations are considered
    global stochasticThreshold = 1e2

    # Indicies which show which population is in the stochastic regime
    global isDiscrete = false

    # Time span on which to solve the continuous regime
    global tspan = (0.0,tfinal)

    # Initalize continuous and discrete population vectors
    global continuousPop    = float.(x0)
    global discretePop      = x0
    global t             = 0.0

    global discreteChange = 0*nu[1,:]

    while t < tfinal

        # Solve the deterministic system until tumor becomes stochastic or
        # the tumor goes past the threshold for "progression"
        (tnew,solContinuous) = continuousRegime(tspan,continuousPop,params,tdoses)

        # update vector containing populations
        if isDiscrete
            solContinuous[:,end] += float(discreteChange)
            solContinuous[solContinuous[:,end].<0,end] .= 0.0
        end
        continuousPop = solContinuous[:,end]
        discretePop = ceil.(Int,continuousPop)

        # Check if tumor is in stochastic regime
        isDiscrete = continuousPop[4] - stochasticThreshold < 1e-6


        if t == tnew[end]
            @warn "Continuous model is having trouble!"
            isDiscrete = true
        end
        t = tnew[end];

        # Check if tumor is extinct
        if discretePop[4] == 0
            break
        end

        # Check if progression has occurred
        if discretePop[4] - tumorSizeThreshold*x0[4] >= -1e-2
            break
        end

        # Update discrete population after time tcurrent + tau
        if isDiscrete
            (tau, discreteChange) = nextDiscreteEvent(discretePop,params,isDiscrete)
            
            # new continuous time span
            tspan = (t,min(t+tau,tfinal))
        else
            tspan = (t,tfinal)
            discreteChange = 0*discreteChange
        end

        # @show t,discretePop,isDiscrete

    end

    return (t, discretePop, continuousPop, string(params))

end



nPatients = 10^3
V = 5.0*10^6        # Volume of blood in the body
VT = 10^9           # Average number of cancer cells per cm^3
freqInPeriph = 0.01

# [6.0;0.2;0.16;52.69] (cells/µL,cells/µL,cells/µL,cm) -> cells
rN      = 0.16;         # WT memory growth rate
rMmax   = 0.414;         # CAR memory growth rate
rMmin   = 0.0214;         # CAR memory growth rate
tau     = 18.78;
kN      = 500.0*V/freqInPeriph;      # WT memory carrying capacity
kM      = 159.54*V/freqInPeriph;     # CAR memory carrying capacity
##### Assymmetric differentiation rate ##########
# rEff(B) = rEff(0)*(1 + exp(-a1*B))
rEff0   = 2.26;
alpha_1 = 4.29;
alpha_2 = 1.35;
####################################################
mu      = 0.481;       # Effector death rate
rB      = 0.065; # 0.21          # Tumor birth rate
# Pair-wise interaction rates
Gamma_B = 0.03/V*freqInPeriph; # Tumor-Effector interaction (tumor dies)
Gamma_E = 0.003/VT;

# Initial condition and parameters (which can be perturbed if sigma )
CARmemoryfreqvec = [0.5]#(0.0:0.1:1.0)
# sigmavec = [0.0;0.05;0.1;0.15;0.2]
B0prefactorvec = [0.01;0.1;1.0;2.0;5.0;10.0]
rBvec = (0.04:0.025:0.19)
# WTinitvec = [6.0;12.0;18.0;24.0;30.0]

CARmemoryfreq = 0.5;
sigmaVal = 0.15;
WTinit = 6.0;
B0prefactor = 1.0

global tdoses = 1.0*[10.0; 2201.0; 3055.0; 4900.0; 6025.0]

firstAdditionalDose = (10.0:10.0:600.0)

firstdose = 100.0

global chemodeplete = true

global tumorSizeThreshold = 5.0

todaysDate = Dates.today()

CAR0 = 0.36
B0 = 200.0

for B0prefactor in B0prefactorvec
    for rB in rBvec

        tdoses[1] = firstdose

        initWT     = WTinit*V/freqInPeriph                          # 6.0*V
        initCARmem = CARmemoryfreq*CAR0*V/freqInPeriph           # 0.2*V
        initCAReff = (1.0 - CARmemoryfreq)*CAR0*V/freqInPeriph    # 0.16*V
        initTumor  = B0prefactor*B0*VT                       # 52.69*VT
        global x0 = [initWT;initCARmem;initCAReff;initTumor]#[50,10,40,50]
        global initparams = [rN,rMmax,rMmin,tau,kN,kM,mu,rB,rEff0,alpha_1,alpha_2,Gamma_E,Gamma_B]

        # Initialize results vector
        global results = Array{Tuple{Float64,String,String}}(undef,nPatients)

        # Introduce param noise
        global sigma = sigmaVal

        directory = string(pwd(),"/additionalDose/")
        if chemodeplete
            filename = "outputData_chemoOn"
        else
            filename = "outputData_chemoOff"
        end
        extension = ".txt"

        for n = 1 : 2000
            global FILE = string(directory,filename,"_",string(n),"_",todaysDate,extension)
            if ~isfile(FILE)
                @printf("%s file created!\n",FILE)
                break
            end
        end
        myfile = open(FILE, "w")

        header = @printf(myfile,"Initial conditions [N0,M0,E0,B0] = [%.3e,%.3e,%.3e,%.3e] with main parameters used to perturb with sigma = %.2e: rN = %.2e,rMmax = %.2e,rMmin = %.2e,tau = %.2e,kN = %.2e,kM = %.2e,mu = %.2e,rB = %.2e, rEff0 = %.2e, alpha_1 = %.2e, alpha_2 = %.2e ,Gamma_E = %.2e,Gamma_B = %.2e, Additional Dose = %.2f days\n",x0[1],x0[2],x0[3],x0[4],sigma,rN,rMmax,rMmin,tau,kN,kM,mu,rB,rEff0,alpha_1,alpha_2,Gamma_E,Gamma_B,firstdose)

        write(myfile,"time to event\tprogression or cure?\tparams\n--------------------------------------------------------------------------------------------------------------------------------------------\n")

        for n = 1 : nPatients

            (t,discretePop,continuousPop,pars) = runHybridModel(x0,initparams,sigma)

            if ~isempty(discretePop) && discretePop[4] == 0.0
                # println("cure at t = $(t[end])")
                results[n] = (t,"cure",pars)
                # curetime[n] = t[end]
                # push!(curetime,t[end])
            elseif discretePop[4] >= x0[4]*tumorSizeThreshold
                # println("progression at t = $(t[end])")
                results[n] = (t,"progression",pars)
                # progressiontime[n] = t[end]
                # push!(progressiontime,t[end])
            else
                results[n] = (t,"growing",pars)
                # println("Stable at t = 365.0")
            end


        end

        writedlm(myfile,results)

        close(myfile)

    end
    
end

# (t, discretePop, continuousPop,pars) = runHybridModel(x0,initparams,sigmaVal)

# if ~isempty(discretePop) && discretePop[4] == 0.0
#     # println("cure at t = $(t[end])")
#     results[n] = (t,"cure",pars)
#     # curetime[n] = t[end]
#     # push!(curetime,t[end])
# elseif discretePop[4] >= x0[4]*tumorSizeThreshold
#     # println("progression at t = $(t[end])")
#     results[n] = (t,"progression",pars)
#     # progressiontime[n] = t[end]
#     # push!(progressiontime,t[end])
# else
#     results[n] = (t,"growing",pars)
#     # println("Stable at t = 365.0")
# end