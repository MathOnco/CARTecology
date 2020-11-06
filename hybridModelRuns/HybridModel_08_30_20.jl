using Plots
using ProgressMeter
using DelimitedFiles
using Distributed
using Printf
using DifferentialEquations
using Dates
using Statistics
using Distributions
using StatsBase

macro name(arg)
    string(arg)
 end

function EmpiricalDistribution(data::Vector{T} where T <: Real)
    sort!(data) #sort the observations
    empirical_cdf = ecdf(data) #create empirical cdf
    data_clean = unique(data) #remove duplicates to avoid allunique error
    cdf_data = empirical_cdf.(data_clean) #apply ecdf to data
    pmf_data = vcat(cdf_data[1],diff(cdf_data)) #create pmf from the cdf
    DiscreteNonParametric(data_clean,pmf_data) #define distribution
end

function cumdist(x)

    y = Vector{Float64}(UndefInitializer(),length(x))
    for n = 1 : length(x)
        y[n] = sum(x[1:n])
    end

    y = y./y[end]

    return y
end

function F(u,simParams)

    (N,C,B) = u
    (rN, rC, kN,kC,k2, a,b, gammaB,rB) = simParams

    tumorBirth        = rB*B
    tumorDeath        = gammaB*B*k2*C/(1.0 + k2*C) 

    # @show (t[end],M,B)

    [tumorBirth,tumorDeath]
end

function CARmodel(du,u,p,t)

	(rN, rC, kN,kC,k2, a,b, gammaB,rB) = p
	(N, C, B) = u

	N = max(N,0)
	C = max(C,0)
	B = max(B,0)

	# Total lymphocyte in peripheral
	T = N + C

	# growth rate functions
	rNfunc = rN
	rCfunc = rC + b*(T - kN)^2/(a*T^2 + (T - kN)^2)
	# rCfunc = (rCmax - rCmin)*(kN - T + 0im)^a/((T+ 0im)^a + (kN - T+ 0im)^a) + rCmin
	
	

	# ODEs
	du[1] = -rNfunc*N*log(T/kN)
	du[2] = -rCfunc*C*log(T/kC)

	# If we are stochastic we will assume the population does not change
	# in the specified interval
	if isDiscrete
		du[3] = 0.0
	else
		du[3] = rB*B - gammaB*B*k2*C/(1.0 + k2*C)
	end

end

function continuousRegime(tspan,currentPop,p)

    callbackON = true
    if callbackON
        # Check to see if the tumor has entered the stochastic regime
        condition(u,t,integrator) = stochasticThreshold-u[end]
        affect!(integrator) = terminate!(integrator)
        cb1 = ContinuousCallback(condition,affect!,nothing,abstol=1e-11)

        # Check to see when the tumor exceeds some c*original size. We define 
        # this as progression.
        condition2(u,t,integrator) = u[end]-tumorSizeThreshold*x0[end]
        affect2!(integrator) = terminate!(integrator)
        cb2 = ContinuousCallback(condition2,affect2!)
        cbset = CallbackSet(cb1,cb2)
    else
        cb = nothing
    end
    prob = ODEProblem(CARmodel,currentPop,tspan,p)
    sol = solve(prob,Rodas4(autodiff=false),maxiters=1e11,force_dtmin=true,callback=cbset,abstol=1e-9,reltol=1e-6)

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

function runHybridModel(x0,initparams,sigma,stochasticThreshold)

    # Stochimetric matrix which determines outcome of the events 
    global nu = [[0 0 1];[0 0 -1]]

    # Perturb the simParams fed by sigma variation
    global simParams = max.(initparams.*(1 .+ sigma*randn(length(initparams))),0.0)

    # Force parameters that go negative to their initial input
    if any(simParams.<0.0)
        simParams[simParams.<0.0] = initparams[simParams.<0.0]
    end

    # Indicies which show which population is in the stochastic regime
    global isDiscrete = false

    # Time span on which to solve the continuous regime
    global tspan = (0.0,tfinal)

    if saveTrajectory
        global saveData = [];
        dirTrajectory = string(pwd(),"/trajectoryData/")
        filenameTrajectory = "trajectory"
        extensionTrajectory = ".dat"
        for n = 1 : 20000
            global TRAJECTORYFILE = string(dirTrajectory,filenameTrajectory,
            "_",string(n),"_",todaysDate,extensionTrajectory)
            if ~isfile(TRAJECTORYFILE)
                @printf("%s file created!\n",TRAJECTORYFILE)
                break
            end
        end
        trajectoryfile = open(TRAJECTORYFILE, "w")
    end

    # Initalize continuous and discrete population vectors
    global continuousPop    = float.(x0)
    global discretePop      = x0
    global tcurrent         = 0.0

    global discreteChange = 0*nu[1,:]

    while tcurrent < tfinal

        # Solve the deterministic system until tumor becomes stochastic or
        # the tumor goes past the threshold for "progression"
        (tnew,solContinuous) = continuousRegime(tspan,continuousPop,simParams)

        # update vector containing populations
        if isDiscrete
            solContinuous[:,end] += float(discreteChange)
            solContinuous[solContinuous[:,end].<1,end] .= 0.0
        end

        if saveTrajectory
            for i = 1 : length(tnew)
                append!(saveData,[tnew[i], solContinuous[1,i], solContinuous[2,i],solContinuous[3,i]])
            end
        end

        continuousPop = solContinuous[:,end]
        discretePop = ceil.(Int,continuousPop)
        tcurrent = tnew[end];

        # Check if tumor is in stochastic regime
        isDiscrete = continuousPop[end] - stochasticThreshold < 1e-11

        # @show continuousPop,discretePop

        # Check if tumor is extinct
        if discretePop[end] == 0
            break
        end

        # Check if progression has occurred
        if discretePop[end] - tumorSizeThreshold*x0[end] >= -1e-5
            break
        end

        # Update discrete population after time tcurrent + tau
        if isDiscrete
            (tau, discreteChange) = nextDiscreteEvent(discretePop,simParams,isDiscrete)
            
            # new continuous time span
            tspan = (tcurrent,min(tcurrent+tau,tfinal))
        else
            tspan = (tcurrent,tfinal)
            discreteChange = 0*discreteChange
        end

        # @show t,tau,discretePop,isDiscrete

    end
    if saveTrajectory
        writedlm(trajectoryfile,saveData)
        close(trajectoryfile)
    end

    return (tcurrent, discretePop, continuousPop, string(simParams))

end



nPatients = 10^3
V = 5.0*10^6        # Volume of blood in the body
VT = 10^9           # Average number of cancer cells per cm^3
freqInPeriph = 0.01

global saveTrajectory = true
global drawFromPatients = false

# Initialize final time
global tfinal = 2000.0  # Final time

# (rN, r1, kC, k1,k2, a,b, gammaB) = p
# (0.1699322195795346	0.751480113	105.8973269	0.338500554	0.616143356
# 3.94E-07	0.064263423	0.5459640682296819)

# [6.0;0.2;0.16;52.69] (cells/µL,cells/µL,cells/µL,cm) -> cells
rN      = 0.171         # WT memory growth rate
rC   	= 0.037;         # CAR growth rate
kN      = 500.0*V/freqInPeriph;      # WT memory carrying capacity
kC      = 139.28*V/freqInPeriph;     # CAR memory carrying capacity
##### Assymmetric differentiation rate ##########
# rEff(B) = rEff(0)*(1 + exp(-a1*B))
k2 		= 0.247/V*freqInPeriph; #0.247
a 		= 0.423;
b 		= 0.525;
####################################################
rB      = 0.065; # 0.21          # Tumor birth rate
# Pair-wise interaction rates
gammaB = 1.15 #0.735; # Tumor-CAR interaction (tumor dies)

# Initial condition and parameters (which can be perturbed if sigma )
sigmavec = [0.05] #[0.0;0.05;0.1]
B0prefactorvec = [1.0] #(0.5:0.5:20) #[0.01;0.1;1.0;2.0;5.0;10.0]
rBvec = [0.065]#(0.065:0.025:0.29) #(0.04:0.01:0.19)#[0.065] #(0.04:0.025:0.19)
# k2vec = []
WTinitvec = (1.0:1.0:12.0)#[6.0;12.0;18.0;24.0;30.0]
kCvec = (100.0:10.0:200.0)*V/freqInPeriph
gammaBvec = (0.15:0.15:1.15)

B0patientList = [4.24
9.21
23.84
37.33
58.27
94.62
132.83
162.2
169.56
170.31
182.51
210.34
225.13
226.52
290.78
324.47
535.53
411.44
499.6
536.3
619.2
659.4
10.3
47.84
4.5
20.85
9.01
8.13
42.73
33.91
49.02
17.38
604.08
533.68
534.23
1160.94
44.21
771.06
342.9
6.1
36.38
770.77
1221.39
184.31
51.55
68.74
44.72
16.84
59.34
10.13
1275.3
9.57
372.58
52.86
27.62
2.3
120.31
95.06
482.95
41.27
454.69
10.71
115.44
403.72
2.35
175.52
66.87
2.67
118.41
748.34
78.8
199.06
312.78
11.03
617.76
325.35
65.36
7.93
179.63
3.56
312.84
259.7
864.43
2.36
11.36
10.85
6.35
643.74
48.27
20.15
382.53
131.98
28.03
10.85
391.45
57.99 ];

B0prefactorvec = quantile!(B0patientList,[0.0,0.25,0.5,0.75,1.0])

if drawFromPatients
	tumorSizeDist = EmpiricalDistribution(B0patientList)
end

# Threshold at which stochastic fluctuations are considered
# stochasticThresholdvec = [(10:10:90);(100:100:1000)]

sigmaVal = 0.05;
WTinit = 6.0;
B0prefactor = 1.0

stochasticThresholdval = 100

CAR0 = 0.36
B0   = 1.0

# global stochasticThreshold = 100;
tumorSizeThresholdval = 1.2

todaysDate = Dates.today()

cd(dirname(@__FILE__))
println(pwd()) 

for rB in rBvec
	for B0prefactor in B0prefactorvec

        initWT     = WTinit*V/freqInPeriph                          # 6.0*V
        initCAR = CAR0*V/freqInPeriph           # 0.2*V
        initTumor  = B0prefactor*B0*VT                       # 52.69*VT
        global x0 = [initWT;initCAR;initTumor]#[50,10,40,50]
        global initparams = [rN, rC, kN,kC,k2, a,b, gammaB,rB]

        # Initialize results vector
        global results = Array{Tuple{Float64,String,String}}(undef,nPatients)

        # Introduce param noise
        global sigma = sigmaVal

        global stochasticThreshold = stochasticThresholdval
        global tumorSizeThreshold = tumorSizeThresholdval

        directory = string(pwd(),"/runData/")
        filename = "outputData"
        extension = ".txt"

        for n = 1 : 2000
            global FILE = string(directory,filename,"_",string(n),"_",todaysDate,extension)
            if ~isfile(FILE)
                @printf("%s file created!\n",FILE)
                break
            end
        end
        myfile = open(FILE, "w")

        header = @printf(myfile,"Initial conditions [N0,C0,B0] = [%.3e,%.3e,%.3e] with main parameters used to perturb with sigma = %.2e: rN = %.2e, rC = %.2e, kN = %.2e, kC = %.2e, rB = %.2e, k2 = %.2e, a = %.2e, b = %.2e, Gamma_B = %.2e, stochastic threshold = %.0f\n",x0[1],x0[2],x0[3],sigma,rN, rC, kN,kC, rB, k2, a,b, gammaB, stochasticThreshold)

        write(myfile,"time to event\tprogression or cure?\tparams\n--------------------------------------------------------------------------------------------------------------------------------------------\n")

		for n = 1 : nPatients

			if drawFromPatients
				x0[end] = rand(tumorSizeDist)
			end

			(tfin,discreteOut,continuousOut,pars) = runHybridModel(x0,initparams,sigma,stochasticThreshold)

            if ~isempty(discreteOut) && discreteOut[end] == 0.0
                # println("cure at t = $(t[end])")
                results[n] = (tfin,"cure",pars)
                # curetime[n] = t[end]
                # push!(curetime,t[end])
            elseif discreteOut[end] >= x0[end]*tumorSizeThreshold
                # println("progression at t = $(t[end])")
                results[n] = (tfin,"progression",pars)
                # progressiontime[n] = t[end]
                # push!(progressiontime,t[end])
            else
                results[n] = (tfin,"growing",pars)
                # println("Stable at t = 365.0")
            end


        end

        writedlm(myfile,results)

        close(myfile)

    end
    
end
