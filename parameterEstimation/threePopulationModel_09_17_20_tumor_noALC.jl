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
using CSV
using DataFrames
# using Statistics

function CARmodel(du,u,p,t)
   
   (rN, rC, kC, k1, k2, rC0, gammaB) = p
   (N, C, B) = u

    N = max(N,0)
    C = max(C,0)

	# Total lymphocyte in peripheral
    T = N + C

    # growth rate functions
	rNfunc = rN
	rCfunc = rC*k1*B/(1.0 + k1*B) + rC0
    # rCfunc = (rCmax - rCmin)*(kN - T + 0im)^a/((T+ 0im)^a + (kN - T+ 0im)^a) + rCmin
    
    # ODEs
    du[1] = -rNfunc*N*log(T/kN)
	du[2] = -rCfunc*C*log(T/kC)
    du[3] = rB*B*(1.0 - B/PD) - gammaB*B*k2*C/(1.0 + k2*C)

end

function my_loss_function(sol,params)
    if maximum(sol.t) < t[end] + 1
        # @warn "max time is $(maximum(sol.t))"
        return 1e10
    end

    tot_loss = 0.0

    (rN, rC, kC, k1, k2, rC0, gammaB) = params;

    tot_loss += lambda0*sum(skipmissing([(CARpatient[n] - sol(t[n])[2])^2 for n in 1:nT ]))/maxCARpatient^2

	# We introduce L1 loss except for the kC term
    tot_loss += lambda1*(sum(params) - kC)

    tot_loss += lambda2*sum(skipmissing([(Ndata[n] - sol(tALC[n])[1])^2 for n in 1:nTALC ]))/maxN^2

	# for n in 1:length(tALC)
	# 	T = sol(tALC[n])[1] + sol(tALC[n])[2] 
    #     tot_loss += lambda2*(T - Ndata[n])^2/maxN^2
	# end
	
	tot_loss += lambda3*sum([(tumorData[n] - sol(tTumor[n])[3])^2 for n in 1:length(tTumor)])/B0^2

    # @show tot_loss
    
   return tot_loss

end

function cost_function2(params, earlyterminate=false)

    prob = ODEProblem(CARmodel,u0,tspan,params)
    cb = nothing
    sol = solve(prob,Rodas4(autodiff=false),maxiters=1e11,force_dtmin=true,callback=cb,abstol=1e-11,reltol=1e-8)

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

    result = bboptimize(cost_function2;SearchRange = bounds, MaxSteps = 1e6,Method=:adaptive_de_rand_1_bin_radiuslimited,TraceMode=:compact)

    # Potential methods:
    # adaptive_de_rand_1_bin_radiuslimited, de_rand_1_bin

    return result

end

function plotsol(CARmodel, result)
    p_opt = result
    prob = ODEProblem(CARmodel,u0,tspan,p_opt)
    cb = nothing
	sol = solve(prob,Rodas4(autodiff=false),maxiters=1e9,force_dtmin=true,callback=cb,abstol=1e-11,reltol=1e-8)

    if any(convert(Array,sol) .<= 0)
        fig = plot(sol)
    else
        fig = plot(sol,yaxis=:log)
        ylims!(1e-5,1e3)
    end 

    # fig = plot(sol,label=["Wildtype","CAR memory"],yaxis=:log,legend=:bottomleft)
    # xlims!((0.0,200.0))
    # ylims!((1e-8,1.1*max(maxCAR,maxCARpatient,maximum(Wildtypedata))))
    scatter!(fig,t,CARpatient,label=L"CAR_{data}")
	scatter!(fig,tALC,Ndata,label=L"Normal_{data}")
	scatter!(fig,tTumor,tumorData,label=L"Tumor_{data}")

    return fig
end

# Parameters
kN = 500.0
rB = 0.065
# rN = 0.16
runlocal = false
nLocalRuns = 10
ploton = true
useMedian = true

# Optimization parameters
lambda0 = 1e1       # Weight of CAR data
lambda1 = 1e-2     # rho > C + D
lambda2 = 1e0      # Weight of the wildtype data
lambda3 = 1e6
Nparams = 7
# Parameters: [rN, rC, kC, k1, k2, rC0, gammaB]
lower = 1e-3*ones(Nparams)
# lower[4] = 0.01
# lower[4] = 0.2
# lower[5] = 0.05;
upper = 5.0*ones(Nparams)
upper[end] = 0.3
bounds = repeat([(0.0,5.0)],Nparams)
bounds[3] = (10.0,kN)
# bounds[end] = (0.0, 0.3)
# bounds[5] = (0.25,3.0)
opts = Optim.Options(iterations=Int(1e4),show_trace=true);

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
# t = [7;14;28;90;180;270;360;450;540;720]


t = [7;14;28;90;180]

nT = length(t)

if !useMedian
	CARdf = CSV.read(string(pwd(),"/z1_CAR_T_by_visit_data_C1C2C3.csv"),header=true)
	nPatients = nrow(CARdf)
else
    CARQ1 = [16.8163475,3.43142355,0.46515,0.0116823,0.00379764]
    CARmedian = [31.55036408,9.05121,2.12625,0.1123343,0.10444368]
    CARQ3 = [58.84667075,35.7016615,4.74985989,0.431325,0.40064941]
    
    CARdf = (CARQ1,CARmedian,CARQ3)
	nPatients = 3
end



# Median data
# CARdata = [31.55036408,9.05121,2.12625,0.1123343,0.10444368]

# Median wildtype data
tALC = [7;14;28;90;180]

ntALC = length(tALC)

ALC = 10^3*[0.16;0.28;0.48;0.49;0.47]

tspan = (0.0,200.0)

# Initial wildtype (cells/µL), CAR (cells/µL) and tumor (mL)
w0 = 6.0
CAR0 = 0.36
B0 = 94.84

# Define CR, PR, SD, PD
(CR,PR,SD,PD, PDafterCR) = (1e-7, 0.5*B0, 1.0*B0, 50.0*B0, 1.0)

# tumor outcomes at specified times
# tTumor = [30;90;180];
# tumorData = [CR; PR; SD]

tTumor = [30];
tumorData = [CR]

global results = Array{Tuple{Array{Any,1},Array{Float64,1},Array{Float64,1},Float64}}(undef,nPatients)

directory = string(pwd(),"/twoPopulationModelData/");
filename = "outputData_Tumor_noALC_dependence"
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

write(myparamfile,"#rN, rC, kC, k1, k2, rC0, gammaB\n")

flush(myfile)
flush(myparamfile)

# We run the optimizer over three possible memory fractions and 3^5 = 243
# time points
global counter = 0;

for m = 1 : nPatients

	if !useMedian
		global CARpatient = convert(Array,CARdf[m,4:8])

		# For now we replace zeros by small numbers (1e-5) to avoid errors in plotting
		# the fits should be unaffected
		CARpatient[findall(<=(0.0),skipmissing(CARpatient))] .= 1e-5
	else
		global CARpatient = CARdf[m]
	end

    # Subtract CAR levels from ALC
    global Ndata = ALC .- CARpatient

    global maxCARpatient = maximum(skipmissing(CARpatient))
    global maxN = maximum(Ndata)


    global u0 = [w0;CAR0;B0]

    # global CARpatient /= maximum(CARpatient)
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

    println(CARpatient,'\t',u0,'\t',bestparams,'\t',bestmin)
    global counter += 1
    results[counter] = (CARpatient,u0,bestparams,bestmin)

    writedlm(myparamfile,bestparams')
    flush(myparamfile)

    if ploton
        fig = plotsol(CARmodel, bestparams)
		
		for n = 1 : 1000
			global pdffilename = string(directory,"TwoPopulation_",string(n),"_",todaysDate,".pdf")
			if ~isfile(pdffilename)
				@printf("%s file created!\n",pdffilename)
				break
			end
		end

        savefig(fig,pdffilename)
    end

end

writedlm(myfile,results)

close(myfile)