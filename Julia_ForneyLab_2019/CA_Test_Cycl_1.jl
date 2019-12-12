#---------------------------------------------------------------
# CyclingTime 3 - version 2

using ForneyLab


TravelTime_data = [ 13.0, 17.0, 16.0, 12.0, 13.0, 12.0, 14.0, 18.0, 16.0, 16.0,27.0, 32.0,27.0, 32.0,27.0, 32.0,27.0, 32.0,27.0, 32.0 ]
n=length(TravelTime_data)

g3 = FactorGraph()

# Specify generative model
@RV _pi ~ Beta(1.0, 1.0)
@RV AverageTime_1 ~ GaussianMeanVariance(15.0, 100.0)
@RV TrafficNoise_1 ~ Gamma(0.0001, 0.0001)
@RV AverageTime_2 ~ GaussianMeanVariance(30.0, 100.0)
@RV TrafficNoise_2 ~ Gamma(0.0001, 0.0001)

z = Vector{Variable}(undef, n)
TravelTime = Vector{Variable}(undef, n)
for i = 1:n
    @RV z[i] ~ Bernoulli(_pi)
    @RV TravelTime[i] ~ GaussianMixture(z[i], AverageTime_1, TrafficNoise_1, AverageTime_2, TrafficNoise_2)
    placeholder(TravelTime[i], :TravelTime, index=i)
end

q = RecognitionFactorization(_pi, AverageTime_1, TrafficNoise_1, AverageTime_2, TrafficNoise_2, z,
                                ids=[:PI, :AverageTime_1, :TrafficNoise_1, :AverageTime_2, :TrafficNoise_2, :Z])

# Generate the algorithm
algo = variationalAlgorithm(q);
algo_F = freeEnergyAlgorithm(q);

eval(Meta.parse(algo))
eval(Meta.parse(algo_F));

data = Dict(:TravelTime => TravelTime_data)

# Prepare recognition distributions
marginals = Dict(:_pi => vague(Beta),
                 :AverageTime_1 => vague(GaussianMeanVariance),
                 :TrafficNoise_1 => vague(Gamma),
                 :AverageTime_2 => vague(GaussianMeanVariance),
                 :TrafficNoise_2 => vague(Gamma))
for i = 1:n
    marginals[:z_*i] = vague(Bernoulli)
end

# Execute algorithm
n_its = 40
F = Float64[]
for i = 1:n_its
    stepZ!(data, marginals)
    stepPI!(data, marginals)
    stepAverageTime_1!(data, marginals)
    stepTrafficNoise_1!(data, marginals)
    stepAverageTime_2!(data, marginals)
    stepTrafficNoise_2!(data, marginals)
    push!(F, freeEnergy(data, marginals))
end

println(ForneyLab.unsafeMeanCov(marginals[:AverageTime_1]))
#(14.701329487009241, 0.465350841764408)
println(ForneyLab.unsafeMeanCov(marginals[:AverageTime_2]))
#(29.503198433790615, 0.6894080071931458)
print(marginals[:TrafficNoise_1]); println(ForneyLab.unsafeMean(marginals[:TrafficNoise_1]))
#Gam(a=5.00, b=23.38), 0.21389621254731633
print(marginals[:TrafficNoise_2]); println(ForneyLab.unsafeMean(marginals[:TrafficNoise_2]))
#Gam(a=5.00, b=34.71), 0.14404889767655651
println(ForneyLab.unsafeMean(marginals[:_pi]))
#0.4999902575945544
println(F)
#[122.893, 92.15, 92.095, 92.0546, 92.0246, 92.0018, 91.9839, 91.9693, 91.9568, 91.9458, 91.9355, 91.9256, 91.9155, 91.9049, 91.893, 91.8792, 91.8622, 91.8397, 91.8079,
#91.7591, 91.6765, 91.5188, 91.1724, 90.2878, 87.8902, 84.1353, 83.3133, 83.2899, 83.2896, 83.2896, 83.2896, 83.2896, 83.2896, 83.2896, 83.2896, 83.2896, 83.2896, 83.2896, 83.2896, 83.2896]

"""
Expected Results:
Average travel time distribution 1 = Gaussian(14.7, 0.3533)
Average travel time distribution 2 = Gaussian(29.51, 1.618)
Traffic noise distribution 1 = Gamma(7, 0.0403)[mean=0.2821]
Traffic noise distribution 2 = Gamma(3, 0.1013)[mean=0.304]
Mixing coefficient distribution = Dirichlet(11 3)
"""


#---------------------------------------------------------------
# CyclingTime 3 version 1

using ForneyLab

TravelTime_data = [ 13.0, 17.0, 16.0, 12.0, 13.0, 12.0, 14.0, 18.0, 16.0, 16.0,27.0, 32.0,27.0, 32.0,27.0, 32.0,27.0, 32.0,27.0, 32.0 ]
n=20

g3 = FactorGraph()

# Specify generative model
@RV _pi ~ Beta(1.0, 1.0)
@RV AverageTime_1 ~ GaussianMeanVariance(15.0, 100.0)
@RV TrafficNoise_1 ~ Gamma(2.0, 0.5)
@RV AverageTime_2 ~ GaussianMeanVariance(30.0, 100.0)
@RV TrafficNoise_2 ~ Gamma(2.0, 0.5)

z = Vector{Variable}(undef, n)
TravelTime = Vector{Variable}(undef, n)
for i = 1:n
    @RV z[i] ~ Bernoulli(_pi)
    @RV TravelTime[i] ~ GaussianMixture(z[i], AverageTime_1, TrafficNoise_1, AverageTime_2, TrafficNoise_2)
    placeholder(TravelTime[i], :TravelTime, index=i)
end

q = RecognitionFactorization(_pi, AverageTime_1, TrafficNoise_1, AverageTime_2, TrafficNoise_2, z,
                                ids=[:PI, :AverageTime_1, :TrafficNoise_1, :AverageTime_2, :TrafficNoise_2, :Z])

# Generate the algorithm
algo = variationalAlgorithm(q);
algo_F = freeEnergyAlgorithm(q);

eval(Meta.parse(algo))
eval(Meta.parse(algo_F));

data = Dict(:TravelTime => TravelTime_data)

# Prepare recognition distributions
marginals = Dict(:_pi => vague(Beta),
                 :AverageTime_1 => ProbabilityDistribution(Univariate, GaussianMeanVariance, m=-1.0, v=1e4),
                 :TrafficNoise_1 => vague(Gamma),
                 :AverageTime_2 => ProbabilityDistribution(Univariate, GaussianMeanVariance, m=1.0, v=1e4),
                 :TrafficNoise_2 => vague(Gamma))
for i = 1:n
    marginals[:z_*i] = vague(Bernoulli)
end

# Execute algorithm
n_its = n*2
F = Float64[]
for i = 1:n_its
    stepZ!(data, marginals)
    stepPI!(data, marginals)
    stepAverageTime_1!(data, marginals)
    stepTrafficNoise_1!(data, marginals)
    stepAverageTime_2!(data, marginals)
    stepTrafficNoise_2!(data, marginals)

    # Store variational free energy for visualization
    push!(F, freeEnergy(data, marginals))
end

mean(marginals[:AverageTime_1]), var(marginals[:AverageTime_1])
#(16.156251733553507, 0.08699525635593493)
mean(marginals[:AverageTime_2]), var(marginals[:AverageTime_2])
#(17.907081197841816, 3.5320629822501064)

mean(marginals[:TrafficNoise_1]), var(marginals[:TrafficNoise_1])
#(4.169853434502669, 5.143993801619434)
mean(marginals[:TrafficNoise_2]), var(marginals[:TrafficNoise_2])
#(0.029331249662152697, 0.00012996177209009975)
F #40-element Array{Float64,1}:

"""
Expected Results:
Average travel time distribution 1 = Gaussian(14.7, 0.3533)
Average travel time distribution 2 = Gaussian(29.51, 1.618)
Traffic noise distribution 1 = Gamma(7, 0.0403)[mean=0.2821]
Traffic noise distribution 2 = Gamma(3, 0.1013)[mean=0.304]
Mixing coefficient distribution = Dirichlet(11 3)
"""
