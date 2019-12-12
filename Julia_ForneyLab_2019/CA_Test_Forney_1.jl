### Coin flip simulation
#example 1
N = 25          # number of coin tosses
p = 0.75        # p parameter of the Bernoulli distribution
sbernoulli(n, p) = [(rand() < p) ? 1 : 0 for _ = 1:n] # define Bernoulli sampler
dataset = sbernoulli(N, p); # run N Bernoulli trials
print("dataset = ") ; show(dataset)
#dataset = [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]

using ForneyLab
g = FactorGraph()       # create a factor graph
a = placeholder(:a)     # define hyperparameter a as placeholder
b = placeholder(:b)     # define hyperparameter b as placeholder
@RV θ ~ Beta(a, b)      # prior
@RV y ~ Bernoulli(θ)    # likelihood
placeholder(y, :y)      # define y as a placeholder for data
draw(g)                 # draw the factor graph

# Generate a message passging sum-product algorithm that infers theta
algo_str = sumProductAlgorithm(θ) # ForneyLab returns the algorithm as a string
algorithm = Meta.parse(algo_str) # parse the algorithm into a Julia expression
eval(algorithm); # evaluate the functions contained in the Julia expression

# Create a marginals dictionary, and initialize hyperparameters
a = 2.0
b = 7.0
marginals = Dict(:θ => ProbabilityDistribution(Beta, a=a, b=b))

for i in 1:N
    # Feed in datapoints 1 at a time
    data = Dict(:y => dataset[i],
                :a => marginals[:θ].params[:a],
                :b => marginals[:θ].params[:b])

    step!(data, marginals)
end

using Plots, LaTeXStrings, SpecialFunctions; theme(:default)
pyplot(fillalpha=0.3, leg=false, xlabel=L"\theta", yticks=nothing)
BetaPDF(α, β) = x ->  x^(α-1)*(1-x)^(β-1)/beta(α, β) # beta distribution definition
BernoulliPDF(z, N) = θ -> θ^z*(1-θ)^(N-z) # Bernoulli distribution definition

rθ = range(0, 1, length=100)
p1 = plot(rθ, BetaPDF(a, b), title="Prior", fill=true, ylabel=L"P(\theta)", c=1,)
p2 = plot(rθ, BernoulliPDF(sum(dataset), N), title="Likelihood", fill=true, ylabel=L"P(D|\theta)", c=2)
p3 = plot(rθ, BetaPDF(marginals[:θ].params[:a], marginals[:θ].params[:b]), title="Posterior", fill=true, ylabel=L"P(\theta|D)", c=3)
plot(p1, p2, p3, layout=@layout([a; b; c]))


#---------------------------------------------------------------
#Variational Message Passing for Estimation
# Generate toy data set

using ForneyLab

n = 5
m_data = 3.0
w_data = 4.0
y_data = sqrt(1/w_data)*randn(n) .+ m_data

g1 = FactorGraph()

# Priors
@RV m ~ GaussianMeanVariance(0.0, 100.0)
@RV w ~ Gamma(0.01, 0.01)
#------

# Observarion model
y = Vector{Variable}(undef, n)

for i = 1:n
    @RV y[i] ~ GaussianMeanPrecision(m, w)
    placeholder(y[i], :y, index=i)
end

# Specify recognition factorization
q = RecognitionFactorization(m, w, ids=[:M, :W])

# Inspect the subgraph for m
#ForneyLab.draw(q.recognition_factors[:M])
# Inspect the subgraph for W
#ForneyLab.draw(q.recognition_factors[:W])

# Generate the variational update algorithms for each recognition factor
algo = variationalAlgorithm(q)

# And inspect the algorithm code
println(algo)

algo_F = freeEnergyAlgorithm(q)
println(algo_F)

# Load algorithms
eval(Meta.parse(algo))
eval(Meta.parse(algo_F))

data = Dict(:y => y_data)

# Initial recognition distributions
marginals = Dict(:m => vague(GaussianMeanVariance),:w => vague(Gamma))

n_its = 2*n
F = Vector{Float64}(undef, n_its) # Initialize vector for storing Free energy
m_est = Vector{Float64}(undef, n_its)
w_est = Vector{Float64}(undef, n_its)
for i = 1:n_its
    stepM!(data, marginals)
    stepW!(data, marginals)
    # Store free energy
    F[i] = freeEnergy(data, marginals)
end

using PyPlot, PyCall, SpecialFunctions
@pyimport matplotlib.lines as mlines

# Definition of the normal-gamma distribution
Z(μ₀, κ₀, α₀, β₀) = gamma(α₀)/(β₀^α₀)*(2*π/κ₀)^(0.5) # normalization constant
NG(rμ, rλ, D, μ₀, κ₀, α₀, β₀) = [ (1/Z(μ₀, κ₀, α₀, β₀))*λ^(α₀-0.5)*exp(-(λ/2)*((κ₀*(μ-μ₀)^2)+(2*β₀))) for μ=rμ, λ=rλ ]

# Calculates posterior distribution using exact inference
function exactPosterior(rμ, rλ, D, μ₀, κ₀, α₀, β₀)
    n = length(D)
    x̄ = sum(D) / n
    μₙ = (κ₀*μ₀ + n*x̄)/(κ₀ + n)
    κₙ = κ₀ + n
    αₙ = α₀ + n/2
    βₙ = β₀ + 0.5*var(D)*n + (κ₀*n*(x̄ - μ₀)^2)/(2*(κ₀+n))
    NG(rμ, rλ, D, μₙ, κₙ, αₙ, βₙ)
end

# Define the x and y limits for the contour plot
std_dev_ml = sqrt(var(y_data)/length(y_data))
rμ = range(mean(y_data)-3*std_dev_ml, mean(y_data)+3*std_dev_ml, length=1000)
rτ = range(0, 3(1/var(y_data)), length=1000)

# Plot the solution found using exact inference
contour([μ for μ=rμ, τ=rτ], [τ for μ=rμ, τ=rτ], exactPosterior(rμ, rτ, y_data, 0, 0.001, 0.01, 0.01), cmap="Greens")

# Generate a mesh grid of the approximated solution needed by the contour plot function
normal(x, μ, σ²) = (1/(sqrt(2π*σ²))) * exp.(-(x .- μ).^2 / (2*σ²)) # definition of the Gaussian distribution
Gam(λ, a, b) = (1/factorial(a-1)) * b^a * λ.^(a-1) .* exp.(-b*λ) # definition of the gamma distribution
approx(rμ, rτ, m_μ, v_μ, a_ω, b_ω) = [normal(μ, m_μ, v_μ) * Gam(τ, a_ω, b_ω) for μ=rμ, τ=rτ ]

# Plot the approximated solution found using VMP
contour([μ for μ=rμ, τ=rτ], [τ for μ=rμ, τ=rτ],
    approx(rμ, rτ, mean(marginals[:m]), var(marginals[:m]), marginals[:w].params[:a], marginals[:w].params[:b]), cmap="Reds")

# Add a legend, a grid and labels to the plot
green_patch = mlines.Line2D([], [], color="green", label="Exact solution")
red_patch = mlines.Line2D([], [], color="red", label="Approximated solution")
legend(handles=[green_patch, red_patch], loc="upper left"); grid(true); xlabel("Mean"); ylabel("Precision");

display(gcf())

# Plot free energy to check for convergence
plot(1:n_its, F, color="black", marker="o")
grid(true); xlabel("Iteration"); ylabel("F"); xlim([1, length(F)]);

display(gcf())

#---------------------------------------------------------------
# CyclingTime 1

using ForneyLab

#Simulation
TravelTimeMonObs= 13.0
TravelTimeTueObs= 17.0
TravelTimeWedObs= 16.0
#----------

#Model
g = FactorGraph()

#variance= (1/sigma^2)
fsigmaTovar(sigma)=1.0/(sigma^2.0)
fvarTosigma(var)=sqrt(1/var)


#
@RV AverageTime ~ GaussianMeanPrecision(15.0,0.01)
@RV TrafficNoise ~ Gamma(2.0, 0.5)
#-------

#Observarion model
@RV TravelTimeMon ~ GaussianMeanPrecision(AverageTime,TrafficNoise)
@RV TravelTimeTue ~ GaussianMeanPrecision(AverageTime,TrafficNoise)
@RV TravelTimeWed ~ GaussianMeanPrecision(AverageTime,TrafficNoise)

placeholder(TravelTimeMon,:TravelTimeMon)
placeholder(TravelTimeTue,:TravelTimeTue)
placeholder(TravelTimeWed,:TravelTimeWed)

#g.variables #Dict{Symbol,Variable} with 9 entries:
#ForneyLab.draw(g)
#g.variables[:AverageTime] #Variable(:AverageTime, Edges: ... )

#algo_str = sumProductAlgorithm([AverageTime,TrafficNoise])
#ERROR: ArgumentError: The input graph contains a loop around Interface 3 (w) of GaussianMeanPrecision gaussianmeanprecision_4

#Variational Option
# Specify recognition factorization
q = RecognitionFactorization(AverageTime, TrafficNoise, ids=[:AverageTime, :TrafficNoise])

# Generate the variational update algorithms for each recognition factor
algo = variationalAlgorithm(q)
println(algo)

algo_F = freeEnergyAlgorithm(q)
println(algo_F)

# Load algorithms
eval(Meta.parse(algo))
eval(Meta.parse(algo_F))

data = Dict(:TravelTimeMon => TravelTimeMonObs,:TravelTimeTue => TravelTimeTueObs,:TravelTimeWed => TravelTimeWedObs)

# Initial recognition distributions
#marginals = Dict(:AverageTime => vague(GaussianMeanPrecision),:TrafficNoise => vague(Gamma))
marginals = Dict(:AverageTime => ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=15.0, w=0.01),
            :TrafficNoise => ProbabilityDistribution(Univariate, Gamma, a=2.0, b=0.5))

n=1
n_its = n+1
F = Vector{Float64}(undef, n_its)
AverageTime_est = Vector{Float64}(undef, n_its)
TrafficNoise_est = Vector{Float64}(undef, n_its)
for i = 1:n_its
    stepAverageTime!(data, marginals)
    stepTrafficNoise!(data, marginals)
    # Store free energy
    F[i] = freeEnergy(data, marginals)
end

mean(marginals[:AverageTime]), var(marginals[:AverageTime])
#(15.331766690145983, 0.4699929562058515)
fvarTosigma(0.4699929562058515) #1.458660845346204

mean(marginals[:TrafficNoise]), var(marginals[:TrafficNoise])
#(0.6319598586512651, 0.11410664655615056)
F
#2-element Array{Float64,1}:11.117512842943897 11.112786743393944

"""
averageTimePosterior: Gaussian(15.33, 1.32)
trafficNoisePosterior: Gamma(2.242, 0.2445)[mean=0.5482]
"""
# Add a prediction variable and retrain the model
#Fail


#---------------------------------------------------------------
# CyclingTime 2

using ForneyLab

TravelTime_data = [ 13.0, 17.0, 16.0, 12.0, 13.0, 12.0, 14.0, 18.0, 16.0, 16.0 ]

#Model
g2 = FactorGraph()

#
@RV AverageTime ~ GaussianMeanPrecision(15.0,0.01)
@RV TrafficNoise ~ Gamma(2.0, 0.5)
#-------

# Observation model
n = 10
TravelTime = Vector{Variable}(undef, n)

for i = 1:n
    @RV TravelTime[i] ~ GaussianMeanPrecision(AverageTime,TrafficNoise)
    placeholder(TravelTime[i], :TravelTime, index=i)
end

data = Dict(:TravelTime => TravelTime_data)

marginals = Dict(:AverageTime => vague(GaussianMeanPrecision),:TrafficNoise => vague(Gamma))

q = RecognitionFactorization(AverageTime, TrafficNoise, ids=[:AverageTime, :TrafficNoise])

# Generate the variational update algorithms for each recognition factor
algo = variationalAlgorithm(q)
println(algo)

algo_F = freeEnergyAlgorithm(q)
println(algo_F)

# Load algorithms
eval(Meta.parse(algo))
eval(Meta.parse(algo_F))

n_its = n+1
F = Vector{Float64}(undef, n_its)
AverageTime_est = Vector{Float64}(undef, n_its)
TrafficNoise_est = Vector{Float64}(undef, n_its)
for i = 1:n_its
    stepAverageTime!(data, marginals)
    stepTrafficNoise!(data, marginals)
    # Store free energy
    F[i] = freeEnergy(data, marginals)
end

mean(marginals[:AverageTime]), var(marginals[:AverageTime])
#(14.70099107795069, 0.33035931689613646)
mean(marginals[:TrafficNoise]), var(marginals[:TrafficNoise])
#(0.30170071066722653, 0.013003331259587077)
F #20-element Array{Float64,1}:

"""
Average travel time = Gaussian(14.65, 0.4459)
Traffic noise = Gamma(5.33, 0.05399)[mean=0.2878]
"""

#---------------------------------------------------------------
# CyclingTime 3

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
