using DifferentialEquations, Optim,  Measurements

H_data =  [20000,   50000,  80000, 110000, 140000]
P5_data = [measurement(264410.0, 10000.0), measurement(257132.0, 10000.0), measurement(251466.0, 10000.0), measurement(218014.0, 10000.0), measurement(199986.0, 10000.0)]

function P5(H, r, K)  # fish population at end of 5 years
  dudt(u, (r, K, H), t) = u > measurement(0.01,0.01) ? r * u * (K - u) - H : measurement(0.0,0.0)
  prob = ODEProblem(dudt, measurement(200000.0, 10000.0), (0.0, 5.0), (; r, K, H))
  sol = solve(prob, Tsit5())
  max(sol[end][1])
end

# loss function uses scaled versions of r and K
loss(x) = sum(abs2, P5(H, x[1]*1e-5, x[2]*1e5) - y for (H, y) ∈ zip(H_data, P5_data))

# optimize r and K using forward-mode AD
sol = optimize(loss, [measurement(1.5,0.0057),measurement(2.0,0.0076)] )   # , LBFGS(); autodiff=:forward
r, K = Optim.minimizer(sol) .* [1e-5, 1e5]    # rescale r and K

# plot output
#plot(H -> P5(H, r, K), 0, 300000; legend=false, xlabel="H", ylabel="P5")

using ForwardDiff
using Optim
using MonteCarloMeasurements, Distributions
using DifferentialEquations , Turing
dudt(u, (r, K, H), t) = u > 0.0 ? r * u * (K - u) - H : 0.0

r_dist = Normal(8.992e-6 , 3.5e-8 )
K_dist = Normal(278000.0 , 1000.0)
P_dist = Normal(200000.0, 10000.0)
HH = 140000.0
count_iter = 0
inrange_stock = []

while true
    stock = []
    for i =1:2000
        r = rand(r_dist)
        K = rand(K_dist)
        P0 = rand(P_dist)
        #println(r, K, P0)
        prob = ODEProblem(dudt, P0, (0.0, 5.0), [r,K,HH])
        sol = solve(prob, Tsit5())
        final_population = sol[end][1]
        push!(stock,final_population)
        # println(length(stock))
    end
    
    below_count = count(x -> x < 100000.0, stock)
    #println(below_count)
    
    if below_count > 200  
    # 90% probability that end stock more than 100000
        break
    end
    
    HH += 500
    count_iter +=1
    inrange_stock = stock
end
println( "The Maximum allowable harvest:", HH-500 )
println("iter times:", count_iter-1 )

using Statistics # esitmate gaussian
using Plots
μ = mean(inrange_stock)
σ = std(inrange_stock)

# generate normal distribution x
x = range(minimum(inrange_stock), stop=maximum(inrange_stock), length=count_iter )

# conut normal y
y = pdf.(Normal(μ, σ), x)

# plot
histogram(inrange_stock, normalize = true , xlabel="Stock value ", ylabel="Frequency", title="Stock value distribution" , label ="true distribution")
plot!(x, y, label="Gaussian fit", linewidth=2, color=:red , size=(800, 500) )
xticks!(collect( 1e4:10000.0:1.8e5 ) )





