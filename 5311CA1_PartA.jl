import Zygote
using ForwardDiff
using DifferentialEquations
using Optim

#### step 1: estimate r,K  ###

##  loss function
function loss_function(params)
    r, K = params
    losses = Float64[]
    dudt(u,(r,K,H),t) = r * u * (K - u) - H  # fishery diff function

    for i in 1:5
        u0 = [241725, 228076, 212319, 193015, 165744][i]
        H = [20000, 50000, 80000, 110000, 140000][i]
        prob = ODEProblem(dudt, u0, (0.0, 5.0), [r, K, H])
        sol = solve(prob)
        predicted_population = sol[end][1]   # get predicted P from diff function
        loss = (predicted_population - u0)^2  # compute loss one by one
        push!(losses, loss)
    end
    # 返回总损失
    return sum(losses)
end

# initializa for r ,K
initial_guess = [0.1, 100000.0]  

# minimize loss
result = optimize(loss_function, initial_guess)

# optimization
optimal_params = Optim.minimizer(result)
println("Optimal r and K: ", optimal_params)

### step2: estimate the largest H

r = optimal_params[1]  # 出生率
K = optimal_params[2]  # 最大承载量
H0 = 140000.0  # 每年的捕捞量

while true
    dudt(u,(r,K,H),t) = r * u * (K - u) - H  # fishery diff function
    prob = ODEProblem(dudt, 165744.0, (0.0, 5.0), [r,K,H0])
    sol = solve(prob)
    final_population = sol[end][1]
    println("Final fish population after 5 years: ", final_population)
    if final_population<= 100000
        break
    end

    H0 += 1000
    
end
println(H)