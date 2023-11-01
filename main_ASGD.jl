using LinearAlgebra, Random, Statistics
using Lux, Zygote
using CSV, DataFrames, JLD2
using Optim
using Plots
cd(@__DIR__)

function construct_FCNN(N_layer::Vector{Int64})
    return Lux.Chain(
        [Dense(fan_in => fan_out, tanh, use_bias=false) for (fan_in, fan_out) in zip(N_layer[1:end-2], N_layer[2:end-1])]...,
        Dense(N_layer[end-1] => N_layer[end], identity, use_bias=false),
    )
end

loss(model, ps, ls, x::Matrix{T}, y::Matrix{T}) where {T<:AbstractFloat} = sum(abs2, first(model(x, ps, ls)) .- y) / Float32(2 * size(x, 2))

exp_S(t, ps, norm_ps, grad) = NamedTuple{keys(ps)}([(; weight=cos(t) * ps[i].weight - sin(t) * norm_ps[i] * grad[i].weight) for i in eachindex(ps)])

function ASGD_AD!(model, ps, norm_ps, ls, L_prev::T, x::Matrix{T}, y::Matrix{T}; ϵ=T(pi / 6.0)) where {T<:AbstractFloat}
    # (L_prev, (grad,)) = Zygote.withgradient(p -> loss(model, p, ls, x, y), ps)
    grad = Zygote.gradient(p -> loss(model, p, ls, x, y), ps)[1]
    neg_dL1 = sum([sqrt(max(abs2(norm_ps[i]) * dot(grad[i].weight, grad[i].weight) - abs2(dot(ps[i].weight, grad[i].weight)), zero(T))) for i in eachindex(ps)])
    for i in eachindex(ps)
        axpy!(-dot(ps[i].weight, grad[i].weight) / abs2(norm_ps[i]), ps[i].weight, grad[i].weight)
        normalize!(grad[i].weight)
    end

    dL2 = Zygote.hessian(t -> loss(model, exp_S(t, ps, norm_ps, grad), ls, x, y), zero(T))
    if dL2 != zero(T)
        t_cr = neg_dL1 / dL2
        if t_cr >= zero(T) && t_cr <= ϵ
            τ = t_cr
        else
            τ = ϵ
        end
    else
        if neg_dL1 == zero(T)
            τ = zero(T)
        else
            τ = ϵ
        end
    end

    # t_probe = Float32.(0:1e-3:1)
    # L_probe = [loss(model, exp_S(t, ps, norm_ps, grad), ls, x, y) for t in t_probe]
    # f_probe = plot(t_probe, L_probe, xlabel="\$\\tau\$", label=false)
    # plot!(f_probe, t_probe, L_probe[1] .- neg_dL1 .* t_probe .+ T(0.5) .* dL2 .* t_probe.^2, label=false, linestyle=:dash)
    # scatter!(f_probe, [τ], [loss(model, exp_S(τ, ps, norm_ps, grad), ls, x, y)], label="\$\\tau\$", color="red")
    # display(f_probe)

    L_τ = loss(model, exp_S(τ, ps, norm_ps, grad), ls, x, y)
    while L_τ > L_prev
        τ /= T(2)
        L_τ = loss(model, exp_S(τ, ps, norm_ps, grad), ls, x, y)
    end

    for i in eachindex(ps)
        axpby!(-norm_ps[i] * sin(τ), grad[i].weight, cos(τ), ps[i].weight)
        # normalize!(ps[i].weight)
        # lmul!(norm_ps[i], ps[i].weight)
    end
    return L_τ, τ
end

# major_sin(t, α, β, P, Q, L) = Float32(α * (cos(t) - 1.0) - β * sin(t) + Q / 2.0 * abs2(P * ((1.0 + 2.0 * sin(t / 2.0))^L - 1.0)))

# function ASGD_MM_sin!(model, ps, norm_ps, P, Q, L, ls, x::Matrix{T}, y::Matrix{T}) where {T<:AbstractFloat}
#     # (L_prev, (grad,)) = Zygote.withgradient(p -> loss(model, p, ls, x, y), ps)
#     grad = Zygote.gradient(p -> loss(model, p, ls, x, y), ps)[1]
#     α = sum([dot(ps[i].weight, grad[i].weight) for i in eachindex(ps)])
#     neg_dL1 = sum([sqrt(max(abs2(norm_ps[i]) * dot(grad[i].weight, grad[i].weight) - abs2(dot(ps[i].weight, grad[i].weight)), zero(T))) for i in eachindex(ps)])
#     for i in eachindex(ps)
#         axpy!(-dot(ps[i].weight, grad[i].weight) / abs2(norm_ps[i]), ps[i].weight, grad[i].weight)
#         normalize!(grad[i].weight)
#     end

#     res = Optim.optimize(t -> major_sin(t, α, neg_dL1, P, Q, L), zero(T), T(2 * pi), Brent())
#     τ = Optim.minimizer(res)

#     for i in eachindex(ps)
#         axpby!(-norm_ps[i] * sin(τ), grad[i].weight, cos(τ), ps[i].weight)
#         # normalize!(ps[i].weight)
#         # lmul!(norm_ps[i], ps[i].weight)
#     end

#     return loss(model, exp_S(τ, ps, norm_ps, grad), ls, x, y)
# end

# function major_sin_op2(t, α, β, Q, L, ps, norm_ps, grad)
#     X = zeros(L)
#     for i in 1:L
#         if i == L
#             X1 = 1.0f0
#         elseif i < L
#             X1 = prod([opnorm(ps[j].weight * cos(t) - norm_ps[j] * grad[j].weight * sin(t)) for j in (i+1):L])
#         end
#         X2 = opnorm(ps[i].weight * (cos(t) - 1.0f0) - norm_ps[i] * grad[i].weight * sin(t))
#         if i == 1
#             X3 = 1.0f0
#         elseif i > 1
#             X3 = prod([opnorm(ps[j].weight) for j in 1:(i-1)])
#         end
#         X[i] = X1 * X2 * X3
#     end
#     P3 = sum(X)

#     return Float32(α * (cos(t) - 1.0f0) - β * sin(t) + Q / 2.0f0 * P3^2)
# end

function major_sin_op(t, α, β, Q, ps, norm_ps, grad)
    P1 = prod([opnorm(ps[i].weight) + opnorm(ps[i].weight * (cos(t) - 1.0f0) - norm_ps[i] * grad[i].weight * sin(t)) for i in eachindex(ps)])
    P2 = prod([opnorm(ps[i].weight) for i in eachindex(ps)])
    return α * (cos(t) - 1.0f0) - β * sin(t) + Q / 2.0f0 * (P1 - P2)^2
end

function ASGD_MM_sin!(model, ps, norm_ps, Q, L, ls, x::Matrix{T}, y::Matrix{T}) where {T<:AbstractFloat}
    # (L_prev, (grad,)) = Zygote.withgradient(p -> loss(model, p, ls, x, y), ps)
    grad = Zygote.gradient(p -> loss(model, p, ls, x, y), ps)[1]
    α = sum([dot(ps[i].weight, grad[i].weight) for i in eachindex(ps)])
    neg_dL1 = sum([sqrt(max(abs2(norm_ps[i]) * dot(grad[i].weight, grad[i].weight) - abs2(dot(ps[i].weight, grad[i].weight)), zero(T))) for i in eachindex(ps)])
    for i in eachindex(ps)
        axpy!(-dot(ps[i].weight, grad[i].weight) / abs2(norm_ps[i]), ps[i].weight, grad[i].weight)
        normalize!(grad[i].weight)
    end

    res = Optim.optimize(t -> major_sin_op(t, α, neg_dL1, Q, ps, norm_ps, grad), 0.0f0, Float32(pi), GoldenSection())
    τ = Optim.minimizer(res)

    for i in eachindex(ps)
        axpby!(-norm_ps[i] * sin(τ), grad[i].weight, cos(τ), ps[i].weight)
        # normalize!(ps[i].weight)
        # lmul!(norm_ps[i], ps[i].weight)
    end

    return loss(model, exp_S(τ, ps, norm_ps, grad), ls, x, y), τ
end

function ASGD_MM_exp!(model, ps, norm_ps, P, Q, L, ls, x::Matrix{T}, y::Matrix{T}) where {T<:AbstractFloat}
    # (L_prev, (grad,)) = Zygote.withgradient(p -> loss(model, p, ls, x, y), ps)
    grad = Zygote.gradient(p -> loss(model, p, ls, x, y), ps)[1]
    neg_dL1 = sum([sqrt(max(abs2(norm_ps[i]) * dot(grad[i].weight, grad[i].weight) - abs2(dot(ps[i].weight, grad[i].weight)), zero(T))) for i in eachindex(ps)])
    for i in eachindex(ps)
        axpy!(-dot(ps[i].weight, grad[i].weight) / abs2(norm_ps[i]), ps[i].weight, grad[i].weight)
        normalize!(grad[i].weight)
    end

    τ = T(log((1.0 + sqrt(1.0 + (4.0 * neg_dL1) / (P^2 * Q * L))) / 2.0) / L)

    for i in eachindex(ps)
        axpby!(-norm_ps[i] * sin(τ), grad[i].weight, cos(τ), ps[i].weight)
        # normalize!(ps[i].weight)
        # lmul!(norm_ps[i], ps[i].weight)
    end

    return loss(model, exp_S(τ, ps, norm_ps, grad), ls, x, y), τ
end

function main(; seed=1, N_layer=[12, 25, 30, 15, 3], N_epochs=100, method=1, flag_plot=1)
    rng = Xoshiro(seed)

    data = CSV.read("neural_lander_data.csv", DataFrame) |> Matrix
    t = Float32.(vec(data[:, 1]))
    # z = data[:, 2]
    # v = data[:, 3:5]
    # q = data[:, 6:9]
    # u = data[:, 10:13]
    # Fa = data[:, 14:16]
    id_perm = randperm(rng, size(data, 1))
    id_train = id_perm[1:round(Int, 0.8 * length(id_perm))]
    id_test = id_perm[round(Int, 0.8 * length(id_perm))+1:end]
    # id_train = 1:round(Int, 0.8 * size(data, 1))
    # id_test = id_train[end]+1:size(data, 1)
    x_train = Float32.(data[id_train, 2:13]')
    y_train = Float32.(data[id_train, 14:16]')
    x_test = Float32.(data[id_test, 2:13]')
    y_test = Float32.(data[id_test, 14:16]')

    model = Lux.Chain(
        [Dense(fan_in => fan_out, Lux.relu, use_bias=false) for (fan_in, fan_out) in zip(N_layer[1:end-2], N_layer[2:end-1])]...,
        Dense(N_layer[end-1] => N_layer[end], identity, use_bias=false),
    )

    ps, ls = Lux.setup(rng, model)
    norm_ps_val = fill(Float32(1), length(N_layer) - 1)
    # norm_ps_val = [Float32(sqrt(size(ps[i].weight,1) / size(ps[i].weight,2))) for i in eachindex(ps)]
    # norm_ps_val = [Float32(sqrt(prod(size(ps[i].weight))) / 3.0) for i in eachindex(ps)]
    norm_ps = NamedTuple{keys(ps)}(norm_ps_val)
    for i in eachindex(ps)
        normalize!(ps[i].weight)
        lmul!(norm_ps[i], ps[i].weight)
    end
    P = prod(norm_ps_val)
    P_op = prod([opnorm(ps[i].weight) for i in eachindex(ps)])
    y_train_norm_max = maximum([norm(y_train[:, i]) for i in 1:length(id_train)])
    for i in 1:length(id_train)
        x_train[:, i] /= norm(x_train[:, i])
        y_train[:, i] *= P_op / y_train_norm_max
    end
    for i in 1:length(id_test)
        x_test[:, i] /= norm(x_test[:, i])
        y_test[:, i] *= P_op / y_train_norm_max
    end
    Q = Float32(sum([dot(x_train[:, i], x_train[:, i]) for i in 1:length(id_train)]) / length(id_train))
    L = length(N_layer) - 1

    L_prev = loss(model, ps, ls, x_train, y_train)
    loss_train_history = []
    loss_test_history = []
    τ_history = []
    for epoch in 1:N_epochs
        if method == 1
            (L_current, τ_current) = ASGD_AD!(model, ps, norm_ps, ls, L_prev, x_train, y_train; ϵ=Float32(pi / 6.0))
        elseif method == 2
            (L_current, τ_current) = ASGD_MM_sin!(model, ps, norm_ps, Q, L, ls, x_train, y_train)
        elseif method == 3
            (L_current, τ_current) = ASGD_MM_exp!(model, ps, norm_ps, P, Q, L, ls, x_train, y_train)
        end

        push!(loss_train_history, L_current)
        push!(loss_test_history, loss(model, ps, ls, x_test, y_test))
        push!(τ_history, τ_current)
        if epoch % 100 == 0
            println("Epoch: $epoch, Loss (Training): $(loss_train_history[end])")
        end
        L_prev = L_current
    end

    RMSE_final_train = sqrt(2.0f0 * loss_train_history[end]) * y_train_norm_max / P_op
    RMSE_final_test = sqrt(2.0f0 * loss_test_history[end]) * y_train_norm_max / P_op

    if flag_plot == 1
        f_loss = plot(1:N_epochs, loss_train_history, xlabel="epoch", ylabel="\$\\mathcal{L}\$", label=false)
        display(f_loss)

        f_y_train = plot(t[id_train], y_train', xlabel="\$t\$ [s]", ylabel=["\$F_{a_{x}}\$" "\$F_{a_{y}}\$" "\$F_{a_{z}}\$"], label=[false false "\$y\$ train"], legend=:topleft, layout=(3, 1), size=(600, 600))
        plot!(f_y_train, t[id_train], model(x_train, ps, ls)[1]', label=[false false "\$\\hat{y}\$ train"])
        display(f_y_train)

        f_y_test = plot(t[id_test], y_test', xlabel="\$t\$ [s]", ylabel=["\$F_{a_{x}}\$" "\$F_{a_{y}}\$" "\$F_{a_{z}}\$"], label=[false false "\$y\$ test"], layout=(3, 1), size=(600, 600))
        plot!(f_y_test, t[id_test], model(x_test, ps, ls)[1]', label=[false false "\$\\hat{y}\$ test"])
        display(f_y_test)
    end

    # grad = Zygote.gradient(p -> loss(model, p, ls, x_train, y_train), ps)[1]
    # for i in eachindex(ps)
    #     axpy!(-dot(ps[i].weight, grad[i].weight), ps[i].weight, grad[i].weight)
    #     normalize!(grad[i].weight)
    # end
    # t_probe = Float32.(0:1e-3:1)
    # L_probe = [loss(model, exp_S(t, ps, grad), ls, x_train, y_train) for t in t_probe]
    # f_probe = plot(t_probe, L_probe, xlabel="\$\\tau\$")
    # display(f_probe)

    return ps, loss_train_history, loss_test_history, τ_history, RMSE_final_train, RMSE_final_test
end

main(N_epochs=10, flag_plot=0)

## Experiment 1
function exp_1()
    seed_list = 1:40
    method_list = 1:2
    N_epochs_list = [200, 3000]
    simD = Array{Any}(undef, length(method_list), length(seed_list))
    for i_method in method_list
        for i_seed in seed_list
            @show (i_method, i_seed)
            T_elap = @elapsed res = main(seed=i_seed, N_epochs=N_epochs_list[i_method], method=i_method, flag_plot=0)
            # (ps, L_train, L_test, τ, RMSE_f_train, RMSE_f_test)=res
            simD[i_method, i_seed] = (; res=res, T_elap=T_elap)
        end
    end
    jldsave("simD_exp_1.jld2"; simD)

    # simD = load("simD_exp_1.jld2", "simD")
    default(fontfamily="Computer Modern")
    for i_method in method_list
        @show i_method
        L_train_mean = mean([simD[i_method, i_seed].res[2] for i_seed in seed_list])
        L_train_std = std([simD[i_method, i_seed].res[2] for i_seed in seed_list])
        L_test_mean = mean([simD[i_method, i_seed].res[3] for i_seed in seed_list])
        L_test_std = std([simD[i_method, i_seed].res[3] for i_seed in seed_list])
        τ_mean = mean([simD[i_method, i_seed].res[4] for i_seed in seed_list])
        τ_std = std([simD[i_method, i_seed].res[4] for i_seed in seed_list])
        @show RMSE_f_train_mean = mean([simD[i_method, i_seed].res[5] for i_seed in seed_list])
        @show RMSE_f_train_std = std([simD[i_method, i_seed].res[5] for i_seed in seed_list])
        @show RMSE_f_test_mean = mean([simD[i_method, i_seed].res[6] for i_seed in seed_list])
        @show RMSE_f_test_std = std([simD[i_method, i_seed].res[6] for i_seed in seed_list])
        @show T_elap_mean = mean([simD[i_method, i_seed].T_elap for i_seed in seed_list])
        @show T_elap_std = std([simD[i_method, i_seed].T_elap for i_seed in seed_list])

        f_L = plot(1:N_epochs_list[i_method], L_train_mean, ribbon=L_train_std, fillalpha=0.2, linewidth=2, xlabel="Iteration", ylabel="\$\\mathcal{L}\$", label="Training Objective", legend=:topleft)
        plot!(f_L, 1:N_epochs_list[i_method], L_test_mean, ribbon=L_test_std, fillalpha=0.2, linewidth=2, label="Test Objective", yticks=[10^i for i in -5.2:0.1:4.7], ylim=(6e-6, 10^-4.7), yaxis=:log)
        plot!(twinx(), τ_mean, ribbon=τ_std, ylabel="\$\\tau\$", fillalpha=0.2, linewidth=2, color=palette(:tab10)[3], label="Stepsize", legend=:topright)
        display(f_L)
        savefig(f_L, "Fig_E1_L_tau_M$(i_method).pdf")
    end
end

exp_1()

## Experiment 2
function heatmap_ann(f, a, width_list, depth_list)
    ann = [(width_list[i], depth_list[j], (round(a[i, j], digits=3), 14, :white, :center))
           for i in axes(a,1) for j in axes(a,2)]
    plot!(f; annotation=ann, linecolor=:white)
end

function exp_2()
    seed_list = 1:40
    width_list = [15, 20, 25, 30, 35]
    depth_list = [4, 6, 8, 10, 12]
    N_layer_list = [[12, repeat([width_list[i_width]], depth_list[i_depth] - 1)..., 3] for i_width in eachindex(width_list), i_depth in eachindex(depth_list)]

    simD = Array{Any}(undef, length(width_list), length(depth_list), length(seed_list))
    for i_width in eachindex(width_list)
        for i_depth in eachindex(depth_list)
            for i_seed in seed_list
                @show (i_width, i_depth, i_seed)
                T_elap = @elapsed res = main(seed=i_seed, N_layer=N_layer_list[i_width, i_depth], N_epochs=200, method=1, flag_plot=0)
                simD[i_width, i_depth, i_seed] = (; res=res, T_elap=T_elap)
            end
        end
    end
    jldsave("simD_exp_2.jld2"; simD)

    simD = load("simD_exp_2.jld2", "simD")

    RMSE_f_train_mean = [mean([simD[i_width, i_depth, i_seed].res[5] for i_seed in seed_list]) for i_width in eachindex(width_list), i_depth in eachindex(depth_list)]
    RMSE_f_train_std = [std([simD[i_width, i_depth, i_seed].res[5] for i_seed in seed_list]) for i_width in eachindex(width_list), i_depth in eachindex(depth_list)]
    RMSE_f_test_mean = [mean([simD[i_width, i_depth, i_seed].res[6] for i_seed in seed_list]) for i_width in eachindex(width_list), i_depth in eachindex(depth_list)]
    RMSE_f_test_std = [std([simD[i_width, i_depth, i_seed].res[6] for i_seed in seed_list]) for i_width in eachindex(width_list), i_depth in eachindex(depth_list)]
    T_elap_mean = [mean([simD[i_width, i_depth, i_seed].T_elap for i_seed in seed_list]) for i_width in eachindex(width_list), i_depth in eachindex(depth_list)]
    T_elap_std = [std([simD[i_width, i_depth, i_seed].T_elap for i_seed in seed_list]) for i_width in eachindex(width_list), i_depth in eachindex(depth_list)]

    default(fontfamily="Computer Modern")

    f_hmap = heatmap(width_list, depth_list, RMSE_f_train_mean', xlabel="Width", ylabel="Depth")
    heatmap_ann(f_hmap, RMSE_f_train_mean, width_list, depth_list)
    display(f_hmap)
    savefig(f_hmap, "Fig_E2_RMSE_f_train_mean.pdf")

    f_hmap = heatmap(width_list, depth_list, RMSE_f_train_std', xlabel="Width", ylabel="Depth")
    heatmap_ann(f_hmap, RMSE_f_train_std, width_list, depth_list)
    display(f_hmap)
    savefig(f_hmap, "Fig_E2_RMSE_f_train_std.pdf")

    f_hmap = heatmap(width_list, depth_list, RMSE_f_test_mean', xlabel="Width", ylabel="Depth")
    heatmap_ann(f_hmap, RMSE_f_test_mean, width_list, depth_list)
    display(f_hmap)
    savefig(f_hmap, "Fig_E2_RMSE_f_test_mean.pdf")

    f_hmap = heatmap(width_list, depth_list, RMSE_f_test_std', xlabel="Width", ylabel="Depth")
    heatmap_ann(f_hmap, RMSE_f_test_std, width_list, depth_list)
    display(f_hmap)
    savefig(f_hmap, "Fig_E2_RMSE_f_test_std.pdf")

    f_hmap = heatmap(width_list, depth_list, T_elap_mean', xlabel="Width", ylabel="Depth")
    heatmap_ann(f_hmap, T_elap_mean, width_list, depth_list)
    display(f_hmap)
    savefig(f_hmap, "Fig_E2_T_elap_mean.pdf")

    f_hmap = heatmap(width_list, depth_list, T_elap_std', xlabel="Width", ylabel="Depth")
    heatmap_ann(f_hmap, T_elap_std, width_list, depth_list)
    display(f_hmap)
    savefig(f_hmap, "Fig_E2_T_elap_std.pdf")

    # f_hmap = heatmap(width_list, depth_list, RMSE_f_test_mean' ./ T_elap_mean', xlabel="Width", ylabel="Depth")
    # heatmap_ann(f_hmap, RMSE_f_test_mean ./ T_elap_mean, width_list, depth_list)
    # display(f_hmap)
    # savefig(f_hmap, "Fig_E2_RMSE_f_test_over_T_elap.pdf")
end

exp_2()