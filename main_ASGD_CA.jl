using LinearAlgebra, Random, Statistics
using Lux, Zygote, ForwardDiff
# using Enzyme
using ComponentArrays
using CSV, DataFrames
using Plots
cd(@__DIR__)

function construct_FCNN(N_layer::Vector{Int64})
    return Lux.Chain(
        [Dense(fan_in => fan_out, tanh, use_bias=false) for (fan_in, fan_out) in zip(N_layer[1:end-2], N_layer[2:end-1])]...,
        Dense(N_layer[end-1] => N_layer[end], identity, use_bias=false),
    )
end

loss(model, ps, ls, x::Matrix{T}, y::Matrix{T}) where {T<:AbstractFloat} = sum(abs2, first(model(x, ps, ls)) .- y) / size(x, 2)

exp_S(t, ps, grad) = cos(t) * ps - sin(t) * grad

function ASGD_AD!(model, ps, norm_ps, grad, lks, ls, L_prev::T, x::Matrix{T}, y::Matrix{T}; ϵ=T(pi / 6.0)) where {T<:AbstractFloat}
    # this function assumes that ps[i].weight is Frobenius-normalised
    grad .= Zygote.gradient(p -> loss(model, p, ls, x, y), ps)[1]
    # Enzyme.gradient!(Reverse, grad, p -> loss(model, p, ls, x, y), ps)

    neg_dL1 = sum([sqrt(max(abs2(norm_ps[k]) * dot(grad[k], grad[k]) - abs2(dot(ps[k], grad[k])), zero(T))) for k in lks])
    for k in lks
        axpy!(-dot(ps[k], grad[k]) / abs2(norm_ps[k]), ps[k], @view grad[k])
        normalize!(@view grad[k])
        lmul!(norm_ps[k], @view grad[k])
    end

    dL2 = Zygote.hessian(t -> loss(model, exp_S(t, ps, grad), ls, x, y), zero(T))
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

    L_τ = loss(model, exp_S(τ, ps, grad), ls, x, y)
    while L_τ > L_prev
        τ /= T(2)
        L_τ = loss(model, exp_S(τ, ps, grad), ls, x, y)
    end

    axpby!(-sin(τ), grad, cos(τ), ps)
    return L_τ  
end

function main(; seed=1, N_layer=[12, 25, 30, 15, 3], N_epochs=100)
    rng = Xoshiro(seed)

    data = CSV.read("neural_lander_data.csv", DataFrame) |> Matrix
    t = Float32.(vec(data[:, 1]))
    # id_perm  = randperm(rng, size(data, 1))
    # id_train = id_perm[1:round(Int, 0.5 * length(id_perm))]
    # id_test  = id_perm[round(Int, 0.5 * length(id_perm))+1:end]
    id_train = 1:round(Int, 0.8 * size(data, 1))
    id_test = id_train[end]+1:size(data, 1)
    x_train = Float32.(data[id_train, 2:13]')
    y_train = Float32.(data[id_train, 14:16]')
    x_test = Float32.(data[id_test, 2:13]')
    y_test = Float32.(data[id_test, 14:16]')

    model = Lux.Chain(
        [Dense(fan_in => fan_out, Lux.relu, use_bias=false) for (fan_in, fan_out) in zip(N_layer[1:end-2], N_layer[2:end-1])]...,
        Dense(N_layer[end-1] => N_layer[end], identity, use_bias=false),
    )

    ps, ls = Lux.setup(rng, model)
    ps = ComponentArray(ps)
    layer_keys = keys(ps)
    norm_ps_val = [Float32(sqrt(prod(size(ps[k].weight))) / 3.0) for k in layer_keys]
    norm_ps = NamedTuple{layer_keys}(norm_ps_val)
    grad = similar(ps)
    for k in layer_keys
        normalize!(@view ps[k])
        lmul!(norm_ps[k], @view ps[k])
    end

    L_prev = loss(model, ps, ls, x_train, y_train)
    loss_history = []
    for epoch in 1:N_epochs
        @time L_current = ASGD_AD!(model, ps, norm_ps, grad, layer_keys, ls, L_prev, x_train, y_train; ϵ=Float32(pi / 6.0))

        push!(loss_history, L_current)
        if epoch % 100 == 0
            println("Epoch: $epoch, Loss: $(loss_history[end])")
        end
        L_prev = L_current
    end

    f_loss = plot(1:N_epochs, loss_history, xlabel="epoch", ylabel="\$\\mathcal{L}\$", label=false)
    display(f_loss)

    f_y_train = plot(t[id_train], y_train', xlabel="\$t\$ [s]", ylabel=["\$F_{a_{x}}\$" "\$F_{a_{y}}\$" "\$F_{a_{z}}\$"], label=[false false "\$y\$ train"], legend=:topleft, layout=(3, 1), size=(600, 600))
    plot!(f_y_train, t[id_train], model(x_train, ps, ls)[1]', label=[false false "\$\\hat{y}\$ train"])
    display(f_y_train)

    f_y_test = plot(t[id_test], y_test', xlabel="\$t\$ [s]", ylabel=["\$F_{a_{x}}\$" "\$F_{a_{y}}\$" "\$F_{a_{z}}\$"], label=[false false "\$y\$ test"], layout=(3, 1), size=(600, 600))
    plot!(f_y_test, t[id_test], model(x_test, ps, ls)[1]', label=[false false "\$\\hat{y}\$ test"])
    display(f_y_test)

    return ps, loss_history
end

main()