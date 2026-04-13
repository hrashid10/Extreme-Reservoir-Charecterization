

# Import packages
using Random
using Statistics: mean, std
using Flux
using Zygote
using DPFEHM
using GaussianRandomFields
using BSON
using Distributed

# num_processor=5
# addprocs(num_processor)
@everywhere begin
    using GaussianRandomFields
    using DPFEHM
    using Random
    using Zygote
    using Flux

    #sensitivity parameters

    N_train = 5000
    num_mon = 50
    λ_cov   = 10.0

    n           = 51                   #  grid is n × n
    ns          = (n, n)
    sidelength  = 100.0                # [m] half‐width in each direction
    thickness   = 1.0                  # [m]
    num_eig     = 200                  #  number of KL modes we keep
    num_cells   = n * n


    println("  ▸ Generating 2D regular grid …")
    coords, neighbors, areasoverlengths, volumes =DPFEHM.regulargrid2d([-sidelength, -sidelength],[sidelength, 2sidelength],ns,thickness)


    # Boundary (Dirichlet) specification: right‐hand side of the domain set to constant “steadyhead” = 0. 
    steadyhead       = 0.0
    dirichletnodes = Int[]
    dirichleths = zeros(size(coords, 2))
    injection_node=26
    for i = 1:size(coords, 2)
        if  (coords[1, i]) == sidelength
            push!(dirichletnodes, i)
            dirichleths[i] = steadyhead
        end
    end
    for i = 1:size(coords, 2)
        if  (coords[1, i]) == -sidelength
            push!(dirichletnodes, i)
            dirichleths[i] = 10
        end
    end

    for i=1:size(coords,2)
        if coords[2,i] ==  maximum(coords[2,:])   # top
            push!(dirichletnodes, i); dirichleths[i] = 0.5   # small head
        elseif coords[2,i] == minimum(coords[2,:]) # bottom
            push!(dirichletnodes, i); dirichleths[i] = 0.0
        end
    end

    injrate = 0.0         # [m^3/s]

    
    Random.seed!(1256879)
    monitoring_nodes = sort!(randperm(num_cells)[1:num_mon])

    println("    → Selected $num_mon monitoring nodes: ",
            monitoring_nodes)


    println("  ▸ Building GaussianRandomFields KL …")
    σ_cov   = 1.0
    
    cov_func = GaussianRandomFields.CovarianceFunction(
                    2,
                    GaussianRandomFields.Matern(λ_cov, 1; σ = σ_cov)
                    )
    x_min, x_max = minimum(coords[1,:]), maximum(coords[1,:])
    y_min, y_max = minimum(coords[2,:]), maximum(coords[2,:])

    x_pts = range(x_min, x_max; length = n)
    y_pts = range(y_min, y_max; length = n)

    # Build the GaussianRandomField object with Karhunen–Loève
    grf = GaussianRandomFields.GaussianRandomField(
            cov_func,
            GaussianRandomFields.KarhunenLoeve(num_eig),
            x_pts,
            y_pts
        )

    println("    → Extracting eigenfunctions/values …")
    ϕ_matrix = grf.data.eigenfunc    # size = (n*n) × num_eig
    σ_vec    = grf.data.eigenval     # length = num_eig

    function getQs(Qs, is)
        sum(Qs .* ((collect(1:size(coords, 2)) .== i) for i in is))
    end

    logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))

    function solve_bhp(x_pred_i)
        logK_vec = ϕ_matrix * (σ_vec .* Float64.(x_pred_i))
        Q_vec = getQs([injrate], [injection_node])
        Ks_neighbors = logKs2Ks_neighbors(reshape(logK_vec,n,n))
        P_full = DPFEHM.groundwater_steadystate(Ks_neighbors,neighbors,areasoverlengths,dirichletnodes,dirichleths,Q_vec)
        return P_full[monitoring_nodes]
    end

    
    function add_white_noise(y; sigma=nothing, rel_std=nothing, snr_db=nothing)
        yy = copy(y)
        if (sigma === nothing) + (rel_std === nothing) + (snr_db === nothing) != 2
            error("Provide exactly one of sigma, rel_std, or snr_db")
        end
        μ = mean(abs, yy)
        σ = sigma !== nothing ? sigma :
            rel_std !== nothing ? rel_std * max(μ, eps()) :
            begin
                rms = sqrt(mean((yy .- mean(yy)).^2))
                rms / (10.0^(snr_db/20))
            end
        return yy .+ σ .* randn(size(yy))
    end

end




# Test that the solver “runs” once on a random logK to ensure no errors:
println("  ▸ Testing solver once …")
x_test = randn(Float32, num_eig)               
# test_logK = ϕ_matrix * (σ_vec .* Float64.(x_test))   # (n*n) × 1
bhp_test = solve_bhp(x_test)     # should return a length‐num_mon vector
println("    → Solver OK: returned BHPs of size ", size(bhp_test))


println("  ▸ Generating $N_train training samples …")
 #Number of training samples
N_val   =  200 #Number of validation samples
train_bhps  = Matrix{Float32}(undef, num_mon, N_train)
train_coeff = Matrix{Float32}(undef, num_eig, N_train)

for i in 1:N_train
    x_true = randn(Float32, num_eig)               
    # logK_vec = ϕ_matrix * (σ_vec .* Float64.(x_true))   # (n*n) × 1
    bhp_vec   = solve_bhp(x_true)                     # length = num_mon
    bhp_vec = add_white_noise(bhp_vec; rel_std=0.1)
    @inbounds begin
        train_bhps[:,  i] = Float32.(bhp_vec)
        train_coeff[:, i] = x_true
    end
end

println("    → Done. Training data ready.")

println("  ▸ Generating $N_val validation samples …")
val_bhps  = Matrix{Float32}(undef, num_mon, N_val)
val_coeff = Matrix{Float32}(undef, num_eig, N_val)

for i in 1:N_val
    x_true = randn(Float32, num_eig)
    # logK_vec = ϕ_matrix * (σ_vec .* Float64.(x_true))
    bhp_vec   = solve_bhp(x_true)
    bhp_vec = add_white_noise(bhp_vec; rel_std=0.1)
    @inbounds begin
      val_bhps[:,  i] = Float32.(bhp_vec)
      val_coeff[:, i] = x_true
    end
end

println("    → Done. validation data ready.")




 α_coeff = 1e-1       # weight on MSE between x_pred and x_true      # weight on MSE between x_pred and x_true


model = Chain(
    Dense(num_mon, 64, relu),
    Dense(64, 64, relu),
    Dense(64, num_eig)
) |> f64
# @BSON.load "testrun_modelEEPdata.bson" model

θ = Flux.params(model)

function loss_calc(bhps_batch, coeffs_batch)

    B = size(bhps_batch, 2)       # batch size
    x_pred_mat = model(bhps_batch)        # size = num_eig × B
    total_bhp_loss=sum(pmap(i-> sum((solve_bhp(x_pred_mat[:, i]).-bhps_batch[:, i]).^2)/num_mon,1:B ))/B
    total_perm_loss=sum(map(i-> sum((x_pred_mat[:, i].-coeffs_batch[:,i]).^2)/num_eig,1:B ))/B

    return total_bhp_loss, α_coeff*total_perm_loss
end


# Loss for one batch: inputs BH X (size num_mon × batch), truths CX (size num_eig × batch).
function loss_batch(bhps_batch,
                    coeffs_batch)
    B = size(bhps_batch, 2)       # batch size
   
    # Forward‐pass through model to get x_pred for all B samples at once:
    x_pred_mat = model(bhps_batch)        # size = num_eig × B

    total_bhp_loss=sum(pmap(i-> sum((solve_bhp(x_pred_mat[:, i]).-bhps_batch[:, i]).^2)/num_mon,1:B ))/B
    total_perm_loss=sum(map(i-> sum((x_pred_mat[:, i].-coeffs_batch[:,i]).^2)/num_eig,1:B ))/B

    total_loss=total_bhp_loss+α_coeff*total_perm_loss

    return total_loss  # mean over batch
end

# Wrap in a Zygote‐compatible Flux loss:
loss_fn = (bhps_batch, coeffs_batch) -> loss_batch(bhps_batch, coeffs_batch)


#Training Loop
# Hyperparameters
lr        = 1e-4 # learning rate
batchsize = 5
n_epochs  = 10000  # adjust or use early‐stopping after “patience” of no improvement

opt = ADAM(lr)



# A simple way: at each epoch, shuffle columns and iterate over minibatches of size batchsize.

println("  ▸ Starting training …")

# Store metrics
train_times = Float32[]
train_losses = Float32[]
val_losses   = Float32[]
bhp_losses = Float32[]
perm_losses= Float32[]

# frac_per_epoch = N_val/N_train
frac_per_epoch =0.1

M = round(Int, N_train * frac_per_epoch)

for epoch in 1:n_epochs


    subset_idx = randperm(N_train)[1:M]
    subset_idx[1:5]
    shuffle_idx = randperm(M)
    # 3) compute how many batches in this subset
    n_batches = div(M, batchsize)

    epoch_train_loss = 0
    # n_batches = div(N_train, batchsize)

    train_time=@elapsed begin
        for b in 1:n_batches
            # idxs = shuffle_idx[( (b-1)*batchsize + 1 ) : (b*batchsize)]

            batch_positions = ((b-1)*batchsize + 1) : (b*batchsize)
            idxs = subset_idx[ shuffle_idx[batch_positions] ]


            bhps_batch   = train_bhps[:,  idxs]   # size = num_mon × batchsize
            coeffs_batch = train_coeff[:, idxs]   # size = num_eig × batchsize

            # One SGD step:
            l, back = Flux.withgradient(θ) do
                loss_fn(bhps_batch, coeffs_batch)
            end
            Flux.Optimise.update!(opt, θ, back)
            epoch_train_loss += l
        end
    end

    epoch_train_loss /= Float32(n_batches)
    push!(train_losses, epoch_train_loss)
    push!(train_times, train_time)

    # Compute validation loss
    n_val_batches = div(N_val, batchsize)
    total_val_loss = 0f0
    epoch_bhp_loss = 0f0
    epoch_perm_loss = 0f0
    for b in 1:n_val_batches
        starti = (b-1)*batchsize + 1
        endi   = b*batchsize
        bhps_batch_val   = val_bhps[:,  starti:endi]
        coeffs_batch_val = val_coeff[:, starti:endi]
        bhp_l2, perm_l2 = loss_calc(bhps_batch_val, coeffs_batch_val)
        epoch_bhp_loss  += bhp_l2
        epoch_perm_loss += perm_l2
        total_val_loss  +=bhp_l2+perm_l2
    end
    val_loss_epoch = total_val_loss / Float32(n_val_batches)

    
    
    epoch_bhp_loss /= Float32(n_val_batches)
    epoch_perm_loss /= Float32(n_val_batches)


    push!(val_losses, val_loss_epoch)
    push!(bhp_losses, epoch_bhp_loss)
    push!(perm_losses, epoch_perm_loss)


    println(string("epoch: ", epoch, " time: ", train_time, " train_loss: ", round(epoch_train_loss, digits=6), " test rmse: ", round(val_loss_epoch, digits=6), " bhp loss: ", round(epoch_bhp_loss, digits=6)," perm loss: ", round(epoch_perm_loss, digits=6)))
end

# Final save (in case early stopping never triggered)
BSON.@save "PhysicsSSSv2n10.bson" train_losses val_losses bhp_losses perm_losses
BSON.@save "ModelPhysicsSSSv2n10.bson" model 

