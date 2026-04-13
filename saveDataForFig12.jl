using PyPlot
using LinearAlgebra
using Statistics

# ----------------------------
# Metrics (columns = samples)
# ----------------------------
function rmse_per_sample(ŷ::AbstractMatrix, y::AbstractMatrix)
    @assert size(ŷ) == size(y) "Shapes must match"
    N = size(y, 2)
    [sqrt(mean((ŷ[:,i] .- y[:,i]).^2)) for i in 1:N]
end

"Relative L2 error of fields per sample: ‖K̂−K‖ / ‖K‖"
function rel_l2_per_sample(x̂::AbstractMatrix, x::AbstractMatrix; eps=1e-12)
    @assert size(x̂) == size(x) "Shapes must match"
    N = size(x, 2)
    [norm(x̂[:,i] .- x[:,i]) / max(norm(x[:,i]), eps) for i in 1:N]
end

"Empirical CDF (sorted x, cumulative y)"
function ecdf(v::AbstractVector)
    s = sort(v)
    n = length(s)
    s, range(1/n, 1; length=n)
end

# ----------------------------------------
# Main plotting function (save optional)
# ----------------------------------------
function plot_model_comparison(bhp_true, bhp_pred_data, bhp_pred_phys,
                               K_true, K_pred_data, K_pred_phys,casename;
                               savepath::Union{Nothing,String}=nothing)

    # ---- compute metrics per sample
    rmse_data = rmse_per_sample(bhp_pred_data, bhp_true)
    rmse_phys = rmse_per_sample(bhp_pred_phys, bhp_true)

    permerr_data = rel_l2_per_sample(K_pred_data, K_true)
    permerr_phys = rel_l2_per_sample(K_pred_phys, K_true)

    # deltas (<0 means physics-embedded is better)
    Δrmse     = rmse_phys .- rmse_data
    Δpermerr  = permerr_phys .- permerr_data

    # quick summary in REPL
    println("BHP RMSE  median (data vs phys): ",
            round(median(rmse_data), sigdigits=4), " vs ",
            round(median(rmse_phys), sigdigits=4))
    println("Perm RelL2 median (data vs phys): ",
            round(median(permerr_data), sigdigits=4), " vs ",
            round(median(permerr_phys), sigdigits=4))
    println("% samples where phys wins (BHP RMSE): ",
            round(100*mean(Δrmse .< 0), digits=1), "%")
    println("% samples where phys wins (Perm): ",
            round(100*mean(Δpermerr .< 0), digits=1), "%")

    # ---- plotting
    # rc("font", size=12)
    fig, axs = subplots(2, 3, dpi=1200,figsize=(9, 5.5), constrained_layout=true)

    # (1,1) ECDF of BHP RMSE
    x1,y1 = ecdf(rmse_data)
    x2,y2 = ecdf(rmse_phys)
    axs[1,1].plot(x1, y1, label="Data driven")
    axs[1,1].plot(x2, y2, label="Physics-informed")
    axs[1,1].set_title("ECDF: BHP RMSE ")
    axs[1,1].set_xlabel("RMSE (MPa)")
    axs[1,1].set_ylabel("Cumulative fraction")
    axs[1,1].grid(true, alpha=0.3)
    axs[1,1].legend()

    # (1,2) ECDF of permeability relative L2 error
    x3,y3 = ecdf(permerr_data)
    x4,y4 = ecdf(permerr_phys)
    axs[1,2].plot(x3, y3, label="Data driven")
    axs[1,2].plot(x4, y4, label="Physics-informed")
    axs[1,2].set_title("ECDF: Perm error")
    axs[1,2].set_xlabel("Rel-L2 error")
    axs[1,2].set_ylabel("Cumulative fraction")
    axs[1,2].grid(true, alpha=0.3)
    axs[1,2].legend()

    # (1,3) Boxplots for quick distribution comparison
    # axs[1,3].boxplot([rmse_data, rmse_phys], labels=["Data-based","Physics-informed"], showfliers=false)
    # axs[1,3].set_title("BHP RMSE (boxplot)")
    # axs[1,3].set_ylabel("RMSE")
    # axs[1,3].grid(true, axis="y", alpha=0.3)

    # (1,3) Boxplots with twin y-axes: BHP RMSE (left) + Perm error (right)
    axL = axs[1,3]
    bpL = axL.boxplot([rmse_data, rmse_phys];
                    positions=[1.0, 1.5], widths=0.25,
                    showfliers=false, patch_artist=true, labels=["DD","PI"] )
    # axL.set_title("MSE (left) & Perm error (right)")
    axL.set_ylabel("BHP MSE")
    axL.grid(true, axis="y", alpha=0.3)

    # Secondary y-axis for permeability error (your rel L2 metric)
    axR = axL.twinx()
    bpR = axR.boxplot([permerr_data, permerr_phys];
                    positions=[2.5, 3], widths=0.25,
                    showfliers=false, patch_artist=true, labels=["DD","PI"])
    axR.set_ylabel("PERM MSE")

    # axL.legend()
    bpR["boxes"][1].set_alpha(0.35 )
    bpL["boxes"][1].set_alpha(0.35 )
    # Lighten / differentiate the right-axis boxes
    # for b in bpR["boxes"]; b.set_alpha(0.35 ); end
    # for m in bpR["medians"]; m.set_alpha(0.7); end
    # for w in bpR["whiskers"]; w.set_alpha(0.7); end
    # for c in bpR["caps"]; c.set_alpha(0.7); end

    # Center x-ticks between paired boxes
    # axL.set_xticks([1.25, 2.75])
    # axL.set_xticklabels(["Data driven", "Physics-informed"])

    # Optional: make the axes visually distinct
    axL.spines["left"].set_linewidth(1.2)
    axR.spines["right"].set_linewidth(1.2)


    # (2,1) Histograms of BHP RMSE
    axs[2,1].hist(rmse_data, bins=50, alpha=0.6, label="Data driven", density=true)
    axs[2,1].hist(rmse_phys, bins=50, alpha=0.6, label="Physics-informed", density=true)
    axs[2,1].set_title("Histogram: BHP RMSE")
    axs[2,1].set_xlabel("RMSE")
    axs[2,1].set_ylabel("Density")
    axs[2,1].grid(true, alpha=0.3)
    axs[2,1].legend()
    # axs[2,1].set_xlim(0.0, 0.05) 

    # (2,2) Histograms of permeability relative L2
    axs[2,2].hist(permerr_data, bins=50, alpha=0.6, label="Data driven", density=true)
    axs[2,2].hist(permerr_phys, bins=50, alpha=0.6, label="Physics-informed", density=true)
    axs[2,2].set_title("Histogram: Perm Rel-L2 error")
    axs[2,2].set_xlabel("Rel-L2 error")
    axs[2,2].set_ylabel("Density")
    axs[2,2].grid(true, alpha=0.3)
    axs[2,2].legend()

    # (2,3) Delta scatter: improvement of Sim-embedded vs Data-only
    axs[2,3].scatter(Δpermerr, Δrmse, s=8, alpha=0.5)
    axs[2,3].axvline(0, linestyle="--", linewidth=1,color="red")
    axs[2,3].axhline(0, linestyle="--", linewidth=1,color="red")
    axs[2,3].set_title("Physics-informed − Data-based")
    axs[2,3].set_xlabel("Δ Perm Rel-L2  ")
    axs[2,3].set_ylabel("Δ BHP RMSE ")
    axs[2,3].grid(true, alpha=0.3)
    # axs[2,3].set_ylim(-0.2, 0.1) 

    fig.suptitle("$casename", fontsize=14)
    if savepath !== nothing
        savefig(savepath, bbox_inches="tight")
        println("Saved figure to: $savepath")
    end
    display(fig)
    return nothing
end
# ----------------------------
# Example (shape notes only)
# ----------------------------
# # Suppose you already have these matrices with columns = samples (N = 10_000):


#
#  Trains a neural network to predict KL‐coefficients for a 2D permeability
#  field from sparse pressure (BHP) measurements.  The loss is defined in
#  physics‐space: we reconstruct the logK via KLE, run a fully‐differentiable
#  DPFEHM steady‐state solver, compare predicted BHPs against the “true” BHPs,
#  and backpropagate through everything.
#
#  Key changes here:
#   •   Precompute ϕ (eigenfunctions) and σ (eigenvalues) once.
#   •   Build a dataset of (bhp_true, x_true) pairs up‐front, so each epoch
#       iterates without re‐generating geometry or KL every time.
#   •   Use a simple MLP: input dim = #monitor nodes, output dim = num_eig.
#   •   Define a loss that, for a batch, reconstructs logK_pred = ϕ·(σ .* x_pred)
#       and then runs the solver to get bhp_pred.  Loss = MSE(bhp_pred, bhp_true).
#   •   Vectorise loops over batch samples where possible; avoid repeated
#       calls to random‐seed or reconstructing the covariance in the inner loop.
#   •   Use ADAM with a small lr and early stopping criteria.
#
#  Requirements:
#    •   DPFEHM.jl 
#    •   GaussianRandomFields.jl (for KL eigen funcs/vals)
#    •   Flux.jl, Zygote.jl, BSON.jl, JLD2.jl, Random


# Import packages
using Random
using Statistics: mean, std
using Flux
using Zygote
using DPFEHM
using GaussianRandomFields
using BSON
using PyPlot
import StatsBase
using JLD2
#Problem Setup

mutable struct Fluid
    vw::Float64
    vo::Float64
    swc::Float64
    sor::Float64
end

n           = 51                   #  grid is n × n
ns          = (n, n)
sidelength  = 100.0                # [m] half‐width in each direction
thickness   = 1.0                  # [m]
num_eig     = 200                #  number of KL modes we keep
num_cells   = n * n

# Build a uniform 2D grid from (–sidelength, –sidelength) to (sidelength, 2*sidelength)


# casedata


# casename="TBeB"
# num_mon  = 200
# λ_cov   = 100.0
# @BSON.load "ModelPhysicsBBBv2.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataBBBv2.bson" model
# modelDATA=model



# casename="TBeRE"
# num_mon  = 200
# λ_cov   = 100.0
# @BSON.load "ModelPhysicsBBBv2n10.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataBBBv2n10.bson" model
# modelDATA=model

# casename="TBnREeRE"
# num_mon  = 200
# λ_cov   = 100.0
# @BSON.load "ModelPhysicsBBBv2Ren10.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataBBBv2REn10.bson" model
# modelDATA=model


casename="TREeRE"
num_mon  = 200
λ_cov   = 100.0
@BSON.load "ModelPhysicsBBBv2ReOnlyn10.bson" model
modelDPFEHM=model
@BSON.load "ModelDataBBBv2REonlyn10.bson" model
modelDATA=model


# casename="TBnREeB"
# num_mon  = 200
# λ_cov   = 100.0
# @BSON.load "ModelPhysicsBBBv2Ren10.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataBBBv2REn10.bson" model
# modelDATA=model


# casename="BBSv2"
# num_mon  = 50
# λ_cov   = 10.0
# @BSON.load "ModelPhysicsBBSv2.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataBBSv2.bson" model
# modelDATA=model

# casename="BSB"
# num_mon   = 10
# λ_cov   = 100.0
# @BSON.load "ModelPhysicsBSB.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataBSB.bson" model
# modelDATA=model

# casename="BSSv2"
# num_mon = 50
# λ_cov   = 10.0
# @BSON.load "ModelPhysicsBSSv2.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataBSSv2.bson" model
# modelDATA=model

# casename="SBB"
# num_mon = 50
# λ_cov   = 100.0
# @BSON.load "ModelPhysicsSBB.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataSBB.bson" model
# modelDATA=model

# casename="SBS"
# num_mon = 50
# λ_cov   = 10.0
# @BSON.load "ModelPhysicsSBS.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataSBS.bson" model
# modelDATA=model

# casename="SSB"
# num_mon = 10
# λ_cov   = 100.0
# @BSON.load "ModelPhysicsSSB.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataSSB.bson" model
# modelDATA=model


# casename="SSSv2"
# num_mon = 10
# λ_cov   = 10.0
# @BSON.load "ModelPhysicsSSSv2.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataSSSv2.bson" model
# modelDATA=model


println("  ▸ Generating 2D regular grid …")
coords, neighbors, areasoverlengths, volumes =DPFEHM.regulargrid2d([-sidelength, -sidelength],[sidelength, 2sidelength],ns,thickness)

h0 = zeros(size(coords, 2))
fluid=Fluid(1.0, 1.0, 0.0, 0.0)
S0=zeros(size(coords, 2))
nt = 1;  dt = 3*24*60*60;

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

injrate         = 0.0        # [m^3/s]

# Monitoring nodes: pick random nodes (to mimic sparse pressure data).
# We fix the seed so that training/test splits are reproducible.

Random.seed!(1256879)
monitoring_nodes = sort!(randperm(num_cells)[1:num_mon])

println("    → Selected $num_mon monitoring nodes: ",
        monitoring_nodes)

critical_nodes=2431


#KL‐(Karhunen–Loève) setup 
# Build a Matern covariance (ν = 1, σ = 1.0, λ = 100.0) on our n×n grid
println("  ▸ Building GaussianRandomFields KL …")
σ_cov   = 1.0

cov_func = GaussianRandomFields.CovarianceFunction(
                   2,
                   GaussianRandomFields.Matern(λ_cov, 1; σ = σ_cov)
                 )

# Extract the x‐ and y‐coordinates of the grid for sampling the KL.
#  coords[1,:] are the x‐values of all cell centers, but we want them arranged
#  as a range in [xmin, xmax] of length n.  Because regulargrid2d produces
#  a lexicographic ordering, we can simply take unique values.
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

# load a fully–differentiable “forward solver”

function getQs(Qs, is)
    sum(Qs .* ((collect(1:size(coords, 2)) .== i) for i in is))
end
logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))#Zygote differentiates this efficiently but the definitions above are ineffecient with Zygote
function solve_bhp(logK_vec,monitoring_nodes)

    Q_vec = getQs([injrate], [injection_node])
    
    # uncomment for  steady state
    Ks_neighbors = logKs2Ks_neighbors(reshape(logK_vec,n,n))
    # # 3) Call DPFEHM steady‐state groundwater solver:
    # #    P_full is a length‐num_cells vector of pressures at each cell.
    P_full = DPFEHM.groundwater_steadystate(
               Ks_neighbors,
               neighbors,
               areasoverlengths,
               dirichletnodes,
               dirichleths,
               Q_vec
             )

    #for transient twophase
    # permVal=exp.(logK_vec)
    # everystep=false # output all the time 
    # args=h0, S0, permVal, dirichleths,  dirichletnodes, Q_vec, volumes, areasoverlengths, fluid, dt, neighbors, nt, everystep
    # P_full, S= solveTwoPhase(args...)
   
    # 4) Extract pressures at monitoring_nodes:
    return @view P_full[monitoring_nodes]
end

function solve_bhp_full(logK_vec,critical_point)

    Q_vec = getQs([injrate], [injection_node])
    
    # uncomment for  steady state
    Ks_neighbors = logKs2Ks_neighbors(reshape(logK_vec,n,n))
    # # 3) Call DPFEHM steady‐state groundwater solver:
    # #    P_full is a length‐num_cells vector of pressures at each cell.
    P_full = DPFEHM.groundwater_steadystate(
               Ks_neighbors,
               neighbors,
               areasoverlengths,
               dirichletnodes,
               dirichleths,
               Q_vec
             )

    #for transient twophase
    # permVal=exp.(logK_vec)
    # everystep=false # output all the time 
    # args=h0, S0, permVal, dirichleths,  dirichletnodes, Q_vec, volumes, areasoverlengths, fluid, dt, neighbors, nt, everystep
    # P_full, S= solveTwoPhase(args...)
   
    # 4) Extract pressures at monitoring_nodes:
    return  P_full[critical_point]
end



@BSON.load "C:/Users/398654/Documents/AGU_MLnC_paperCode/Tail_Aware_POD-main/sampled_tails_trainTestval.bson" x_tail h_mon_tail;


true_bhps=h_mon_tail[:,5201:end]
true_xs=x_tail[:,5201:end]




# threshold =1e-2
# true_bhps_i=true_bhps
# true_xs_i=true_xs
# keep=isweight_tail[5201:end].>threshold
# true_bhps=true_bhps_i[:,keep]
# true_xs=true_xs_i[:,keep]
# true_bhps=h_mon_tail[:,isweight_tail .> threshold]
# true_xs=x_tail[:, isweight_tail .> threshold]


# N_val=1000

# true_bhps  = Matrix{Float32}(undef, num_mon, N_val)
# true_xs = Matrix{Float32}(undef, num_eig, N_val)

# xrand=123 # fixed for plotting(1234)
# for i in 1:N_val
#     Random.seed!(xrand+i)
#     x_true = randn(Float32, num_eig)
#     logK_vec = ϕ_matrix * (σ_vec .* Float64.(x_true))
#     bhp_vec   = solve_bhp(logK_vec,monitoring_nodes)
#     @inbounds begin
#       true_bhps[:,  i] = Float32.(bhp_vec)
#       true_xs[:, i] = x_true
#     end
# end



N_val=size(true_xs,2)
true_critical_pres=Matrix{Float32}(undef, 1, N_val)
pred_critical_pres_data=Matrix{Float32}(undef, 1, N_val)
pred_critical_pres_physics=Matrix{Float32}(undef, 1, N_val)

test_bhps  = Matrix{Float32}(undef, num_mon, N_val)
pred_bhps_data  = Matrix{Float32}(undef, num_mon, N_val)
pred_xs_data = Matrix{Float32}(undef, num_eig, N_val)


pred_bhps_sim  = Matrix{Float32}(undef, num_mon, N_val)
pred_xs_sim= Matrix{Float32}(undef, num_eig, N_val)


perms_true = Matrix{Float32}(undef, n*n, N_val)
perms_data = Matrix{Float32}(undef, n*n, N_val)
perms_sim= Matrix{Float32}(undef, n*n, N_val)

for i in 1:N_val

    x_pred_data = modelDATA(true_bhps[:,  i])        # size = num_eig × B
    x_pred_sim = modelDPFEHM(true_bhps[:,  i])        # size = num_eig × B

    logK_vec_true = ϕ_matrix * (σ_vec .* Float64.(true_xs[:, i] ))
    logK_vec_data = ϕ_matrix * (σ_vec .* Float64.(x_pred_data))
    logK_vec_sim = ϕ_matrix * (σ_vec .* Float64.(x_pred_sim))


    true_critical_pres[i]=  solve_bhp_full(logK_vec_true,critical_nodes)
    pred_critical_pres_data[i]=  solve_bhp_full(logK_vec_data,critical_nodes)
    pred_critical_pres_physics[i]=  solve_bhp_full(logK_vec_sim,critical_nodes)

    bhp_vec_data   = solve_bhp(logK_vec_data,monitoring_nodes)
    bhp_vec_sim   = solve_bhp(logK_vec_sim,monitoring_nodes)

    pred_bhps_data[:,  i] = bhp_vec_data
    pred_bhps_sim[:,  i] = bhp_vec_sim

    pred_xs_data[:,  i] = x_pred_data
    pred_xs_sim[:,  i] = x_pred_sim


    perms_true[:,  i] = logK_vec_true
    perms_data[:,  i] = logK_vec_data
    perms_sim[:,  i] = logK_vec_sim
end


rmse_data_bbb = rmse_per_sample(pred_critical_pres_data, true_critical_pres)
rmse_phys_bbb  = rmse_per_sample(pred_critical_pres_physics, true_critical_pres)



fig, axs = subplots(constrained_layout=true, figsize=(12,5))

axL = axs
bp=axL.boxplot([rmse_data_bbb, rmse_phys_bbb];
                positions=[1,2], widths=0.2,
                showfliers=false, patch_artist=true, labels=["$(casename)\nDD", "$(casename)\nPI"] )


# axL.set_title("MSE (left) & Perm error (right)")
axL.set_ylabel("BHP RMSE (MPa)", fontsize=16)

# Define custom colors
colors = ["#ff7f0e", "#1f77b4"]

# Apply colors to each box
for (patch, color) in zip(bp["boxes"], colors)
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
    patch.set_edgecolor("black")
end

axL.grid(true, axis="y", alpha=0.3)
axL.tick_params(labelsize=16)

display(fig)
fig.savefig("Pressureatcriticaloc.pdf")
println()
PyPlot.close(fig)



# true_bhps=test_bhps

bhp_true       = true_bhps
bhp_pred_data  = pred_bhps_data
bhp_pred_phys  = pred_bhps_sim
K_true         = perms_true
K_pred_data    = perms_data
K_pred_phys    = perms_sim


BSON.@save "Evaluation_$casename.bson" bhp_true bhp_pred_data bhp_pred_phys K_true K_pred_data K_pred_phys

plot_model_comparison(bhp_true, bhp_pred_data, bhp_pred_phys,
                      K_true, K_pred_data, K_pred_phys,casename;
                      savepath="$casename.pdf")


# rmseP_critical_data= rmse_per_sample(pred_critical_pres_data, true_critical_pres)
# rmseP_critical_physics  = rmse_per_sample(pred_critical_pres_physics, true_critical_pres)




# fig, axs = subplots(constrained_layout=true, figsize=(12,5))

# axL = axs
# bp=axL.boxplot([rmseP_critical_data, rmseP_critical_physics];
#                 positions=[1.0,1.5, ], widths=0.2,
#                 showfliers=false, patch_artist=true, labels=["DD", "PI", ] )


# # axL.set_title("MSE (left) & Perm error (right)")
# axL.set_ylabel("BHP RMSE (MPa)", fontsize=16)

# # Define custom colors
# colors = ["#ff7f0e", "#1f77b4"]

# # Apply colors to each box
# for (patch, color) in zip(bp["boxes"], colors)
#     patch.set_facecolor(color)
#     patch.set_alpha(0.6)
#     patch.set_edgecolor("black")
# end

# axL.grid(true, axis="y", alpha=0.3)
# axL.tick_params(labelsize=16)

# display(fig)
# fig.savefig("PressureOverall.pdf")
# println()
# PyPlot.close(fig)


# # fig, ax = PyPlot.subplots()
# # ax.imshow(reshape(K_true[:,2],51,51), origin="lower")
# # display(fig); println()
# # PyPlot.close(fig)

# # fig, ax = PyPlot.subplots()
# # ax.imshow(reshape(K_pred_data[:,2],51,51), origin="lower")
# # display(fig); println()
# # PyPlot.close(fig)

# # fig, ax = PyPlot.subplots()
# # ax.imshow(reshape(K_pred_phys[:,2],51,51), origin="lower")
# # display(fig); println()
# # PyPlot.close(fig)

# # using Statistics
# # std(true_xs[:,1])


# # fig, ax = PyPlot.subplots()
# # ax.hist(vec(true_bhps_full); bins=100, edgecolor="black", facecolor="darkblue")
# # # PyPlot.xlim(-0.005, 0.005)
# # ax.set_xlabel("Pressure at the critical location, MPa",
# #                fontsize=15, fontname="Arial")
# # ax.set_ylabel("Frequency", fontsize=15, fontname="Arial")
# # # Tick labels
# # for lbl in ax.get_xticklabels();  lbl.set_fontname("Arial"); lbl.set_fontsize(15); end
# # for lbl in ax.get_yticklabels();  lbl.set_fontname("Arial");  lbl.set_fontsize(15); end
# # display(fig)
# # close(fig)
# # @show true_bhps_full[1]
# # y=true_bhps_full[:,:]