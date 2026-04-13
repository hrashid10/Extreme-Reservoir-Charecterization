using PyPlot
using LinearAlgebra
using Statistics
using Random
using Flux
using Zygote
using DPFEHM
using GaussianRandomFields
using BSON

# ----------------------------
# Metrics (columns = samples)
# ----------------------------
function rmse_per_sample(ŷ::AbstractMatrix, y::AbstractMatrix)
    @assert size(ŷ) == size(y) "Shapes must match"
    N = size(y, 2)
    [sqrt(mean((ŷ[:, i] .- y[:, i]).^2)) for i in 1:N]
end

"Relative L2 error of fields per sample: ‖K̂−K‖ / ‖K‖"
function rel_l2_per_sample(x̂::AbstractMatrix, x::AbstractMatrix; eps = 1e-12)
    @assert size(x̂) == size(x) "Shapes must match"
    N = size(x, 2)
    [norm(x̂[:, i] .- x[:, i]) / max(norm(x[:, i]), eps) for i in 1:N]
end

"Empirical CDF (sorted x, cumulative y)"
function ecdf(v::AbstractVector)
    s = sort(v)
    n = length(s)
    s, range(1 / n, 1; length = n)
end

# ----------------------------------------
# Main plotting function (revised)
# ----------------------------------------
function plot_model_comparison(
    bhp_true,
    bhp_pred_data,
    bhp_pred_phys,
    K_true,
    K_pred_data,
    K_pred_phys,
    casename;
    savepath::Union{Nothing, String} = nothing
)

    # ---- compute metrics per sample
    rmse_data = rmse_per_sample(bhp_pred_data, bhp_true)
    rmse_phys = rmse_per_sample(bhp_pred_phys, bhp_true)

    permerr_data = rel_l2_per_sample(K_pred_data, K_true)
    permerr_phys = rel_l2_per_sample(K_pred_phys, K_true)

    # Negative means physics-informed has lower error
    Δrmse = rmse_phys .- rmse_data
    Δpermerr = permerr_phys .- permerr_data

    # Positive means physics-informed improvement
    imp_rmse = rmse_data .- rmse_phys
    imp_permerr = permerr_data .- permerr_phys

    # quick summary in REPL
    println("Pressure RMSE median (data vs phys): ",
            round(median(rmse_data), sigdigits = 4), " vs ",
            round(median(rmse_phys), sigdigits = 4))
    println("Perm RelL2 median (data vs phys): ",
            round(median(permerr_data), sigdigits = 4), " vs ",
            round(median(permerr_phys), sigdigits = 4))
    println("% samples where phys wins (Pressure RMSE): ",
            round(100 * mean(Δrmse .< 0), digits = 1), "%")
    println("% samples where phys wins (Perm): ",
            round(100 * mean(Δpermerr .< 0), digits = 1), "%")

    rc("font", size = 12)

    # =========================================================
    # Figure A-D together: ECDFs on top, matching histograms below
    # =========================================================
     fig, axs = subplots(constrained_layout=true)

    # (a) Pressure ECDF
    x1, y1 = ecdf(rmse_data)
    x2, y2 = ecdf(rmse_phys)
    axs.plot(x1, y1, linewidth = 2, label = "Data driven")
    axs.plot(x2, y2, linewidth = 2, label = "Physics informed")
    # axs.set_title("(a) Pressure error: ECDF", fontsize = 16)
    axs.set_xlabel("Pressure RMSE (MPa)", fontsize = 16)
    axs.set_ylabel("Cumulative fraction", fontsize = 16)
    axs.grid(true, alpha = 0.3)
    axs.legend(fontsize = 12)
    axs.tick_params(labelsize = 12)

    display(fig)
    fig.savefig("Figure$(casename)a.pdf")
    println()
    PyPlot.close(fig)

    # (b) Permeability ECDF
    fig, axs = subplots(constrained_layout=true)
    x3, y3 = ecdf(permerr_data)
    x4, y4 = ecdf(permerr_phys)
    axs.plot(x3, y3, linewidth = 2, label = "Data driven")
    axs.plot(x4, y4, linewidth = 2, label = "Physics informed")
    # axs.set_title("(b) Permeability error: ECDF", fontsize = 16)
    axs.set_xlabel("Permeability relative L2 error", fontsize = 16)
    axs.set_ylabel("Cumulative fraction", fontsize = 16)
    axs.grid(true, alpha = 0.3)
    axs.legend(fontsize = 14)
    axs.tick_params(labelsize = 14)
    display(fig)
    fig.savefig("Figure$(casename)b.pdf")
    println()
    PyPlot.close(fig)


    # (c) Pressure histogram
    fig, axs = subplots(constrained_layout=true)
    axs.hist(rmse_data, bins = 50, alpha = 0.6, label = "Data driven", density = true)
    axs.hist(rmse_phys, bins = 50, alpha = 0.6, label = "Physics informed", density = true)
    # axs.set_title("(c) Pressure error: histogram", fontsize = 16)
    axs.set_xlabel("Pressure RMSE (MPa)", fontsize = 16)
    axs.set_ylabel("Density", fontsize = 16)
    axs.grid(true, alpha = 0.3)
    axs.legend(fontsize = 14)
    axs.tick_params(labelsize = 14)
    display(fig)
    fig.savefig("Figure$(casename)c.pdf", bbox_inches = "tight")
    println()
    PyPlot.close(fig)

    # (d) Permeability histogram
    fig, axs = subplots(constrained_layout=true)
    axs.hist(permerr_data, bins = 50, alpha = 0.6, label = "Data driven", density = true)
    axs.hist(permerr_phys, bins = 50, alpha = 0.6, label = "Physics informed", density = true)
    # axs.set_title("(d) Permeability error: histogram", fontsize = 16)
    axs.set_xlabel("Permeability relative L2 error", fontsize = 16)
    axs.set_ylabel("Density", fontsize = 16)
    axs.grid(true, alpha = 0.3)
    axs.legend(fontsize = 14)
    axs.tick_params(labelsize = 14)
    display(fig)
    fig.savefig("Figure$(casename)d.pdf", bbox_inches = "tight")
    println()
    PyPlot.close(fig)


    # =========================================================
    # (e) Clearer grouped boxplot with twin y-axis
    # =========================================================
    fig, axL = subplots(figsize = (8.5, 6.0), constrained_layout = true)

    # Pressure error boxplots on left
    bpL = axL.boxplot(
        [rmse_data, rmse_phys];
        positions = [1.0, 2.0],
        widths = 0.45,
        showfliers = false,
        patch_artist = true,
        labels = ["DD", "PI"]
    )
    axL.set_ylabel("Pressure RMSE (MPa)", fontsize = 16)
    axL.grid(true, axis = "y", alpha = 0.3)

    # Permeability error boxplots on right
    axR = axL.twinx()
    bpR = axR.boxplot(
        [permerr_data, permerr_phys];
        positions = [4.0, 5.0],
        widths = 0.45,
        showfliers = false,
        patch_artist = true,
        labels = ["DD", "PI"]
    )
    axR.set_ylabel("Permeability relative L2 error", fontsize = 16)

    # Visual grouping
    axL.axvspan(0.5, 2.5, alpha = 0.08)
    axL.axvspan(3.5, 5.5, alpha = 0.08)

    axL.set_xlim(0.3, 5.7)
    axL.set_xticks([1.0, 2.0, 4.0, 5.0])
    axL.set_xticklabels(["DD", "PI", "DD", "PI"], fontsize = 14)
    axL.tick_params(labelsize = 14)
    axR.tick_params(labelsize = 14)

    # Box transparency
    for b in bpL["boxes"]
        b.set_alpha(0.45)
    end
    for b in bpR["boxes"]
        b.set_alpha(0.45)
    end

    # axL.set_title("(e) Distribution summary: pressure vs permeability", fontsize = 16)

    # Group labels
    y0, y1 = axL.get_ylim()
    txt_y = y0 - 0.08 * (y1 - y0)
    axL.text(1.5, txt_y, "Pressure error", ha = "center", va = "top", fontsize = 15)
    axL.text(4.5, txt_y, "Permeability error", ha = "center", va = "top", fontsize = 15)

    display(fig)
    fig.savefig("Figure$(casename)e.pdf", bbox_inches = "tight")
    println()
    PyPlot.close(fig)

    # =========================================================
    # (f) Scatter with explicit "physics-informed improvement"
    # =========================================================
    fig, axs = subplots(figsize = (8.5, 6.5), constrained_layout = true)

    axs.scatter(imp_permerr, imp_rmse, s = 10, alpha = 0.5)
    axs.axvline(0, linestyle = "--", linewidth = 1, color = "red")
    axs.axhline(0, linestyle = "--", linewidth = 1, color = "red")

    # axs.set_title("(f) Sample-wise improvement of physics-informed vs data-driven", fontsize = 16)
    axs.set_xlabel("PI improvement in permeability error\n(DD error - PI error)", fontsize = 16)
    axs.set_ylabel("PI improvement in pressure error\n(DD error - PI error)", fontsize = 16)
    axs.grid(true, alpha = 0.3)
    axs.tick_params(labelsize = 14)

    # Expand axes a bit for quadrant labels
    xmin, xmax = minimum(imp_permerr), maximum(imp_permerr)
    ymin, ymax = minimum(imp_rmse), maximum(imp_rmse)
    dx = max(xmax - xmin, 1e-8)
    dy = max(ymax - ymin, 1e-8)

    axs.set_xlim(xmin - 0.20 * dx, xmax + 0.30 * dx)
    axs.set_ylim(ymin - 0.20 * dy, ymax + 0.30 * dy)

    xmin2, xmax2 = axs.get_xlim()
    ymin2, ymax2 = axs.get_ylim()

    # Quadrant labels
    axs.text(0.55 * xmax2, 0.78 * ymax2,
             "PI improves\npressure and permeability",
             ha = "center", va = "center", fontsize = 14)

    axs.text(0.55 * xmax2, 0.70 * ymin2,
             "PI improves\npermeability only",
             ha = "center", va = "center", fontsize = 14)

    axs.text(0.55 * xmin2, 0.78 * ymax2,
             "PI improves\npressure only",
             ha = "center", va = "center", fontsize = 14)

    axs.text(0.45 * xmin2, 0.70 * ymin2,
             "DD improves\npressure and permeability",
             ha = "center", va = "center", fontsize = 14)

    display(fig)
    fig.savefig("Figure$(casename)f.pdf", bbox_inches = "tight")
    println()
    PyPlot.close(fig)

    return nothing
end

# ----------------------------
# Problem Setup
# ----------------------------
mutable struct Fluid
    vw::Float64
    vo::Float64
    swc::Float64
    sor::Float64
end

n = 51
ns = (n, n)
sidelength = 100.0
thickness = 1.0
num_eig = 200
num_cells = n * n

# case data
casename = "BBB"
num_mon = 200
λ_cov = 100.0
@BSON.load "ModelPhysicsBBBv2n10.bson" model
modelDPFEHM = model
@BSON.load "ModelDataBBBv2n10.bson" model
modelDATA = model

# ---------------------------------------------------------
# Alternative cases (uncomment as needed)
# ---------------------------------------------------------
# casename="BBS"
# num_mon  = 200
# λ_cov   = 10.0
# @BSON.load "ModelPhysicsBBS.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataBBS.bson" model
# modelDATA=model

# casename="BSB"
# num_mon   = 10
# λ_cov   = 100.0
# @BSON.load "ModelPhysicsBSB.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataBSB.bson" model
# modelDATA=model

# casename="BSS"
# num_mon = 10
# λ_cov   = 10.0
# @BSON.load "ModelPhysicsBSS.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataBSS.bson" model
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

# casename="SSS"
# num_mon = 10
# λ_cov   = 10.0
# @BSON.load "ModelPhysicsSSS.bson" model
# modelDPFEHM=model
# @BSON.load "ModelDataSSS.bson" model
# modelDATA=model

println("  ▸ Generating 2D regular grid …")
coords, neighbors, areasoverlengths, volumes =
    DPFEHM.regulargrid2d(
        [-sidelength, -sidelength],
        [sidelength, 2sidelength],
        ns,
        thickness
    )

h0 = zeros(size(coords, 2))
fluid = Fluid(1.0, 1.0, 0.0, 0.0)
S0 = zeros(size(coords, 2))
nt = 1
dt = 3 * 24 * 60 * 60

steadyhead = 0.0
dirichletnodes = Int[]
dirichleths = zeros(size(coords, 2))
injection_node = 26

for i = 1:size(coords, 2)
    if coords[1, i] == sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = steadyhead
    end
end

for i = 1:size(coords, 2)
    if coords[1, i] == -sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = 10
    end
end

for i = 1:size(coords, 2)
    if coords[2, i] == maximum(coords[2, :])
        push!(dirichletnodes, i)
        dirichleths[i] = 0.5
    elseif coords[2, i] == minimum(coords[2, :])
        push!(dirichletnodes, i)
        dirichleths[i] = 0.0
    end
end

injrate = 0.0

Random.seed!(1256879)
monitoring_nodes = sort!(randperm(num_cells)[1:num_mon])
println("    → Selected $num_mon monitoring nodes: ", monitoring_nodes)

# ----------------------------
# KL setup
# ----------------------------
println("  ▸ Building GaussianRandomFields KL …")
σ_cov = 1.0

cov_func = GaussianRandomFields.CovarianceFunction(
    2,
    GaussianRandomFields.Matern(λ_cov, 1; σ = σ_cov)
)

x_min, x_max = minimum(coords[1, :]), maximum(coords[1, :])
y_min, y_max = minimum(coords[2, :]), maximum(coords[2, :])

x_pts = range(x_min, x_max; length = n)
y_pts = range(y_min, y_max; length = n)

grf = GaussianRandomFields.GaussianRandomField(
    cov_func,
    GaussianRandomFields.KarhunenLoeve(num_eig),
    x_pts,
    y_pts
)

println("    → Extracting eigenfunctions/values …")
ϕ_matrix = grf.data.eigenfunc
σ_vec = grf.data.eigenval

# ----------------------------
# Forward solver
# ----------------------------
function getQs(Qs, is)
    sum(Qs .* ((collect(1:size(coords, 2)) .== i) for i in is))
end

logKs2Ks_neighbors(Ks) =
    exp.(0.5 * (Ks[map(p -> p[1], neighbors)] .+ Ks[map(p -> p[2], neighbors)]))

function solve_bhp(logK_vec, monitoring_nodes)
    Q_vec = getQs([injrate], [injection_node])

    Ks_neighbors = logKs2Ks_neighbors(reshape(logK_vec, n, n))

    P_full = DPFEHM.groundwater_steadystate(
        Ks_neighbors,
        neighbors,
        areasoverlengths,
        dirichletnodes,
        dirichleths,
        Q_vec
    )

    return @view P_full[monitoring_nodes]
end

println("  ▸ Testing solver once …")
test_logK = randn(num_cells)
bhp_test = solve_bhp(test_logK, monitoring_nodes)
println("    → Solver OK: returned BHPs of size ", size(bhp_test))

# ----------------------------
# Generate evaluation samples
# ----------------------------
x_true = 0
N_val = 1000
Random.seed!()

true_bhps = Matrix{Float32}(undef, num_mon, N_val)
true_xs = Matrix{Float32}(undef, num_eig, N_val)

xrand = 123
for i = 1:N_val
    Random.seed!(xrand + i)
    x_true = randn(Float32, num_eig)
    logK_vec = ϕ_matrix * (σ_vec .* Float64.(x_true))
    bhp_vec = solve_bhp(logK_vec, monitoring_nodes)
    @inbounds begin
        true_bhps[:, i] = Float32.(bhp_vec)
        true_xs[:, i] = x_true
    end
end

test_bhps = Matrix{Float32}(undef, num_mon, N_val)
pred_bhps_data = Matrix{Float32}(undef, num_mon, N_val)
pred_xs_data = Matrix{Float32}(undef, num_eig, N_val)

pred_bhps_sim = Matrix{Float32}(undef, num_mon, N_val)
pred_xs_sim = Matrix{Float32}(undef, num_eig, N_val)

perms_true = Matrix{Float32}(undef, n * n, N_val)
perms_data = Matrix{Float32}(undef, n * n, N_val)
perms_sim = Matrix{Float32}(undef, n * n, N_val)

for i = 1:N_val
    x_pred_data = modelDATA(true_bhps[:, i])
    x_pred_sim = modelDPFEHM(true_bhps[:, i])

    logK_vec_true = ϕ_matrix * (σ_vec .* Float64.(true_xs[:, i]))
    logK_vec_data = ϕ_matrix * (σ_vec .* Float64.(x_pred_data))
    logK_vec_sim = ϕ_matrix * (σ_vec .* Float64.(x_pred_sim))

    test_bhps[:, i] = solve_bhp(logK_vec_true, monitoring_nodes)
    bhp_vec_data = solve_bhp(logK_vec_data, monitoring_nodes)
    bhp_vec_sim = solve_bhp(logK_vec_sim, monitoring_nodes)

    pred_bhps_data[:, i] = bhp_vec_data
    pred_bhps_sim[:, i] = bhp_vec_sim

    pred_xs_data[:, i] = x_pred_data
    pred_xs_sim[:, i] = x_pred_sim

    perms_true[:, i] = logK_vec_true
    perms_data[:, i] = logK_vec_data
    perms_sim[:, i] = logK_vec_sim
end

true_bhps = test_bhps

bhp_true = true_bhps
bhp_pred_data = pred_bhps_data
bhp_pred_phys = pred_bhps_sim
K_true = perms_true
K_pred_data = perms_data
K_pred_phys = perms_sim

BSON.@save "Evaluation_$casename.bson" bhp_true bhp_pred_data bhp_pred_phys K_true K_pred_data K_pred_phys

plot_model_comparison(
    bhp_true,
    bhp_pred_data,
    bhp_pred_phys,
    K_true,
    K_pred_data,
    K_pred_phys,
    casename;
    savepath = "$casename.pdf"
)