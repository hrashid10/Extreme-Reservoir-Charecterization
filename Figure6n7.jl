################################################################################
# Compare data-driven vs physics-informed models (DPFEHM) with better evaluation
# Suggestions implemented:
#   (1) Unified evaluation over many validation samples with metrics
#   (2) Consistent plotting with shared color limits + colorbar units
#   (3) Summary comparison (metrics + ECDF curves) instead of single example only
################################################################################

using Random
using Statistics: mean, std
using Flux
using DPFEHM
using GaussianRandomFields
using BSON
using PyPlot

# ---------------------------
# Case / model configuration
# ---------------------------
casename = "BBB"
num_mon  = 200
λ_cov    = 100.0
xrand    = 12345                # fixed seed baseline for validation
model_phys_path = "ModelPhysicsBBBv2n10.bson"
model_data_path = "ModelDataBBBv2n10.bson"

@BSON.load model_phys_path model
modelDPFEHM = model
@BSON.load model_data_path model
modelDATA   = model

# ---------------------------
# Problem setup (grid, BCs)
# ---------------------------
n           = 51
ns          = (n, n)
sidelength  = 100.0                  # [m]
thickness   = 1.0                    # [m]
num_eig     = 200
num_cells   = n * n
injection_node = 26
injrate     = 0.0                    # [m^3/s]

println("▸ Building 2D grid …")
coords, neighbors, areasoverlengths, volumes =
    DPFEHM.regulargrid2d([-sidelength, -sidelength], [sidelength, 2sidelength], ns, thickness)

# Dirichlet BCs
steadyhead = 0.0
dirichletnodes = Int[]
dirichleths = zeros(size(coords, 2))

# Right boundary x = +sidelength
for i in 1:size(coords, 2)
    if coords[1, i] == sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = steadyhead
    end
end

# Left boundary x = -sidelength
for i in 1:size(coords, 2)
    if coords[1, i] == -sidelength
        push!(dirichletnodes, i)
        dirichleths[i] = 10.0
    end
end

# Top/bottom y boundaries
for i in 1:size(coords, 2)
    if coords[2, i] == maximum(coords[2, :])         # top
        push!(dirichletnodes, i); dirichleths[i] = 0.5
    elseif coords[2, i] == minimum(coords[2, :])     # bottom
        push!(dirichletnodes, i); dirichleths[i] = 0.0
    end
end

# Monitoring nodes (fixed)
Random.seed!(1256879)
monitoring_nodes = sort!(randperm(num_cells)[1:num_mon])
println("▸ Selected $num_mon monitoring nodes.")

# ---------------------------
# KL setup (GRF)
# ---------------------------
println("▸ Building KL basis …")
σ_cov = 1.0
cov_func = GaussianRandomFields.CovarianceFunction(
    2, GaussianRandomFields.Matern(λ_cov, 1; σ = σ_cov)
)

x_min, x_max = minimum(coords[1, :]), maximum(coords[1, :])
y_min, y_max = minimum(coords[2, :]), maximum(coords[2, :])
x_pts = range(x_min, x_max; length=n)
y_pts = range(y_min, y_max; length=n)

grf = GaussianRandomFields.GaussianRandomField(
    cov_func,
    GaussianRandomFields.KarhunenLoeve(num_eig),
    x_pts,
    y_pts
)

ϕ_matrix = grf.data.eigenfunc           # (n*n) × num_eig
σ_vec    = grf.data.eigenval            # num_eig

# ---------------------------
# Helper functions
# ---------------------------

# Efficient Q vector (only one injection node here)
function make_Q_vec(injrate::Float64, inj_node::Int, ncell::Int)
    Q = zeros(Float64, ncell)
    Q[inj_node] = injrate
    return Q
end

Q_vec = make_Q_vec(injrate, injection_node, num_cells)

# Convert logK on cells -> transmissivity on neighbors (your original mapping)
logKs2Ks_neighbors(logK_cell_vec) = exp.(0.5 .* (logK_cell_vec[map(p->p[1], neighbors)] .+
                                               logK_cell_vec[map(p->p[2], neighbors)]))

# Forward solver: given KL coeffs x -> pressures at monitors
function solve_bhp_from_x(x_coeff::AbstractVector,
                          ϕ_matrix, σ_vec, n::Int,
                          neighbors, areasoverlengths, dirichletnodes, dirichleths, Q_vec,
                          monitoring_nodes)

    logK_cell = ϕ_matrix * (σ_vec .* Float64.(x_coeff))          # length n*n
    Ks_neighbors = logKs2Ks_neighbors(logK_cell)                 # length = number of edges/neighbors
    P_full = DPFEHM.groundwater_steadystate(
        Ks_neighbors, neighbors, areasoverlengths,
        dirichletnodes, dirichleths, Q_vec
    )
    return Float32.(P_full[monitoring_nodes]), Float32.(logK_cell)
end

# Full pressure field (all cells) from logK cell vector (length n*n)
function solve_pressure_full_from_logK(logK_cell::AbstractVector,
                                       n::Int, neighbors, areasoverlengths,
                                       dirichletnodes, dirichleths, Q_vec)
    Ks_neighbors = logKs2Ks_neighbors(Float64.(logK_cell))
    P_full = DPFEHM.groundwater_steadystate(
        Ks_neighbors, neighbors, areasoverlengths,
        dirichletnodes, dirichleths, Q_vec
    )
    return Float32.(P_full)
end

# Metrics
rmse(a, b) = sqrt(mean((a .- b).^2))
rel_l2(a, b) = sqrt(sum((a .- b).^2) / (sum(a.^2) + 1e-12))

# Simple ECDF
function ecdf_xy(v::AbstractVector)
    xs = sort(collect(v))
    ys = (1:length(xs)) ./ length(xs)
    return xs, ys
end

# ---------------------------
# Validation / comparison
# ---------------------------
N_val = 200   # <-- set this to something meaningful (e.g., 200–2000)

true_xs        = Matrix{Float32}(undef, num_eig, N_val)
true_bhps      = Matrix{Float32}(undef, num_mon, N_val)
true_logK      = Matrix{Float32}(undef, n*n, N_val)

pred_x_data    = Matrix{Float32}(undef, num_eig, N_val)
pred_x_phys    = Matrix{Float32}(undef, num_eig, N_val)

pred_bhp_data  = Matrix{Float32}(undef, num_mon, N_val)
pred_bhp_phys  = Matrix{Float32}(undef, num_mon, N_val)

pred_logK_data = Matrix{Float32}(undef, n*n, N_val)
pred_logK_phys = Matrix{Float32}(undef, n*n, N_val)

# Metrics arrays
perm_rmse_data = zeros(Float64, N_val)
perm_rmse_phys = zeros(Float64, N_val)
perm_rel_data  = zeros(Float64, N_val)
perm_rel_phys  = zeros(Float64, N_val)

bhp_rmse_data  = zeros(Float64, N_val)
bhp_rmse_phys  = zeros(Float64, N_val)

println("▸ Running validation on N_val = $N_val samples …")
for i in 1:N_val
    Random.seed!(xrand + i)
    x_true = randn(Float32, num_eig)
    true_xs[:, i] .= x_true

    bhp_true, logK_true = solve_bhp_from_x(
        x_true, ϕ_matrix, σ_vec, n,
        neighbors, areasoverlengths, dirichletnodes, dirichleths, Q_vec,
        monitoring_nodes
    )
    true_bhps[:, i] .= bhp_true
    true_logK[:, i] .= logK_true

    # Model predictions in KL space
    xhat_data = modelDATA(bhp_true)
    xhat_phys = modelDPFEHM(bhp_true)

    pred_x_data[:, i] .= Float32.(xhat_data)
    pred_x_phys[:, i] .= Float32.(xhat_phys)

    # Push through forward model for pressures + logK
    bhp_data, logK_data = solve_bhp_from_x(
        xhat_data, ϕ_matrix, σ_vec, n,
        neighbors, areasoverlengths, dirichletnodes, dirichleths, Q_vec,
        monitoring_nodes
    )
    bhp_phys, logK_phys = solve_bhp_from_x(
        xhat_phys, ϕ_matrix, σ_vec, n,
        neighbors, areasoverlengths, dirichletnodes, dirichleths, Q_vec,
        monitoring_nodes
    )

    pred_bhp_data[:, i] .= bhp_data
    pred_bhp_phys[:, i] .= bhp_phys
    pred_logK_data[:, i] .= logK_data
    pred_logK_phys[:, i] .= logK_phys

    # Metrics (logK-cell space + monitor pressures)
    perm_rmse_data[i] = rmse(logK_true, logK_data)
    perm_rmse_phys[i] = rmse(logK_true, logK_phys)
    perm_rel_data[i]  = rel_l2(logK_true, logK_data)
    perm_rel_phys[i]  = rel_l2(logK_true, logK_phys)

    bhp_rmse_data[i]  = rmse(bhp_true, bhp_data)
    bhp_rmse_phys[i]  = rmse(bhp_true, bhp_phys)
end

println("\n================ SUMMARY ================")
println("Permeability (logK) RMSE:   data = $(mean(perm_rmse_data)) ± $(std(perm_rmse_data))")
println("Permeability (logK) RMSE:   phys = $(mean(perm_rmse_phys)) ± $(std(perm_rmse_phys))")
println("Permeability (logK) rel L2: data = $(mean(perm_rel_data)) ± $(std(perm_rel_data))")
println("Permeability (logK) rel L2: phys = $(mean(perm_rel_phys)) ± $(std(perm_rel_phys))")
println("Monitor BHP RMSE:           data = $(mean(bhp_rmse_data)) ± $(std(bhp_rmse_data))")
println("Monitor BHP RMSE:           phys = $(mean(bhp_rmse_phys)) ± $(std(bhp_rmse_phys))")
println("=========================================\n")

# ---------------------------
# Pick one index to visualize
# ---------------------------
i_show = 2

logK_true = true_logK[:, i_show]
logK_data = pred_logK_data[:, i_show]
logK_phys = pred_logK_phys[:, i_show]

# Consistent limits for logK plots
vminK = minimum([minimum(logK_true), minimum(logK_data), minimum(logK_phys)])
vmaxK = maximum([maximum(logK_true), maximum(logK_data), maximum(logK_phys)])

# Error limits (squared error)
err_data = (logK_true .- logK_data).^2
err_phys = (logK_true .- logK_phys).^2
vminE = minimum([minimum(err_data), minimum(err_phys)])
vmaxE = maximum([maximum(err_data), maximum(err_phys)])

# ---------------------------
# Plot: permeability comparison
# ---------------------------
PyPlot.close("all")
fig, axs = PyPlot.subplots(1, 3, dpi=600,figsize=(8, 1.9), constrained_layout=true)

# Row 1: Data-driven
im1 = axs[1].imshow(reshape(logK_true, n, n), vmin=vminK, vmax=vmaxK, cmap="viridis")
axs[1].set_title("True logK")
cb = fig.colorbar(im1, ax=axs[1])
cb.set_label(L"log(k)\;\;[\,\log(\mathrm{m}^2)\,]")   # <-- unit label example (LaTeX)

im2 = axs[2].imshow(reshape(logK_data, n, n), vmin=vminK, vmax=vmaxK, cmap="viridis")
axs[2].set_title("Data-driven logK")
cb = fig.colorbar(im2, ax=axs[2])
cb.set_label(L"log(k)\;\;[\,\log(\mathrm{m}^2)\,]")

im3 = axs[3].imshow(reshape(err_data, n, n), vmin=vminE, vmax=vmaxE, cmap="viridis")
axs[3].set_title(" Data-driven error")
cb = fig.colorbar(im3, ax=axs[3])
cb.set_label(L"(logK\;error)^2")
fig.savefig("Figure$(casename)aPerm.pdf", dpi=600)
display(fig)

fig, axs = PyPlot.subplots(1, 3, dpi=600,figsize=(8, 1.9), constrained_layout=true)
# Row 2: Physics-informed
im4 = axs[1].imshow(reshape(logK_true, n, n), vmin=vminK, vmax=vmaxK, cmap="viridis")
axs[1].set_title("True logK")
cb = fig.colorbar(im4, ax=axs[1])
cb.set_label(L"log(k)\;\;[\,\log(\mathrm{m}^2)\,]")

im5 = axs[2].imshow(reshape(logK_phys, n, n), vmin=vminK, vmax=vmaxK, cmap="viridis")
axs[2].set_title("Physics-informed logK")
cb = fig.colorbar(im5, ax=axs[2])
cb.set_label(L"log(k)\;\;[\,\log(\mathrm{m}^2)\,]")

im6 = axs[3].imshow(reshape(err_phys, n, n), vmin=vminE, vmax=vmaxE, cmap="viridis")
axs[3].set_title("Physics-informed error")
cb = fig.colorbar(im6, ax=axs[3])
cb.set_label(L"(logK\;error)^2")

# fig.suptitle("Permeability comparison ($casename), sample = $i_show", fontsize=12)
fig.savefig("Figure$(casename)bPerm.pdf", dpi=600)
display(fig)

# ---------------------------
# Plot: pressure field comparison (full field)
# Convert to MPa (your scaling)
# ---------------------------
P_true = solve_pressure_full_from_logK(logK_true, n, neighbors, areasoverlengths, dirichletnodes, dirichleths, Q_vec)
P_data = solve_pressure_full_from_logK(logK_data, n, neighbors, areasoverlengths, dirichletnodes, dirichleths, Q_vec)
P_phys = solve_pressure_full_from_logK(logK_phys, n, neighbors, areasoverlengths, dirichletnodes, dirichleths, Q_vec)

# Convert head -> MPa (your earlier: * 9.8*1000*1e-6)
toMPa(x) = x .* (9.8f0 * 1000f0 * 1f-6)
P_true_mpa = toMPa(P_true)
P_data_mpa = toMPa(P_data)
P_phys_mpa = toMPa(P_phys)

vminP = minimum([minimum(P_true_mpa), minimum(P_data_mpa), minimum(P_phys_mpa)])
vmaxP = maximum([maximum(P_true_mpa), maximum(P_data_mpa), maximum(P_phys_mpa)])

eP_data = abs.(P_true_mpa .- P_data_mpa)
eP_phys = abs.(P_true_mpa .- P_phys_mpa)
vminPe = minimum([minimum(eP_data), minimum(eP_phys)])
vmaxPe = maximum([maximum(eP_data), maximum(eP_phys)])

fig, axs = PyPlot.subplots(1, 3, dpi=600,figsize=(8, 1.9), constrained_layout=true)

im1 = axs[1].imshow(reshape(P_true_mpa, n, n), vmin=vminP, vmax=vmaxP, cmap="jet")
axs[1].set_title("True Pressure")
cb = fig.colorbar(im1, ax=axs[1])
cb.set_label("Pressure (MPa)")  # <-- unit in colorbar

im2 = axs[2].imshow(reshape(P_data_mpa, n, n), vmin=vminP, vmax=vmaxP, cmap="jet")
axs[2].set_title("Data-driven Pressure")
cb = fig.colorbar(im2, ax=axs[2])
cb.set_label("Pressure (MPa)")

im3 = axs[3].imshow(reshape(eP_data, n, n), vmin=vminPe, vmax=vmaxPe, cmap="jet")
axs[3].set_title("Data-driven error")
cb = fig.colorbar(im3, ax=axs[3])
cb.set_label("Absolute error (MPa)")

display(fig)
fig.savefig("Figure$(casename)aPres.pdf", dpi=600)
println()
PyPlot.close(fig)


fig, axs = PyPlot.subplots(1, 3, dpi=600,figsize=(8, 1.9), constrained_layout=true)
im4 = axs[1].imshow(reshape(P_true_mpa, n, n), vmin=vminP, vmax=vmaxP, cmap="jet")
axs[1].set_title("True Pressure")
cb = fig.colorbar(im4, ax=axs[1])
cb.set_label("Pressure (MPa)")

im5 = axs[2].imshow(reshape(P_phys_mpa, n, n), vmin=vminP, vmax=vmaxP, cmap="jet")
axs[2].set_title("Physics-informed Pressure")
cb = fig.colorbar(im5, ax=axs[2])
cb.set_label("Pressure (MPa)")

im6 = axs[3].imshow(reshape(eP_phys, n, n), vmin=vminPe, vmax=vmaxPe, cmap="jet")
axs[3].set_title("Physics-informed error")
cb = fig.colorbar(im6, ax=axs[3])
cb.set_label("Absolute error (MPa)")


display(fig)
fig.savefig("Figure$(casename)bPres.pdf", dpi=600)
println()
PyPlot.close(fig)

# ---------------------------
# Plot: ECDF of metrics (summary comparison)
# ---------------------------
# fig3, ax3 = PyPlot.subplots(1, 1, dpi=300, figsize=(6.6, 4.0), constrained_layout=true)

# x1, y1 = ecdf_xy(bhp_rmse_data)
# x2, y2 = ecdf_xy(bhp_rmse_phys)
# ax3.plot(x1, y1, label="Data-driven (monitor RMSE)")
# ax3.plot(x2, y2, label="Physics-informed (monitor RMSE)")
# ax3.set_xlabel("Monitor pressure RMSE (solver units)")
# ax3.set_ylabel("ECDF")
# ax3.grid(true, alpha=0.25)
# ax3.legend()
# fig3.savefig("Figure_$(casename)_ECDF_MonitorRMSE.png", dpi=300)
# display(fig3)

println("Saved figures:")
println("  Figure_$(casename)_Perm_Compare.png")
println("  Figure_$(casename)_Pressure_Compare.png")
# println("  Figure_$(casename)_ECDF_MonitorRMSE.png")
