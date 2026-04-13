using PyPlot
using LinearAlgebra
using Statistics
using BSON
# Import packages
using Random
using Statistics: mean, std
using Flux
using Zygote
using DPFEHM
using GaussianRandomFields
import StatsBase
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

    # 4) Extract pressures at monitoring_nodes:
    return  P_full[critical_point]
end

# Test that the solver “runs” once on a random logK to ensure no errors:
println("  ▸ Testing solver once …")
test_logK = randn(num_cells)        # random logK field
bhp_test = solve_bhp(test_logK,monitoring_nodes)     # should return a length‐num_mon vector
println("    → Solver OK: returned BHPs of size ", size(bhp_test))
x_true=0
N_val=10000
Random.seed!()
true_bhps_bulk  = Matrix{Float32}(undef, num_mon, N_val)
true_xs_bulk = Matrix{Float32}(undef, num_eig, N_val)

xrand=123 # fixed for plotting(1234)

critical_point=2431
true_bhps_critical  = zeros(N_val)

for i in 1:N_val
    Random.seed!(xrand+i)
    x_true = randn(Float32, num_eig)
    logK_vec = ϕ_matrix * (σ_vec .* Float64.(x_true))
    bhp_vec   = solve_bhp(logK_vec,monitoring_nodes)
    bhp_critical   = solve_bhp_full(logK_vec,critical_point)
    @inbounds begin
      true_bhps_bulk[:,  i] = Float32.(bhp_vec)
      true_xs_bulk[:, i] = x_true
      true_bhps_critical[i] = bhp_critical
    end
end



fig, ax = PyPlot.subplots(dpi=1200)
ax.hist(true_bhps_critical; bins=200, edgecolor="black", facecolor="darkblue")

ax.set_xlabel("Pressure at the critical location, MPa", fontsize=15, fontname="Arial")
ax.set_ylabel("Frequency", fontsize=15, fontname="Arial")
for lbl in ax.get_xticklabels();  lbl.set_fontname("Arial"); lbl.set_fontsize(15); end
for lbl in ax.get_yticklabels();  lbl.set_fontname("Arial"); lbl.set_fontsize(15); end

# --- Mark top 1% ---
p99 = quantile(true_bhps_critical, 0.99)          # 99th percentile
xlo, xhi = ax.get_xlim()                          # current x-limits after the hist
ax.axvline(p99, color="crimson", linewidth=2, linestyle="--", label="99th percentile")
ax.axvspan(p99, xhi; color="crimson", alpha=0.15) # shade top 1% region

# Optional: annotate how many samples fall in the top 1%
n = length(true_bhps_critical)
n_top = count(>(p99), true_bhps_critical)
ax.text(0.95, 0.95,
        "Top 1%: \nthreshold = $(round(p99, sigdigits=4))",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=12, color="crimson")

ax.legend(frameon=true, fontsize=12)

display(fig)
fig.savefig("FigurePrescr.pdf", dpi=600, bbox_inches="tight")
close(fig)




p1 = StatsBase.percentile(true_bhps_critical, 99)

# p1 = StatsBase.percentile(true_bhps_critical, 99)
p2 = StatsBase.percentile(true_bhps_critical, 100)


# p1 = StatsBase.percentile(true_bhps_critical, 0)
# p2 = StatsBase.percentile(true_bhps_critical, 100)

# idxs_crtical = findall(>(p1), true_bhps_critical)   
idxs_critical = findall(x -> p1 < x < p2, true_bhps_critical)

critical_bhps  = true_bhps_bulk[:,idxs_critical]
critical_xs = true_xs_bulk[:,idxs_critical]


true_bhps=critical_bhps
true_xs=critical_xs
N_val=length(idxs_critical)
test_bhps  = Matrix{Float32}(undef, num_mon, N_val)
pred_bhps_data  = Matrix{Float32}(undef, num_mon, N_val)
pred_xs_data = Matrix{Float32}(undef, num_eig, N_val)


pred_bhps_sim  = Matrix{Float32}(undef, num_mon, N_val)
pred_xs_sim= Matrix{Float32}(undef, num_eig, N_val)


perms_true = Matrix{Float32}(undef, n*n, N_val)
perms_data = Matrix{Float32}(undef, n*n, N_val)
perms_sim= Matrix{Float32}(undef, n*n, N_val)
logK_vec_true = ϕ_matrix * (σ_vec .* Float64.(true_xs[:, 1] ))


fig, ax = PyPlot.subplots(dpi=1200)
xmin, xmax = -sidelength, sidelength
ymin, ymax = -sidelength, sidelength

x = range(xmin, xmax, length=n)
y = range(ymin, ymax, length=n)

# Plot using imshow with correct spatial coordinates using extent
img = ax.imshow(
    reshape(logK_vec_true, ns[1], ns[2]),
    origin="lower",
    cmap="viridis",
    extent=[xmin, xmax, ymin, ymax],
    aspect="equal",  # or use "equal"
    # interpolation="bicubic"
)

# cb = fig.colorbar(img)
# cb.ax.tick_params(labelsize=12)  # Adjust 12 to your desired font size
# cb.set_label("Permeability (m²)", fontsize=14)
monitoring_nodes=2431-500
# Cell numbers to mark
monit = monitoring_nodes
Inj = injection_node

# Convert linear indices to (row, column)
row_monit = div.(monit .- 1, ns[1]) .+ 1
col_monit = mod.(monit .- 1, ns[1]) .+ 1


row_Inj = div(Inj - 1, ns[1]) + 1+1
col_Inj = mod(Inj - 1, ns[1]) + 1

x = range(xmin, xmax, length=ns[1])
y = range(ymin, ymax, length=ns[2])
# Convert indices to coordinates
x_monit = x[col_monit]
y_monit = y[row_monit]

x_Inj = x[col_Inj]
y_Inj = y[row_Inj]

offset = sidelength * 0.05  # adjust as needed for spacing

# Marker for Monitoring Well
ax.scatter(x_monit, y_monit, marker="o", color="red", label="critical point")
ax[:tick_params](axis="both", which="major", labelsize=14)
ax.legend(fontsize=12,loc="upper right")
display(fig)
fig.savefig("Figure1cr.pdf",  dpi=600, bbox_inches="tight")
PyPlot.close(fig)

