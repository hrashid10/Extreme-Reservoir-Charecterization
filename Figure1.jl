# Import packages
using Random
using Statistics: mean, std
using Flux
using Zygote
using DPFEHM
using GaussianRandomFields
using BSON
import PyPlot

n           = 101                  #  grid is n × n
ns          = (n, n)
sidelength  = 100.0                # [m] half‐width in each direction
thickness   = 1.0                  # [m]
num_eig     = 200                  #  number of KL modes we keep
num_cells   = n * n

# Build a uniform 2D grid from (–sidelength, –sidelength) to (sidelength, 2*sidelength)

println("  ▸ Generating 2D regular grid …")
coords, neighbors, areasoverlengths, volumes =DPFEHM.regulargrid2d([-sidelength, -sidelength],[sidelength, sidelength],ns,thickness)

# Monitoring nodes: pick random nodes (to mimic sparse pressure data).
# We fix the seed so that training/test splits are reproducible.
num_mon         = 50
Random.seed!(1256)
# Random.seed!(1256)

monitoring_nodes = sort!(randperm(num_cells)[1:num_mon])

println("    → Selected $num_mon monitoring nodes: ",
        monitoring_nodes)

#KL‐(Karhunen–Loève) setup 
# Build a Matern covariance (ν = 1, σ = 1.0, λ = 100.0) on our n×n grid
println("  ▸ Building GaussianRandomFields KL …")
σ_cov   = 1.0
λ_cov   = 20.0
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



x_true = randn(Float32, num_eig)                # a random vector of KL coefficients

logK_vec = ϕ_matrix * (σ_vec .* Float64.(x_true))   # (n*n) × 1
logK_vec=reshape(GaussianRandomFields.sample(grf),n*n)
# load a fully–differentiable “forward solver”

logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))#Zygote differentiates this efficiently but the definitions above are ineffecient with Zygote


#plotting


Ts=exp.(reshape(logK_vec,size(coords,2)))


fig, ax = PyPlot.subplots()
xmin, xmax = -sidelength, sidelength
ymin, ymax = -sidelength, sidelength

x = range(xmin, xmax, length=n)
y = range(ymin, ymax, length=n)
const mpl = PyPlot.matplotlib
# Plot using imshow with correct spatial coordinates using extent
img = ax.imshow(
    reshape(Ts, n, n),
    origin="lower",
    cmap="viridis",
    extent=[xmin, xmax, ymin, ymax],
    aspect="equal",  # or use "equal"
    # interpolation="bicubic"
    norm=mpl.colors.LogNorm()
)

cb = fig.colorbar(img)
cb.ax.tick_params(labelsize=12)  # Adjust 12 to your desired font size
cb.set_label("Permeability (m²)", fontsize=14)

# Cell numbers to mark
monit = monitoring_nodes

# Convert linear indices to (row, column)
row_monit = div.(monit .- 1, n) .+ 1
col_monit = mod.(monit .- 1, n) .+ 1

# Convert indices to coordinates
x_monit = x[col_monit]
y_monit = y[row_monit]

x_Inj = x[col_Inj]
y_Inj = y[row_Inj]

offset = sidelength * 0.05  # adjust as needed for spacing

# Marker for Monitoring Well
ax.scatter(x_monit, y_monit, marker="o", color="red", label="Monitoring Points")
ax[:tick_params](axis="both", which="major", labelsize=14)
ax.legend(fontsize=12,loc="upper right")
display(fig)
fig.savefig("Figure1.pdf", dpi=600)
PyPlot.close(fig)


