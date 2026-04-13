using DataFrames
using CSV
import BSON
using Flux
using PyPlot
using PyCall

casename="BBB"

dict1 = BSON.load("Data$(casename)v2n10.bson")
dict2 = BSON.load("Physics$(casename)v2n10.bson")

df1 = DataFrame(rmses_train = dict1[:train_losses], rmses_test = dict1[:val_losses][1:end])
df2 = DataFrame(rmses_train = dict2[:train_losses], rmses_test = dict2[:val_losses][1:end])

num_epoch = 10000
epochs = 1:num_epoch

# --- compute common axis limits ---
all_y = vcat(
    df1.rmses_train,
    df2.rmses_train,
    df1.rmses_test,
    df2.rmses_test
)

# for log scale, keep only positive values
all_y = all_y[all_y .> 0]

ymin = minimum(all_y)
ymax = maximum(all_y)

# optional: add a little padding
ymin *= 0.9
ymax *= 1.1

xmin = 1
xmax = num_epoch

# ---------------- First plot: training ----------------
fig, ax = subplots(dpi=600, constrained_layout=true)

ax.plot(epochs, df1.rmses_train, label="Training MSE Data Driven", color="red")
ax.plot(epochs, df2.rmses_train, label="Training MSE Physics Informed", color="blue")

ax.set_xlabel("Iterations", fontsize=16)
ax.set_ylabel("MSE", fontsize=16)
ax.set_yscale("log")
ax.set_xscale("log")

# set same axis limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.minorticks_on()
ax.yaxis.grid(true, which="major", linestyle=":", linewidth=1)
ax.xaxis.grid(true, which="major", linestyle=":", linewidth=1)

ax.legend(loc="upper right")
ax.tick_params(labelsize=16)

display(fig)
fig.savefig("Figure$(casename)aOE.pdf", dpi=600)
PyPlot.close(fig)

# ---------------- Second plot: validation ----------------
fig, ax = subplots(dpi=600, constrained_layout=true)

ax.plot(epochs, df1.rmses_test, label="Validation MSE Data Driven", color="red")
ax.plot(epochs, df2.rmses_test, label="Validation MSE Physics Informed", color="blue")

ax.set_xlabel("Iterations", fontsize=16)
ax.set_ylabel("MSE", fontsize=16)
ax.set_yscale("log")
ax.set_xscale("log")

# set same axis limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.minorticks_on()
ax.yaxis.grid(true, which="major", linestyle=":", linewidth=1)
ax.xaxis.grid(true, which="major", linestyle=":", linewidth=1)

ax.legend(loc="upper right")
ax.tick_params(labelsize=16)

display(fig)
fig.savefig("Figure$(casename)bOE.pdf", dpi=600)
PyPlot.close(fig)