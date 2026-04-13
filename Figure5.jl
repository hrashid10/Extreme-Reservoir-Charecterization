using DataFrames
using CSV
import BSON
using Flux


casename="BBB"
# casename="BBS"
# casename="BSB"
# casename="BSS"
# casename="SBB"
# casename="SBS"
# casename="SSB"
# casename="SSS"




# dict1=BSON.load("DataBBBv2REonlyn10.bson")
# dict2=BSON.load("PhysicsBBBv2REonlyn10.bson")


# dict1=BSON.load("DataBBBv2REn10.bson")
# dict2=BSON.load("PhysicsBBBv2Ren10.bson")

dict1=BSON.load("Data$(casename)v2n10.bson")
dict2=BSON.load("Physics$(casename)v2n10.bson")

yl=10000
# df = DataFrame(rmses_train = dict[:train_losses], rmses_test = dict[:val_losses][1:end], rmses_bhp = dict[:bhp_losses][1:end], rmses_perm = dict[:perm_losses][1:end])
df1 = DataFrame(rmses_bhp = dict1[:bhp_losses][1:yl], rmses_perm = dict1[:perm_losses][1:yl])
df2 = DataFrame(rmses_bhp = dict2[:bhp_losses][1:yl], rmses_perm = dict2[:perm_losses][1:yl])

using PyPlot
using PyCall
@pyimport matplotlib.patches as mpatches
@pyimport matplotlib.transforms as mtransforms

num_epoch=10000
epochs=1:yl


fig, ax = subplots(dpi=1200, constrained_layout=true)
# Plot data
ax.plot(epochs', df1.rmses_perm, label="Perm coeffcient MSE, Data Driven", color="red")
# ax.plot(epochs', df1.rmses_test, label="Validation MSE Data Driven", color="red")
ax.plot(epochs', df2.rmses_perm, label="Perm coeffcient MSE, Physics Informed", color="blue")
# ax.plot(epochs', df2.rmses_test, label="Validation MSE Physics Informed", color="black")

# Set log scale on y-axis
# ax.set_yscale("log")

# Axis labels and title
ax.set_xlabel("Iterations", fontsize=14)
ax.set_ylabel("Perm coeffcient MSE", fontsize=14)

ax.set_yscale("log")
ax.set_xscale("log")
# Enable minor ticks
ax.minorticks_on()
# Grid: Only on y-axis
ax.yaxis.grid(true, which="major", linestyle=":", linewidth=1)    # major y grid
# ax.yaxis.grid(true, which="minor", linestyle=":", linewidth=0.5)  # minor y grid

ax.xaxis.grid(true, which="major", linestyle=":", linewidth=1)    # major y grid
# ax.xaxis.grid(false, which="minor")  # minor y grid

# Legend
ax.legend(loc="upper right",fontsize=14)
ax.tick_params(labelsize=14)
# Display plot
display(fig)
fig.savefig("Figure$(casename)aPermEr.pdf", dpi=1200)
PyPlot.close(fig)


fig, ax = subplots(dpi=1200, constrained_layout=true)
# Plot data
# ax.plot(epochs', df1.rmses_train, label="Training MSE Data Driven", color="blue")
ax.plot(epochs', df1.rmses_bhp, label="Pressure MSE, Data Driven", color="red")
# ax.plot(epochs', df2.rmses_train, label="Training MSE Physics Informed", color="green")
ax.plot(epochs', df2.rmses_bhp, label="Pressure MSE,  Physics Informed", color="blue")

# Set log scale on y-axis
ax.set_yscale("log")
ax.set_xscale("log")

# Axis labels and title
ax.set_xlabel("Iterations", fontsize=14)
ax.set_ylabel(" Pressure MSE, MPa", fontsize=14)

# ax.set_yscale("log")
# Enable minor ticks
ax.minorticks_on()
# Grid: Only on y-axis
ax.yaxis.grid(true, which="major", linestyle=":", linewidth=1)    # major y grid
# ax.yaxis.grid(true, which="minor", linestyle=":", linewidth=0.5)  # minor y grid

ax.xaxis.grid(true, which="major", linestyle=":", linewidth=1)    # major y grid
# ax.xaxis.grid(false, which="minor")  # minor y grid

# Legend
ax.legend(loc="upper right", fontsize=14)
ax.tick_params(labelsize=14)
# Display plot
display(fig)
fig.savefig("Figure$(casename)bPresEr.pdf", dpi=1200)
PyPlot.close(fig)
