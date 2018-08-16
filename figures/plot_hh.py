
import uncertainpy as un
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import h5py
import chaospy as cp

from HodgkinHuxley import HodgkinHuxley

from prettyplot import prettyPlot, set_xlabel, set_ylabel, get_colormap
from prettyplot import fontsize, labelsize, titlesize, spines_color, set_style
from prettyplot import prettyBar, get_colormap_tableu20



labelsize = 16
ticksize = 14

data = un.Data("valderrama.h5")
time = data["valderrama"].time
mean = data["valderrama"].mean
variance = data["valderrama"].variance
percentile_95 = data["valderrama"].percentile_95
percentile_5 = data["valderrama"].percentile_5
sobol = data["valderrama"].sobol_first
V = data["valderrama"].evaluations


colors = [(0.898, 0, 0), (0.976, 0.729, 0.196), (0.259, 0.431, 0.525), (0.4375, 0.13671875, 0.4375)]
style = "seaborn-white"
linewidth = 3



###############################
#   Single result             #
###############################


prettyPlot(time, V[2517], color=colors[2], style=style, linewidth=linewidth)
plt.xlabel("Time (ms)", fontsize=ticksize)
plt.ylabel("Membrane potential (mv)", fontsize=ticksize)

plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)

plt.xlim([5, 15])

plt.savefig("hh_single.png")


###############################
#   Three different results   #
###############################

prettyPlot(time, V[44], color=colors[1], style=style, linewidth=linewidth)
prettyPlot(time, V[2517], new_figure=False, color=colors[2], style=style, linewidth=linewidth)
prettyPlot(time, V[2382], new_figure=False, color=colors[0], style=style, linewidth=linewidth)
plt.xlabel("Time (ms)", fontsize=ticksize)
plt.ylabel("Membrane potential (mv)", fontsize=ticksize)

plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)

plt.xlim([5, 15])

plt.savefig("hh.pdf")
plt.savefig("hh.png")


###############################
#   Plot prediction interval  #
###############################


ax = prettyPlot(time, mean, color=colors[2], palette="deep", linewidth=2, style=style)

ax.set_ylabel("Membrane potential (mV)", fontsize=labelsize)
ax.set_xlabel("Time (ms)", fontsize=labelsize)

ax.fill_between(time,
                percentile_5,
                percentile_95,
                color=colors[2],
                alpha=0.6,
                linewidth=0)

plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)

plt.xlim([5, 15])

plt.savefig("hh_prediction.png")
plt.savefig("hh_prediction.pdf")



###############################
#   Plot mean and variance    #
###############################



# ax = prettyPlot(time, mean, color=colors[2], palette="deep", linewidth=2, style="seaborn-white")
# ax.set_ylabel(r"Mean (mV)", fontsize=labelsize, color=colors[2])
# ax.tick_params(axis="y", which="both", right="off", labelright="off", labelcolor=colors[2])
# ax.set_xlabel("Time (ms)", fontsize=labelsize, color="black")
# ax.spines["left"].set_edgecolor(colors[2])



# plt.xticks(fontsize=ticksize)

# ax.tick_params(axis="y", which="both", right="off", left="on", labelleft="on",
#                color=colors[2], labelcolor=colors[2], labelsize=ticksize)


# ax2 = ax.twinx()
# ax2.plot(time, variance, color=colors[0], linewidth=2)
# ax2.grid(False)

# ax2.spines["left"].set_edgecolor(colors[2])

# ax2.spines["right"].set_visible(True)
# ax2.spines["right"].set_edgecolor(colors[0])
# ax2.patch.set_visible(False)
# ax2.tick_params(axis="x", labelbottom="off")

# ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on", labelleft="off",
#                 color=colors[0], labelcolor=colors[0], labelsize=ticksize)


# plt.xticks(fontsize=ticksize)

# ax2.set_ylabel(r"Variance ($\mathrm{mV}^2$)", color=colors[0], fontsize=labelsize)
# plt.xlim([5, 15])
# plt.savefig("hh_mean.png")



###############################
#   Plot mean and std         #
###############################


ax = prettyPlot(time, mean, color=colors[2], palette="deep", linewidth=2, style=style, label="Mean")

prettyPlot(time, np.sqrt(variance), ax=ax, color=colors[0], palette="deep", linewidth=2, style=style, label="Standard deviation")


plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.legend(fontsize=ticksize)
plt.xlim([5, 15])



ax.set_ylabel("Membrane potential (mV)", fontsize=labelsize)
ax.set_xlabel("Time (ms)", fontsize=labelsize)

plt.savefig("hh_mean.png")




###############################
#   Plot sensitivity          #
###############################


gbar_K = data.uncertain_parameters.index("gbar_K")

ax = prettyPlot(time, sobol[gbar_K], color=colors[0], palette="deep", linewidth=2,
           style=style, label="Potassium")

ax.set_ylabel(r"Sensitivity", fontsize=labelsize)
ax.set_xlabel("Time (ms)", fontsize=labelsize)
ax.set_ylim([0, 1])
ax.set_xlim([min(time), max(time)])
ax.legend(fontsize=ticksize, loc=2)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)

plt.savefig("sensitivity.png")



###############################
#   Plot sensitivity and mean #
###############################

gbar_K = data.uncertain_parameters.index("gbar_K")
gbar_Na = data.uncertain_parameters.index("gbar_Na")
gbar_L = data.uncertain_parameters.index("gbar_L")

ax = prettyPlot(time, sobol[gbar_K], color=colors[0], palette="deep",
                linewidth=2, style="seaborn-white", label="Potassium")
ax.set_ylabel(r"Sensitivity", fontsize=labelsize, color="black")
ax.tick_params(axis="y", which="both", right="off", labelright="off", labelcolor="black")
ax.set_xlabel("Time (ms)", fontsize=labelsize, color="black")
# ax.spines["left"].set_edgecolor(colors[0])
ax.set_xlim([min(time), max(time)])


plt.xticks(fontsize=ticksize)

ax.tick_params(axis="y", which="both", right="off", left="on", labelleft="on",
               labelcolor="black", labelsize=ticksize)


ax2 = ax.twinx()
ax2.plot(time, mean, color=colors[2], linewidth=2)
ax2.grid(False)

# ax2.spines["left"].set_edgecolor("black")

ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_edgecolor(colors[2])
ax2.patch.set_visible(False)
ax2.tick_params(axis="x", labelbottom="off")

ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on", labelleft="off",
                color=colors[2], labelcolor=colors[2], labelsize=ticksize)

ax2.set_ylabel(r"Membrane potential (mV)", color=colors[2], fontsize=labelsize)


ax2.fill_between(time,
                 percentile_5,
                 percentile_95,
                 color=colors[2],
                 alpha=0.6,
                 linewidth=0,
                 label="Prediction interval")

ax.legend(fontsize=ticksize, loc=2)
plt.yticks(fontsize=ticksize)
ax.set_ylim([0, 1])
ax2.set_ylim([0, 75])
plt.savefig("sensitivity_mean_1.png")

# ax = prettyPlot(time, sobol[gbar_Na], color=colors[1], palette="deep",
#                 linewidth=2, style="seaborn-white", label="Sodium", ax=ax)

# plt.legend(["Mean", "Prediction interval", "Sodium"], fontsize=ticksize)
# plt.xticks(fontsize=ticksize)
# plt.yticks(fontsize=ticksize)
# ax.tick_params(axis="y", which="both", right="off", left="on", labelleft="on",
#                labelcolor="black", labelsize=ticksize)
# ax.set_ylabel(r"Sensitivity", fontsize=labelsize, color="black")
# ax.set_xlabel("Time (ms)", fontsize=labelsize, color="black")
# plt.savefig("sensitivity_mean_2.png")

# ax = prettyPlot(time, sobol[gbar_L], color="black", palette="deep",
#                 linewidth=2, style="seaborn-white", label="Leak", ax=ax)

# plt.legend(["Mean", "Prediction interval", "Sodium", "Lean"], fontsize=ticksize)
# plt.xticks(fontsize=ticksize)
# plt.yticks(fontsize=ticksize)
# ax.tick_params(axis="y", which="both", right="off", left="on", labelleft="on",
#                labelcolor="black", labelsize=ticksize)
# ax.set_ylabel(r"Sensitivity", fontsize=labelsize, color="black")
# ax.set_xlabel("Time (ms)", fontsize=labelsize, color="black")

# plt.savefig("sensitivity_mean_3.png")















###############################
#   Plot sensitivity and mean 2 #
###############################

gbar_K = data.uncertain_parameters.index("gbar_K")
gbar_Na = data.uncertain_parameters.index("gbar_Na")
gbar_L = data.uncertain_parameters.index("gbar_L")

ax = prettyPlot(time, sobol[gbar_K], color=colors[0], palette="deep",
                linewidth=2, style="seaborn-white", label="Potassium")
ax.set_ylabel(r"Sensitivity", fontsize=labelsize, color="black")
ax.tick_params(axis="y", which="both", right="off", labelright="off", labelcolor="black")
ax.set_xlabel("Time (ms)", fontsize=labelsize, color="black")
# ax.spines["left"].set_edgecolor(colors[0])
ax.set_xlim([min(time), max(time)])


plt.xticks(fontsize=ticksize)

ax.tick_params(axis="y", which="both", right="off", left="on", labelleft="on",
               labelcolor="black", labelsize=ticksize)


ax2 = ax.twinx()
ax2.plot(time, mean, color=colors[2], linewidth=2)
ax2.grid(False)

# ax2.spines["left"].set_edgecolor("black")

ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_edgecolor(colors[2])
ax2.patch.set_visible(False)
ax2.tick_params(axis="x", labelbottom="off")

ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on", labelleft="off",
                color=colors[2], labelcolor=colors[2], labelsize=ticksize)

ax2.set_ylabel(r"Membrane potential (mV)", color=colors[2], fontsize=labelsize)


ax2.fill_between(time,
                 percentile_5,
                 percentile_95,
                 color=colors[2],
                 alpha=0.6,
                 linewidth=0,
                 label="Prediction interval")



ax = prettyPlot(time, sobol[gbar_Na], color=colors[1], palette="deep",
                linewidth=2, style="seaborn-white", label="Sodium", ax=ax)

plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
ax.tick_params(axis="y", which="both", right="off", left="on", labelleft="on",
               labelcolor="black", labelsize=ticksize)
ax.set_ylabel(r"Sensitivity", fontsize=labelsize, color="black")
ax.set_xlabel("Time (ms)", fontsize=labelsize, color="black")
ax.legend(fontsize=ticksize, loc=2)
plt.yticks(fontsize=ticksize)
ax.set_ylim([0, 1])
ax2.set_ylim([0, 75])
plt.savefig("sensitivity_mean_2.png")






###############################
#   Plot sensitivity and mean 2 #
###############################

gbar_K = data.uncertain_parameters.index("gbar_K")
gbar_Na = data.uncertain_parameters.index("gbar_Na")
gbar_L = data.uncertain_parameters.index("gbar_L")

ax = prettyPlot(time, sobol[gbar_K], color=colors[0], palette="deep",
                linewidth=2, style="seaborn-white", label="Potassium")
ax.set_ylabel(r"Sensitivity", fontsize=labelsize, color="black")
ax.tick_params(axis="y", which="both", right="off", labelright="off", labelcolor="black")
ax.set_xlabel("Time (ms)", fontsize=labelsize, color="black")
# ax.spines["left"].set_edgecolor(colors[0])
ax.set_xlim([min(time), max(time)])


plt.xticks(fontsize=ticksize)

ax.tick_params(axis="y", which="both", right="off", left="on", labelleft="on",
               labelcolor="black", labelsize=ticksize)


ax2 = ax.twinx()
ax2.plot(time, mean, color=colors[2], linewidth=2)
ax2.grid(False)

# ax2.spines["left"].set_edgecolor("black")

ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_edgecolor(colors[2])
ax2.patch.set_visible(False)
ax2.tick_params(axis="x", labelbottom="off")

ax2.tick_params(axis="y", which="both", right="on", left="off", labelright="on", labelleft="off",
                color=colors[2], labelcolor=colors[2], labelsize=ticksize)

ax2.set_ylabel(r"Membrane potential (mV)", color=colors[2], fontsize=labelsize)


ax2.fill_between(time,
                 percentile_5,
                 percentile_95,
                 color=colors[2],
                 alpha=0.6,
                 linewidth=0,
                 label="Prediction interval")



ax = prettyPlot(time, sobol[gbar_Na], color=colors[1], palette="deep",
                linewidth=2, style="seaborn-white", label="Sodium", ax=ax)


ax = prettyPlot(time, sobol[gbar_L], color=colors[3], palette="deep",
                linewidth=2, style="seaborn-white", label="Leak", ax=ax)

ax.legend(fontsize=ticksize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
ax.tick_params(axis="y", which="both", right="off", left="on", labelleft="on",
               labelcolor="black", labelsize=ticksize)
ax.set_ylabel(r"Sensitivity", fontsize=labelsize, color="black")
ax.set_xlabel("Time (ms)", fontsize=labelsize, color="black")
ax.legend(fontsize=ticksize, loc=2)
plt.yticks(fontsize=ticksize)
ax.set_ylim([0, 1])
ax2.set_ylim([0, 75])
plt.savefig("sensitivity_mean_3.png")



###############################
#   Plot all sensitivity      #
###############################

linewidth = 2

titles = [r"Potassium conductance $\bar{g}_\mathrm{K}$", r"Sodium conductance $\bar{g}_\mathrm{Na}$", r"Leak conductance $\bar{g}_\mathrm{l}$"]

gbar_K = data.uncertain_parameters.index("gbar_K")
gbar_Na = data.uncertain_parameters.index("gbar_Na")
gbar_L = data.uncertain_parameters.index("gbar_L")
indices = [gbar_K, gbar_Na, gbar_L]

nr_plots = 3
grid_size = np.ceil(np.sqrt(nr_plots))
grid_x_size = int(grid_size)
grid_y_size = int(np.ceil(nr_plots/float(grid_x_size)))

fig, axes = plt.subplots(nrows=grid_y_size, ncols=grid_x_size, squeeze=False,
                         sharex="col", sharey="row")


labels = data.get_labels("valderrama")
xlabel, ylabel = labels

set_style("seaborn-white")
ax = fig.add_subplot(111, zorder=-10)
spines_color(ax, edges={"top": "None", "bottom": "None",
                        "right": "None", "left": "None"})
ax.tick_params(labelcolor="w", top="off", bottom="off", left="off", right="off")
ax.set_xlabel(xlabel.capitalize(), labelpad=8, fontsize=labelsize)
ax.set_ylabel("Sensitivity", labelpad=8, fontsize=labelsize)

for i in range(0, grid_x_size*grid_y_size):
    nx = i % grid_x_size
    ny = int(np.floor(i/float(grid_x_size)))

    ax = axes[ny][nx]

    if i < nr_plots:
        title = titles[i]

        index = indices[i]
        prettyPlot(time, sobol[index],
                color=colors[i],
                ax=ax,
                linewidth=linewidth,
                style=style)

        ax.set_title(title, fontsize=14)

        ax.set_ylim([-0.0, 1.0])
        ax.set_xlim([min(time), max(time)])
        ax.tick_params(labelsize=fontsize)
        ax.set_yticks([0, 0.5, 1])
    else:
        ax.axis("off")

plt.tight_layout()
plt.savefig("sensitivity_all.png")




###############################
#   Average isi mean          #
###############################



data = un.Data("brunel_AI.h5")


width = 0.2
distance = 0.5


# xlabels = ["Mean", "Standard deviation", "$P_5$", "$P_{95}$"]
# xticks = [0, width, distance + width, distance + 2*width]

# values = [data["average_isi"].mean, np.sqrt(data["average_isi"].variance),
#           data["average_isi"].percentile_5, data["average_isi"].percentile_95]

xlabels = ["Mean", "Standard deviation"]
xticks = [0, width]

values = [data["average_isi"].mean, np.sqrt(data["average_isi"].variance)]


ylabel = data.get_labels("average_isi")[0]

ax = prettyBar(values,
               index=xticks,
               xlabels=xlabels,
               ylabel=ylabel.capitalize(),
               palette=get_colormap_tableu20(),
               style="seaborn-white")

ax.set_ylabel("Time (ms)", fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)

plt.savefig("average_isi.png")


###############################
#   Average isi sensitivity   #
###############################




xlabels = [r"Potassium conductance $\bar{g}_\mathrm{K}$", r"Sodium conductance $\bar{g}_\mathrm{Na}$", r"Leak conductance $\bar{g}_\mathrm{l}$"]

xticks = [0, width + 0.1, 2*(width + 0.1)]


values = [data["average_isi"].mean, data["average_isi"].variance,
          data["average_isi"].percentile_5, data["average_isi"].percentile_95]

colors = [(0.898, 0, 0), (0.259, 0.431, 0.525), (0.976, 0.729, 0.196)]

labels = ["Input relative \n to threshold rate", "Inhibitory synapse \n strength", "Synaptic delay"]
ax = prettyBar(data["average_isi"].sobol_first,
               index=xticks,
               xlabels=labels,
               ylabel="Sensitivity",
               palette=colors,
               style="seaborn-white")

ax.set_ylabel("Sensitivity", fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)

for tick in ax.get_xticklabels():
    tick.set_rotation(-25)

plt.ylim([0, 1])

plt.tight_layout()


plt.savefig("average_isi_sensitivity.png")



###############################
#   MC vs PC                  #
###############################

# Load the analysed data
with h5py.File("pc_mc.h5", "r") as f:
    pc_evaluations_3 = f["pc_evaluations_3"][()]
    pc_mean_errors_3 = f["pc_mean_errors_3"][()]

    mc_evaluations_3 = f["mc_evaluations_3"][()]
    mc_mean_errors_3 = f["mc_mean_errors_3"][()]

    pc_evaluations_11 = f["pc_evaluations_11"][()]
    pc_mean_errors_11 = f["pc_mean_errors_11"][()]

    mc_evaluations_11 = f["mc_evaluations_11"][()]
    mc_mean_errors_11 = f["mc_mean_errors_11"][()]

# 11 uncertain parameters
ax = prettyPlot(pc_evaluations_11,
                pc_mean_errors_11,
                linewidth=linewidth,
                palette=colors,
                label="Polynomial chaos",
                style="seaborn-white")

prettyPlot(mc_evaluations_11,
           mc_mean_errors_11,
           linewidth=linewidth,
           palette=colors,
           new_figure=False,
           label="Monte Carlo",
           style="seaborn-white")


ax.set_ylabel("Average absolute relative\nerror over time", fontsize=labelsize)
ax.set_xlabel("Number of model evaluations", fontsize=labelsize)
ax.set_yscale("log")
ax.set_xlim([0, 10000])
# axes[1].set_ylim([5*10**-7, 10**8])
# axes[1].set_yticks([10**-6, 10**-3, 10**0, 10**3, 10**6])
ax.legend(fontsize=labelsize)

plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)

plt.tight_layout()
plt.savefig("mc_vs_pc.png")




colors = [(0.898, 0, 0), (0.976, 0.729, 0.196), (0.259, 0.431, 0.525)]


###############################
#   Three different results   #
###############################


scale1 = 1.5
scale2 = 0.5

model = HodgkinHuxley()

parameters_1 = {"gbar_Na": 120,
                "gbar_K": 36,
                "gbar_l": 0.5}


time, V, info = model.run(**parameters_1)
prettyPlot(time, V, color=colors[1], style=style, linewidth=linewidth)


parameters_2 = {"gbar_Na": scale1*120,
                "gbar_K": scale1*36,
                "gbar_l": scale1*0.5}

time, V, info = model.run(**parameters_2)
prettyPlot(time, V, new_figure=False, color=colors[0], style=style, linewidth=linewidth)



parameters_3 = {"gbar_Na": scale2*120,
                "gbar_K": scale2*36,
                "gbar_l": scale2*0.5}

time, V, info = model.run(**parameters_3)
prettyPlot(time, V, new_figure=False, color=colors[2], style=style, linewidth=linewidth)

plt.xlabel("Time (ms)", fontsize=ticksize)
plt.ylabel("Membrane potential (mv)", fontsize=ticksize)

plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)

plt.xlim([0, 23])
plt.xticks([0, 5, 10, 15, 20])
plt.ylim([-80, 43])
plt.savefig("features_why.pdf")


#####################################
#   Plot prediction interval old hh #
#####################################



parameters = {"gbar_Na": cp.Uniform(60, 180),       # 120
              "gbar_K": cp.Uniform(18, 54),         # 36
              "gbar_l": cp.Uniform(0.15, 0.45)}     # 0.3


# Initialize the model
model = HodgkinHuxley()

# Perform the uncertainty quantification
UQ = un.UncertaintyQuantification(model=model,
                                  parameters=parameters,
                                  features=un.SpikingFeatures())
data = UQ.quantify(plot=None, nr_pc_mc_samples=10**3, save=False)


time = data["HodgkinHuxley"].time
mean = data["HodgkinHuxley"].mean
percentile_95 = data["HodgkinHuxley"].percentile_95
percentile_5 = data["HodgkinHuxley"].percentile_5


ax = prettyPlot(time, mean, color=colors[2], palette="deep", linewidth=linewidth, style=style)

ax.set_ylabel(r"Membrane potential (mV)", fontsize=labelsize)
ax.set_xlabel("Time (ms)", fontsize=labelsize)


ax.fill_between(time,
                percentile_5,
                percentile_95,
                color=colors[2],
                alpha=0.6,
                linewidth=0)

plt.xlim([0, 23])
plt.xticks([0, 5, 10, 15, 20])


plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)


plt.savefig("features_why_prediction.png")