# Trying out GR, StatsPlots, Plots
# Need:
# - Heatmaps
# - KDE
# - Histograms
# - Grouped line plots
# - Grouped scatter plots

import Pkg; Pkg.activate(".")
@time begin
  using Plots; gr()
end


x, y = randn(100), randn(100)
p1 = plot(x, y, label="p1")
p2 = plot(x, y, seriestype=:scatter, label="p2")
p3 = plot(x, seriestype=:hist, label="p3")
p4 = plot(y, seriestype=:hist, label="p4", color="red")
p = plot(p1, p2, p3, p4, layout=(2, 2))
savefig(p, "test.pdf")
