import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

kappa = 1.1e-7         # [m2/s]
temp_init = 10.0       # [C]
temp_boundary = 100.0  # [C]
rmax = 4 * 0.01 * 0.5  # half-diameter [m]
a_factor = 0.0         # 0 for slab, 1 for cylinder, 2 for sphere

tmax = 100.0 * 1.5 * (rmax / 0.005) ** 2
delta_t = 0.01 * (rmax / 0.005) ** 2
delta_r = rmax / 25.0
nr = int(rmax / delta_r)
nstep = int(tmax / delta_t)
plot_intvl = 200

def main():
    os.system("mkdir -p img")
    temperature = np.full((nr,), temp_init)
    temperature[-1] = temp_boundary  # outer boundary condition

    for i in range(nstep):
        if (i % plot_intvl == 0):
            plot_snap(temperature, i)
        dtdt = time_derivative(temperature)
        temperature[:] = temperature[:] + delta_t * dtdt[:]

    os.system("convert -delay 15 -loop 0 ./img/*.png ./img/all.gif")
    os.system("rm -f img/*.png")

def time_derivative(temperature):
    r = np.linspace(0, rmax, nr, endpoint=True) - delta_r / 2
    dtdt = np.empty((nr,))
    dtdt[-1] = 0.0     # outer boundary condition (constant)
    dtdt[1:-1] = kappa * \
        ((temperature[2:] - 2.0 * temperature[1:-1] + temperature[0:-2]) / (delta_r * delta_r) \
         + a_factor / r[1:-1] * (temperature[2:] - temperature[0:-2]) / (2 * delta_r))
    dtdt[0] = dtdt[1]  # inner boundary condition temperature[0] = temperature[1]
    return dtdt

def plot_snap(temperature, i):
    # save a snapshot at i-th step
    dict_beef = {'red':   ((0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (0.8, 0.6, 0.6), (1.0, 0.6, 0.6)),
                 'green': ((0.0, 0.2, 0.2), (0.5, 0.2, 0.2), (0.8, 0.5, 0.5), (1.0, 0.5, 0.5)),
                 'blue':  ((0.0, 0.3, 0.3), (0.5, 0.3, 0.3), (0.8, 0.4, 0.4), (1.0, 0.4, 0.4)) }
    cm_beef = LinearSegmentedColormap('beef', dict_beef)

    xdim = 10  # horizontal dimension for visualization
    data = np.zeros((nr*2,xdim))
    for k in range(xdim):  # plot both sides mirrored
        data[0:nr,   k] = temperature[::-1]
        data[nr:2*nr,k] = temperature
    fig, ax = plt.subplots(1)
    plt.imshow(data, cmap=cm_beef, interpolation="bilinear", aspect=0.1)
    plt.clim(0, 100.0)
    plt.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
    plt.tick_params(axis="y", which="both", left="off", right="off", labelleft="off")
    plt.colorbar(orientation="horizontal")

    levels = np.arange(0, 100, 10)
    extent = (-0.5, xdim-0.5, 0, nr*2-1)
    map2 = ax.contour(data, levels, colors="w", extent=extent)
    ax.clabel(map2, fmt="%02d", colors="w")

    plt.title("[%d cm, init %i deg, boundary %i deg] t = %isec" %
              (rmax*2*100, temp_init, temp_boundary, i*delta_t))
    plt.savefig("./img/%06i.png" % i)
    plt.close()

main()