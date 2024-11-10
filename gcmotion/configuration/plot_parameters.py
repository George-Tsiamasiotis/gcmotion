# fmt: off
figsize = (16, 9) # Global window size
dpi = 90 # Global dpi

time_evolution = {

    "fig_parameters": {
        "figsize": figsize,
        "dpi": dpi,
        "sharex": True,
    },

    "scatter_args": {
        "s" : 0.2,
        "color" : "blue",
        "marker" : "o",
    },

    "labelpad" : 45,
    "loc" : "bottom",
    "ylabel_args": {
        "rotation": 0,
        "fontsize" : 15,
    },
}

qfactor_profile = {
    "figsize": (16, 6),

    "plot_params": {
        "color": "b",
        "linewidth": 2,
    },

    "vline_params": {
        "color": "r",
        "linewidth": 2,
    }
}

magnetic_profile = {
    "figsize": (16, 6),
    "grid_density": 100,

    "bcontour_params": {
        "levels": 100,
        "cmap": "PuBuGn",
    },

    "icontour_params": {
        "levels": 100,
        "cmap": "YlOrRd",
    },

    "gcontour_params": {
        "levels": 100,
        "cmap": "Purples",
    },
}

electric_profile = {
    "figsize": (16, 6),
    "grid_density": 200,

    "aspect_ratio": 0.5,

    "contour_params": {
        "levels": 100,
        "cmap": "PuBuGn",
    },

    "plot_params": {
        "color": "b",
        "linewidth": 2,
    },

    "vline_params": {
        "color": "r",
        "linewidth": 2,
    }
}

tokamak_profile = {
    "figsize": figsize,
    "contour_params": {
        "levels": 100,
        "cmap": "winter",
    },
}

drift = {

    "fig_parameters": {
        "figsize": figsize,
        "dpi": dpi,
    },

    "scatter_args": {
        "s" : 0.1,
        "color" : "red",
    },
    "hardylim": 3,
    "yfontsize": 16,
    "xfontsize": 12,
}

drifts = {

    "fig_parameters": {
        "figsize": figsize,
        "dpi": dpi,
    },

    "scatter_args": {
        "s" : 0.1,
        "color" : "red",
    },
    "hardylim": 3,
    "yfontsize": 16,
    "xfontsize": 12,
}
energy_contour = {

    "fig_parameters": {
        "figsize": figsize,
        "dpi": dpi,
    },
    "auto_yaxis_zoom": 0.75,
    "hardylim": 4, # times PSI_WALL
    "contour_grid_density" : 200,
    "contour_levels" : 25,
    "contour_cmap" : "plasma",
    "locator": "log", # "log" or anything else for default
    "log_base": 1.04, # Values closer to 1 seem to space out contour lines more evenly
    "cbar_color": "red",
}

parabolas = {
    
    "parabolas_normal_kw": {
        "color" : "b",
        "linewidth" : 0.6,
    },

    "parabolas_dashed_kw": {
        "color" : "b",
        "linewidth" : 0.6,
    },
}

orbit_point = {

    "orbit_point_kw": {
        "s" : 15,
        "marker" : "o",
        "edgecolor" : "k",
        "facecolor" : "red",
    },
}

torus2d = {

    "wall_points": 2000,

    "torus2d_wall_kw": {
        "color" : "k",
        "s" : 0.5,
    },
    
    "torus2d_orbit_kw": {
        "color" : "blue",
        "s" : 0.07,
    },
}

torus3d = {
    # Higher rstride/cstride and lower points give better performance
    "rstride": 5,
    "cstride": 5,
    "wall_points": 301,

    "torus3d_wall_kw": {
        "color" : "cyan",
        "alpha" : 0.3,
    },

    "torus3d_orbit_kw": {
        "color" : "red",
        "alpha" : 0.6,
        "linewidth" : 0.2,
    },
}

poloidal_cut = {
    "ylim": 1.1, #times the minor radius
    "wall_points": 2000,

    "orbit_kw": {
        "color" : "blue",
        "s" : 0.05,
        "marker": "."
    },

    "axis_size": 60,
    "axis_kwargs": {
        "facecolor": "r",
        "edgecolor": "k",
        "linewidths": 2,
    },

    "wall_kw": {
        "color" : "k",
        "s" : 0.5,
    },
}
