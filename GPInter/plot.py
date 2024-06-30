import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
from radiobuttons import MyRadioButtons

deep_blue = "#00008B"
tomato = "#D5120D"


def plot_gp(
    gp_list,
    X_pred,
    slider_bounds={
        "length_scale": (-0.1, 100.0, 0.1),  # (min, max, step)
        "amplitude": (-0.1, 100.0, 0.1),
        "p": (-0.1, 100.0, 0.1),
        "alpha": (-0.1, 100.0, 0.1),
    },
):
    if not isinstance(gp_list, list):
        gp_list = [gp_list]
    global shared_X, shared_Y

    fig, ax = plt.subplots(figsize=(10, 6))

    current_gp = gp_list[0]
    Y_init, X_init = current_gp.Y.copy(), current_gp.X.copy()
    kernel_init = {gp: gp.kernel.get_params() for gp in gp_list}

    if len(gp_list) > 1:
        for kernel_params in kernel_init.values():
            for key in kernel_params.keys():
                if key not in slider_bounds:
                    raise ValueError(f"Missing slider bounds and steps for parameter: {key}")

    shared_X = np.array(gp_list[0].X)
    shared_Y = np.array(gp_list[0].Y)

    def plot(ax, gp):
        ax.clear()
        ax.grid(True)
        plt.subplots_adjust(left=0.1, bottom=0.35)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        Y_pred, Y_var = gp.predict(X_pred)
        ax.plot(X_pred, Y_pred, "r-", label="GP Prediction")
        ax.fill_between(
            X_pred.flatten(),
            Y_pred - 1 * np.sqrt(Y_var),
            Y_pred + 1 * np.sqrt(Y_var),
            alpha=0.3,
            color=deep_blue,
            label=r"$1\sigma$",
        )
        ax.fill_between(
            X_pred.flatten(),
            Y_pred - 2 * np.sqrt(Y_var),
            Y_pred + 2 * np.sqrt(Y_var),
            alpha=0.2,
            color=deep_blue,
            label=r"$2\sigma$",
        )
        ax.fill_between(
            X_pred.flatten(),
            Y_pred - 3 * np.sqrt(Y_var),
            Y_pred + 3 * np.sqrt(Y_var),
            alpha=0.1,
            color=deep_blue,
            label=r"$3\sigma$",
        )

        _reshape_to = gp.X.shape[0]
        ax.errorbar(
            gp.X.reshape(_reshape_to),
            gp.Y.reshape(_reshape_to),
            yerr=np.abs(gp.Y_err.reshape(_reshape_to)),
            fmt="o",
            color=tomato,
            zorder=10,
        )

        ax.legend(bbox_to_anchor=(0.0, -0.25, 1, 0), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    plot(ax, current_gp)

    def on_click(event):
        global shared_X, shared_Y
        if event.button == 1 and ax.contains(event)[0]:
            x_new = np.array([[event.xdata]])
            y_new = np.array([[event.ydata]])
            shared_X = np.vstack((shared_X, x_new))
            shared_Y = np.vstack((shared_Y, y_new))
            for gp in gp_list:
                gp.update_data(shared_X, shared_Y)
            update_plot()

        elif event.button == 3 and ax.contains(event)[0]:
            if len(shared_X) > 1:  # at least one point
                dists = np.sqrt((shared_X.flatten() - event.xdata) ** 2 + (shared_Y.flatten() - event.ydata) ** 2)
                idx = np.argmin(dists)
                shared_X = np.delete(shared_X, idx, axis=0)
                shared_Y = np.delete(shared_Y, idx, axis=0)
                for gp in gp_list:
                    gp.X = shared_X
                    gp.Y = shared_Y
                update_plot()

    def update_plot():
        plot(ax, current_gp)
        fig.canvas.draw()

    def reset(event):
        global shared_X, shared_Y
        shared_X, shared_Y = X_init.copy(), Y_init.copy()
        for gp in gp_list:
            gp.X, gp.Y = shared_X.copy(), shared_Y.copy()
            gp.kernel.set_params(**kernel_init[gp])
        update_sliders(kernel_init[current_gp])
        update_plot()

    reset_button_ax = plt.axes([0.85, 0.10, 0.1, 0.04])
    reset_button = Button(reset_button_ax, "Reset", color="lightgoldenrodyellow", hovercolor="0.975")
    reset_button.on_clicked(reset)

    fig.canvas.mpl_connect("button_press_event", on_click)

    sliders = {}
    slider_axes = {}

    def update_kernel_param(val, param):
        try:
            current_gp.kernel.set_params(**{param: val})
            update_plot()
        except ValueError as e:
            print(f"Invalid value for {param}: {e}. Re-setting to initial value = {kernel_init[current_gp][param]}.")
            sliders[param].set_val(kernel_init[current_gp][param])

    def create_sliders(kernel_params):
        for ax in slider_axes.values():
            ax.remove()
        slider_axes.clear()
        sliders.clear()

        slider_positions = [(0.2, 0.05, 0.6, 0.03), (0.2, 0.10, 0.6, 0.03), (0.2, 0.15, 0.6, 0.03)]
        for i, (param, value) in enumerate(kernel_params.items()):
            slider_ax = plt.axes(slider_positions[i % len(slider_positions)])
            slider_axes[param] = slider_ax
            sliders[param] = Slider(
                slider_ax,
                param.capitalize(),
                valmin=slider_bounds[param][0],
                valmax=slider_bounds[param][1],
                valinit=value,
                valstep=slider_bounds[param][2],
            )
            sliders[param].on_changed(lambda val, p=param: update_kernel_param(val, p))

    def update_sliders(kernel_params):
        for param, value in kernel_params.items():
            sliders[param].set_val(value)

    create_sliders(current_gp.kernel.get_params())

    if len(gp_list) > 1:
        kernel_names = [str(gp.kernel) for gp in gp_list]
        kernel_selector_ax = plt.axes((0.1, 0.90, 0.8, 0.06))
        kernel_selector = MyRadioButtons(kernel_selector_ax, kernel_names, orientation="horizontal", fontsize=15, size=60)

        def change_kernel(label):
            nonlocal current_gp
            current_gp = gp_list[kernel_names.index(label)]
            current_gp.X, current_gp.Y = shared_X, shared_Y
            create_sliders(current_gp.kernel.get_params())
            update_plot()

        kernel_selector.on_clicked(change_kernel)

    plt.show()
