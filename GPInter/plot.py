import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from gp import GaussianProcess


def plot_gp(gp: GaussianProcess, X_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)
    plt.subplots_adjust(left=0.1, bottom=0.35)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    Y_init, X_init, length_scale_init, amplitude_init = gp.Y, gp.X, gp.length_scale, gp.amplitude
    Y_pred, Y_var = gp.predict(X_pred)
    ax.plot(X_pred, Y_pred, "b-", label="GP Prediction")
    ax.fill_between(X_pred.flatten(), Y_pred - 1 * np.sqrt(Y_var), Y_pred + 1 * np.sqrt(Y_var), alpha=0.7, color="lightblue", label=r"$1\sigma$")
    ax.fill_between(X_pred.flatten(), Y_pred - 2 * np.sqrt(Y_var), Y_pred + 2 * np.sqrt(Y_var), alpha=0.5, color="lightblue", label=r"$2\sigma$")
    ax.fill_between(X_pred.flatten(), Y_pred - 3 * np.sqrt(Y_var), Y_pred + 3 * np.sqrt(Y_var), alpha=0.3, color="lightblue", label=r"$3\sigma$")

    ax.scatter(gp.X, gp.Y, c="g", s=50, zorder=10, edgecolors=(0, 0, 0))

    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    def on_click(event):
        if event.button == 1 and ax.contains(event)[0]:
            x_new = np.array([[event.xdata]])
            y_new = np.array([[event.ydata]])
            gp.update_data(x_new, y_new)
            update_plot()

        elif event.button == 3 and ax.contains(event)[0]:
            if len(gp.X) > 1:  # at least one point
                dists = np.sqrt((gp.X.flatten() - event.xdata)**2 + (gp.Y.flatten() - event.ydata)**2)
                idx = np.argmin(dists)
                gp.X = np.delete(gp.X, idx, axis=0)
                gp.Y = np.delete(gp.Y, idx, axis=0)
                update_plot()

    def update_plot():
        ax.clear()
        ax.grid(True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        Y_pred, Y_var = gp.predict(X_pred)
        ax.plot(X_pred, Y_pred, "b-", label="GP Prediction")
        ax.fill_between(X_pred.flatten(), Y_pred - 1 * np.sqrt(Y_var), Y_pred + 1 * np.sqrt(Y_var), alpha=0.7, color="lightblue", label=r"$1\sigma$")
        ax.fill_between(X_pred.flatten(), Y_pred - 2 * np.sqrt(Y_var), Y_pred + 2 * np.sqrt(Y_var), alpha=0.5, color="lightblue", label=r"$2\sigma$")
        ax.fill_between(X_pred.flatten(), Y_pred - 3 * np.sqrt(Y_var), Y_pred + 3 * np.sqrt(Y_var), alpha=0.3, color="lightblue", label=r"$3\sigma$")
        ax.scatter(gp.X, gp.Y, c="g", s=50, zorder=10, edgecolors=(0, 0, 0))
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
        fig.canvas.draw()

    def reset(event):
        gp.Y, gp.X = Y_init, X_init
        slider_length_scale.set_val(length_scale_init)
        slider_amplitude.set_val(amplitude_init)

        update_plot()

    reset_button_ax = plt.axes([0.81, 0.2, 0.1, 0.04])
    reset_button = Button(reset_button_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
    reset_button.on_clicked(reset)

    fig.canvas.mpl_connect("button_press_event", on_click)

    axcolor = "lightgoldenrodyellow"
    ax_length_scale = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor=axcolor)
    ax_amplitude = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor=axcolor)

    slider_length_scale = Slider(ax_length_scale, "Length Scale", 0.1, 10.0, valinit=length_scale_init, valstep=0.1)
    slider_amplitude = Slider(ax_amplitude, "Amplitude", 0.1, 10, valinit=amplitude_init, valstep=0.1)

    def update(val):
        gp.length_scale = slider_length_scale.val
        gp.amplitude = slider_amplitude.val
        
        update_plot()

    slider_length_scale.on_changed(update)
    slider_amplitude.on_changed(update)

    plt.show()



def plot_gp_notebook(gp: GaussianProcess, X_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    Y_init, X_init, length_scale_init, amplitude_init = gp.Y, gp.X, gp.length_scale, gp.amplitude
    Y_pred, Y_var = gp.predict(X_pred)

    Y_pred, Y_var = gp.predict(X_pred)
    ax.plot(X_pred, Y_pred, "b-", label="GP Prediction")
    ax.fill_between(X_pred.flatten(), Y_pred - 1 * np.sqrt(Y_var), Y_pred + 1 * np.sqrt(Y_var), alpha=0.7, color="lightblue", label=r"$1\sigma$")
    ax.fill_between(X_pred.flatten(), Y_pred - 2 * np.sqrt(Y_var), Y_pred + 2 * np.sqrt(Y_var), alpha=0.5, color="lightblue", label=r"$2\sigma$")
    ax.fill_between(X_pred.flatten(), Y_pred - 3 * np.sqrt(Y_var), Y_pred + 3 * np.sqrt(Y_var), alpha=0.3, color="lightblue", label=r"$3\sigma$")
    ax.scatter(gp.X, gp.Y, c="g", s=50, zorder=10, edgecolors=(0, 0, 0))

    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)

    def on_click(event):
        if event.button == 1 and ax.contains(event)[0]:
            x_new = np.array([[event.xdata]])
            y_new = np.array([[event.ydata]])
            gp.update_data(x_new, y_new)
            update_plot()

        elif event.button == 3 and ax.contains(event)[0]:
            if len(gp.X) > 1:  # at least one point
                dists = np.sqrt((gp.X.flatten() - event.xdata)**2 + (gp.Y.flatten() - event.ydata)**2)
                idx = np.argmin(dists)
                gp.X = np.delete(gp.X, idx, axis=0)
                gp.Y = np.delete(gp.Y, idx, axis=0)
                update_plot()

    def reset(event):
        gp.Y, gp.X = Y_init, X_init

        length_scale_slider.value = length_scale_init
        amplitude_slider.value = amplitude_init

        update_plot()

    def update_plot():
        ax.clear()
        ax.grid(True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        Y_pred, Y_var = gp.predict(X_pred)
        ax.plot(X_pred, Y_pred, "b-", label="GP Prediction")
        ax.fill_between(
            X_pred.flatten(), Y_pred - 1 * np.sqrt(Y_var), Y_pred + 1 * np.sqrt(Y_var), alpha=0.7, color="lightblue", label=r"$1\sigma$"
        )
        ax.fill_between(
            X_pred.flatten(), Y_pred - 2 * np.sqrt(Y_var), Y_pred + 2 * np.sqrt(Y_var), alpha=0.5, color="lightblue", label=r"$2\sigma$"
        )
        ax.fill_between(
            X_pred.flatten(), Y_pred - 3 * np.sqrt(Y_var), Y_pred + 3 * np.sqrt(Y_var), alpha=0.3, color="lightblue", label=r"$3\sigma$"
        )
        ax.scatter(gp.X, gp.Y, c="g", s=50, zorder=10, edgecolors=(0, 0, 0))
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
        fig.canvas.draw()


    length_scale_slider = widgets.FloatSlider(value=gp.length_scale, min=0.1, max=10.0, step=0.1, description='Length Scale')
    amplitude_slider = widgets.FloatSlider(value=gp.amplitude, min=0.1, max=10.0, step=0.1, description='Amplitude')
    reset_button = widgets.Button(description='Reset')
    reset_button.on_click(reset)

    fig.canvas.mpl_connect("button_press_event", on_click)
    
    def update(change):
        gp.length_scale = length_scale_slider.value
        gp.amplitude = amplitude_slider.value
        update_plot()

    length_scale_slider.observe(update, names='value')
    amplitude_slider.observe(update, names='value')

    display(length_scale_slider, amplitude_slider, reset_button)
    plt.show()