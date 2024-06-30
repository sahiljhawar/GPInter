# GPInter

Gaussian Processes from scratch with interactive visualization.

## Install requirements
```bash
pip install -r requirements.txt
```

## Usage
```bash
cd GPInter
python main.py
```

- `Left click` on the plot to add a new point (new points are added with 0 error)
- `Right click `on the plot to remove a point (deletes the nearest point to cursor)
- Use `sliders` to change length scale and amplitude (+ other kernel parameters if available)
- `Reset` button resets the plot to initial state
- `Kernel` radio button to choose different kernels

## TODO

- [x] Refactor code and implement `Kernel` base class
- [x] Implement more kernels
- [x] Drop down menu to choose different kernels on the fly
- [ ] (Re)Implement `plot_gp` for notebook
- [ ] Implement kernel operations (maybe at the cost of interactivity)
