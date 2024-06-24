# GPInter

Gaussian Process from scratch with interactive visualization.

## Install requirements
```bash
pip install -r requirements.txt
```

## Usage
```bash
cd GPInter
python main.py
```

or 

Run the notebook `main.ipynb`

- `Left click` on the plot to add a new point
- `Right click `on the plot to remove a point (deletes the nearest point to cursor)
- Use `sliders` to change length scale and amplitude
- `Reset` button resets the plot to initial state

## TODO

- [ ] Refactor code and implement `Kernel` base class
- [ ] Implement more kernels
- [ ] Drop down menu to choose different kernels on the fly
