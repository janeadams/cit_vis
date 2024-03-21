# Conditional Inference Tree Dashboard

This dashboard is for visualizing the results of conditional inference trees.

## Setup

1. Clone this repository:

   ```
   git clone XXX
   cd assay
   ```

2. Install virtual env, then create and activate a virtual environment (optional but recommended):

    ```
    pip install virtualenv
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install requirements with `pip3 install -r requirements.txt` (or just `pip install -r requirements.txt` if pip is aliased to pip3).

## Demo

There is a Jupyter notebook provided to walk through the new dashboard creation. There is also a new notebook for the original dashboard, which included a sankey plot from the tree splits. An explanation of the data structure is:

![](figs/graphic_ofa_distance-01.png)

## Demo

The new dashboard takes the data from this figure:

![](figs/original_tree.png)

Uses the encoding sketched out here:

![](figs/example.png)

And generates these graphs:

![](figs/matrix.png)

## Dashboard

The old dashboard code is in `_archive` along with earlier code for processing trees and performing analysis. These are less ideal than the new methods in `demo.ipynb`.