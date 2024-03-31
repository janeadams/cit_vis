# Conditional Inference Tree Dashboard

This dashboard is for visualizing the results of XAI trees on wide data. It was originally designed for another data structure; the example here uses the results of a SciKit Learn Decision Tree Classifier for demonstration purposes.

## Setup

1. Clone this repository:

   ```
   git clone XXX
   cd XXX
   ```

2. Install virtual env, then create and activate a virtual environment (optional but recommended):

    ```
    pip install virtualenv
    python3 -m venv .venv
    source .venv/bin/activate
    ```

    If you are on another operating system, refer to this table for the correct activation command:

    | Platform |    Shell   | Command to activate virtual environment |
    |:--------:|:----------:|:---------------------------------------:|
    | POSIX    | bash/zsh   | $ source .venv/bin/activate            |
    |          | fish       | $ source .venv/bin/activate.fish       |
    |          | csh/tcsh   | $ source .venv/bin/activate.csh        |
    |          | PowerShell | $ .venv/bin/Activate.ps1               |
    | Windows  | cmd.exe    | C:\> .venv\Scripts\activate.bat        |
    |          | PowerShell | PS C:\> .venv\Scripts\Activate.ps1     |

3. Install requirements with `pip3 install -r requirements.txt` (or just `pip install -r requirements.txt` if pip is aliased to pip3). There is also `setup-requirements.txt` included, but everything there is also in the main requirements file, so you shouldn't need to install both.

## Demo

Both `main.ipynb` and `main.py` are provided to run the demo; choose whichever you prefer. The workflow is as follows:

- `make_data()` generates the synthetic data; by default, this is stored in the `data` directory. You can change the data storage directory in the sample `.env` file provided. This is also where you will find parameters for the fake data generation such as the number of mice, microbes. and traits.
- `parse_data()` parses all the data in the data directory (also taken from the `.env` file) and outputs additional files such as `relevance.json` in `data` summarizing all the relevant traits for each microbe, and the `grid.pkl` files for making the Gridded Trees data structure for the visualization.
- `make_dashboard()` is the main visualization script, which will generate the Plotly Dash dashboard. By default, this runs on port `:8050`; this can also be changed in the `.env` file under `PORT`.

## Usage

Run the program from the command line by running:

```bash
python3 main.py
```

or in a Jupyter Notebook by opening `main.ipynb`.

The visualization application should launch by default at `localhost:8050`, which means you can open any browser window and navigate to [http://127.0.0.1:8050](http://127.0.0.1:8050) to view.

If you get an error that the port is already in use (this can happen if you're launching the app multiple times, or trying to run the app from a notebook and the CLI at the same time), you can `lsof -i:8050` to find the PID and kill it if you need. If something besides this app is running at `:8050` already, change the default port in the `.env`.

You can opt to select a microbe of interest from the dropdown on the left, which will filter down the dropdown on the right to relevant traits; this is optional, and if left unselected, all traits will be available. Select a trait from the dropdown to render the charts for that trait.

The system is designed to output all the data structures necessary for the rendering, therefore making the transformations accessible for downstream tasks; for our collaborators' uses, saving these out to the data directory structure is a sufficient solution, but having a "download data" button on each chart would be a nice addition in the future for GUI-only users.

## Future Work

Some known things we would like to improve upon:

- Adjust strip plot jitter: this is a trace patch omission in the Plotly px.strip config for accessing the Box plot constructor. We have filed a pull request to make this work.
- Adding unit tests
- Keeping ordering of groups consistent across all three charts; these get shuffled at some point
- Algorithmic optimization (and a loading screen for when changes are slow)
- Adding command line arguments to match the `.env` variables options
- Data egress: this, and some other features, are in the private repo, but don't work on the synthetic data yet

## Contribution and Collaboration

This is a research prototype, not a system, so it is not yet robust to general deployment. If you have trouble, please file an issue on GitHub. If you'd like to chat about how to apply this to your own data on a real system, e.g. GCP or your own HPC cluster, by extending the existing functionality, please email XXX
