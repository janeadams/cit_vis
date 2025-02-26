# Conditional Inference Tree Dashboard

This dashboard is for visualizing the results of XAI trees on wide data. It was originally designed for another data structure; the example here uses the results of a SciKit Learn Decision Tree Classifier for demonstration purposes.

## Setup

1. Clone this repository:

   ```
   git clone XXX
   cd XXX
   ```

2. If you don't already have `uv` installed, install it with one of the following commands:

On MacOS with Homebrew:
```bash
brew install uv
```

On Windows:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

On Linux:
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

or choose one of the other uv install methods detailed [here](https://docs.astral.sh/uv/getting-started/installation/).

3. Run `uv sync` to sync dependencies for the project. This process replaces using anaconda or pip environments. You can read more about uv in the "Dependencies" section below.

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

### Dependencies
> [!WARNING]  
> We are using uv. DO NOT use `pip` directly or create a requirements.txt file. For example, instead of `pip install tqdm`, you should use `uv add tqdm`, or prepend all pip commands, e.g. `uv pip install tqdm`.

This project uses `uv` for dependency management. [Read more about `uv` here.](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) To set up your environment on first clone, run `uv sync`. Learn more about uv project structure [here](https://docs.astral.sh/uv/guides/projects/#project-structure).

To activate the virtual environment after it is created, you can call `source .venv/bin/activate` on Mac/Linux or `.venv\Scripts\activate` on Windows. You might want to just add an alias to your bash config, like `vim .zshrc` > `alias activate="source .venv/bin/activate"` so you can just type `activate` to activate.

To use the uv-created .venv in a Jupyter Notebook in VSCode, choose "Select kernel..." > "Python Environments" > "Create Python Environment" > ".venv" > "Use Existing".

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
