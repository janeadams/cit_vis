from cit_vis.setup import make_data
from cit_vis.parse import parse_data
from cit_vis.visualization import *

if __name__ == "__main__":
    make_data()
    parse_data()
    make_dashboard()