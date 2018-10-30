import pandas

from lookout.style.typos.research.typos_functions import rand_bool


def pick_subset(args):
    data = pandas.read_pickle(args.input_file)
    pick_indices = [rand_bool(args.picked_portion) for _ in range(len(data))]
    data = data[pick_indices]
    data.to_pickle(args.out_file)
