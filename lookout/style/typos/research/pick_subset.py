import numpy
import pandas

from lookout.style.typos.research.dev_utils import rand_bool


def pick_subset(args):
    data = pandas.read_pickle(args.input_file)
    pick_indices = [rand_bool(args.picked_portion) for _ in range(len(data))]
    if args.weight_column in data.columns:
        weights = numpy.array(args.weight_column)
        average = numpy.sum(weights * 1.0 / len(occs))
        pick_indices = [
            rand_bool(1.0 * args.picked_portion * weight / average) for weight in weights]
    data = data[pick_indices]
    data.to_pickle(args.out_file)
