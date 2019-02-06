import numpy
import pandas

from lookout.style.typos.preprocessing.dev_utils import rand_bool


def pick_subset_of_df(data: pandas.DataFrame, portion: float, weight_column: str = None) -> pandas.DataFrame:
    pick_indices = [rand_bool(portion) for _ in range(len(data))]
    if weight_column in data.columns:
        weights = numpy.array(weight_column)
        average = numpy.sum(weights * 1.0 / len(weights))
        pick_indices = [rand_bool(1.0 * portion * weight / average) for weight in weights]
    return data[pick_indices]


def pick_subset(args):
    data = pandas.read_csv(args.input_file, index_col=0)
    pick_subset_of_df(data, args.picked_portion, args.weight_column).to_csv(args.out_file)

