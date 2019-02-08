import numpy
import pandas


def pick_subset_of_df(data: pandas.DataFrame, size: int = None, portion: float = None,
                      weight_column: str = None) -> pandas.DataFrame:
    if size is None:
        if portion is None:
            raise ValueError("Either size or portion should be specified.")
        size = int(portion * len(data))

    if weight_column in data.columns:
        probs = numpy.array(data[weight_column]) / sum(data[weight_column])
    else:
        probs = None

    result = data.loc[numpy.random.choice(data.index, size, p=probs)]
    result["id"] = result.index
    result.index = range(len(result))
    return result


def pick_subset(args):
    data = pandas.read_csv(args.input_file, index_col=0)
    pick_subset_of_df(data, args.picked_portion, args.weight_column).to_csv(args.out_file)
