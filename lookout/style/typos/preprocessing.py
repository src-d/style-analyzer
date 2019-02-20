from typing import Union

import numpy
import pandas

from lookout.style.typos.utils import Columns


def pick_subset_of_df_rows(data: Union[pandas.DataFrame, str], size: int = None,
                           portion: float = None, weight_column: str = None) -> pandas.DataFrame:
    """
    Pick subset of given dataframe's rows. Every row can be picked not more that once.

    :param data: Dataframe or its .csv dump which contains
    :param size: Number of rows to pick.
    :param portion: Portion of `data`'s rows to pick.
    :param weight_column: Column in data to use as weights for picking.
    :return: Dataframe, composed of picked rows from `data`
    """
    if size is None:
        if portion is None:
            raise ValueError("Either size or portion should be specified.")
        size = int(portion * len(data))

    if isinstance(data, str):
        data = pandas.read_csv(data, index_col=0)

    if weight_column in data.columns:
        probs = numpy.array(data[weight_column]) / sum(data[weight_column])
    else:
        probs = None

    result = data.loc[numpy.random.choice(data.index, size, replace=False, p=probs)]
    result[Columns.Id] = result.index
    result.index = range(len(result))
    return result
