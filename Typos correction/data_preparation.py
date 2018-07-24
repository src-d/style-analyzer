import pandas
import numpy
import itertools
import sys
import modelforge

path_to_sourced = '/Users/irina/Documents/GitHub/ml'
sys.path.append(path_to_sourced)
import sourced
from sourced.ml.models import id2vec

def flattify(dataframe, column, new_column, apply_function = lambda x:x):
    """
    Flattify dataframe on 'column' with extracted elemenents put to 'new_column'
    """
    other_columns = list(dataframe.columns)
    flat_other = numpy.repeat(dataframe.loc[:, other_columns].values, repeats = numpy.array(dataframe[column].apply(lambda x:len(apply_function(x))).tolist()), axis=0)
    flat_column = list(itertools.chain.from_iterable(dataframe[column].apply(apply_function).tolist()))
    result = pandas.DataFrame(flat_other, columns = other_columns)
    result[new_column] = flat_column
    return result

def split_function(string):
    try:
        return string.split()
    except AttributeError:
        print(string)

def get_info(data, info_file):
    """
    Extract tokens from 'token_split' column and put them to 'identifier' column, dump result
    """
    id_info = flattify(data, 'token_split', 'identifier', apply_function = split_function)
    columns = list(id_info.columns)
    columns.remove('token')
    
    id_info = id_info.loc[:, columns]
    id_info.to_csv(info_file)
    return id_info

def get_stats(id_info, stats_file):
    """
    Leave only stats for identifiers, dump result
    """
    id_stats = id_info.loc[:, ['num_files', 'num_occ', 'num_repos', 'identifier']]
    id_stats = id_stats.groupby(['identifier']).sum()
    id_stats.to_csv(stats_file)
    return id_stats
  
def embedding(model, token):
    return model.embeddings[model['i.' + token]]

if __name__ == "__main__":
    data = pandas.read_csv('repos2ids_v3.4_stats.csv')
    id_info = get_info(data, 'id_info.csv')  
    id_stats = get_stats(id_info, 'id_stats.csv')

    model = id2vec.Id2Vec()
    model.load('emb-18.asdf')

    common_tokens = list(sorted(list(set(id_stats.index.tolist()).intersection(set([x[2:] for x in model.tokens])))))
    vector_matrix = numpy.array([embedding(model, token) for token in common_tokens])

    vector_matrix.dump('vector_matrix')
    common_tokens.dump('common_tokens')