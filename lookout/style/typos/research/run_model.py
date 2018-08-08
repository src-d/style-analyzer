from baseline import Baseline
from neighbors_model import NeighborsSuggestionModel
from vectors_model import VectorsDependentModel

def test_model(args):
    if args.pretrained_file is not None:
        with open(args.pretrained_file, "rb") as f:
            model = pickle.load(f)
        if args.emb_file is not None:
            model.fasttext = fastText.load_model(args.emb_file)
    else:
        model = args.model(args.frequencies_file, 
                              args.tokens_file,
                              args.emb_file)

    if args.train_file is not None:
        model.fit(args.train_file, args.cand_train_file)

    if args.dump_file is not None:
        model.dump(args.dump_file)

    if args.test_file is not None:
        suggestions = model.suggest(args.test_file, args.cand_test_file)
        with open(args.out_file, "w") as out_file:
            typos_functions.print_scores(pandas.read_pickle(args.test_file),
                                         suggestions, out_file)
    return model