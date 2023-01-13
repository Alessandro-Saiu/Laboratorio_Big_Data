from exploration import Evaluator
from random_forest import Preparation, RandomTree
from hyperparameter_tuning import HyperParameters

if __name__ == '__main__':
    # Questa prima sezione serve per fare un analisi preliminare sul dataframe
    colonne = ['turns', 'victory_status', 'winner', 'white_rating', 'black_rating', 'opening_eco']
    categoriche = ['victory_status', 'opening_eco']
    Chess_evaluator = Evaluator('games.csv')
    Chess_evaluator.select_columns(colonne)
    Chess_evaluator.start_expl(categoriche)

    # Questa seconda sezione serve a preparare il dataframe alla modellazione
    Chess_preparation = Preparation('games.csv')
    Chess_preparation.select_columns(colonne)
    indexed_features = Chess_preparation.make_string_indexing(categoriche)
    dummy_features = Chess_preparation.make_one_hot_encoding()
    dataframe, features = Chess_preparation.vectorize(target='winner')

    # Prima d' iniziare la modellazione voglio testare gli hyper-parameters migliori per il modello
    Chess_HyperParameters = HyperParameters(indexed_features)
    Chess_HyperParameters.tuning()

    # Una volta restituiti i parametri migliori possiamo passare alla random forest vera e propria
    # dataframe, features = vector_features
    Chess_rt = RandomTree(dataframe, features)
    Chess_rt.apply_model(target='winner')
