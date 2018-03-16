from grammar_object import Grammar
from parse_tools import *
from stat_tools import *
from sklearn_batch import *

def rank(examples, phrase_cache=None, cache_write_dest=None):
    grammar = Grammar(phrase_cache)

    # Grammar transformation functions
    def replace_ats_with_John(string):
        return replace_ats(string, "John")
    args_list = Grammar.cartesian_product(list(range(1, 15)), [True, False], [None, remove_ats, remove_hts, replace_ats_with_John, [remove_hts, remove_ats], [remove_hts, replace_ats_with_John], [remove_ats, replace_ats_with_John], [remove_hts, remove_ats, replace_ats_with_John]])
    func_list = [grammar.get_avg_error_func(*args) for args in args_list]

    grammared = grammar.label_apply(examples, func_list)

    if cache_write_dest is not None:
        grammar.write_phrase_cache(cache_write_dest)

    skl_b = SklearnBatch.load_from_folder('GrammarsAll')
    skld = skl_b.predict(grammared, data_column_name='data')

    del skld['data']
    summed = skld.sum(axis=1)
    totals = pd.DataFrame({
        'tweet': examples,
        'score': summed
    })

    return list(totals.sort_values('score', ascending=False)['tweet'])
