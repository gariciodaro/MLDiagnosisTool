# Linear algebra library.
import numpy as np


class DictionaryAuxiliar:
    """Auxiliar operation over dictionaries."""


    def __init__(self):
        pass

    @staticmethod
    def key_with_maxval(d):
        """Get the key with maximum value in a dictionary
        Paramerers:
            d (dictionary)
        """
        v=list(d.values())
        k=list(d.keys())
        return k[v.index(max(v))]

    @staticmethod
    def key_with_minval(d):
        """Get the key with minimum value in a dictionary
        Paramerers:
            d (dictionary)
        """
        v=list(d.values())
        k=list(d.keys())
        return k[v.index(min(v))]

    @staticmethod
    def get_max_mean(d):
        """Get the maximum mean of the first value

        Paramerers:
            d (dictionary)
        """
        print("models",[k for k,v in d.items()])
        print("scores",[np.mean(v[0]) for k,v in d.items()])
        max_mean_cv_validation=max([np.mean(v[0]) for k,
                                    v in d.items()])
        return max_mean_cv_validation


