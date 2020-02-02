# For exporting and importing binary files between python enviroments.
import pickle


class OFHandlers:


    def __init__(self):
        pass

    @staticmethod
    def save_object(path,object):
        """Saves a python object to path (in filesytem).

        Parameters:
            object: python object. 
            path: path in filesystem.
        """
        with open(path,"wb") as f:
            pickle.dump(object,f,pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_object(path):
        """loads a python object from path (in filesytem).

        Parameters:
            path: path in filesystem where the python object file is.

        Returns:
            object: python object to be used in current python enviroment.
        """
        with open(path,"rb") as f:
            object = pickle.load(f) 
        return object