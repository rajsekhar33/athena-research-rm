"""
AthenaDataSet class for handling multiple Athenak simulation data files.
"""

class AthenaDataSet:
    """
    Class for handling multiple Athena++ simulation data files.
    """
    
    def __init__(self, version='1.0'):
        """
        Initialize an AthenaDataSet object.
        
        Parameters
        ----------
        version : str, optional
            Data format version
        """
        self.version = version
        self.ns = []
        self.ads = {}
        
    def load(self, ns, basename, path='.', dtype='athdf', info=False, **kwargs):
        """
        Load multiple data files.

        Parameters
        ----------
        ns : list of int
            File numbers to load
        basename : str
            Output basename, matching AthenaK's <basename>.<00000>.<dtype> naming
            (e.g. 'TRML' for 'TRML.00010.athdf')
        path : str, optional
            Directory containing the data files. Defaults to the current directory.
        dtype : str, optional
            File extension (without the leading dot). Defaults to 'athdf'.
        info : bool, optional
            Whether to print info
        **kwargs : dict, optional
            Additional arguments passed through to AthenaData.load()

        Returns
        -------
        self : AthenaDataSet
            The loaded dataset
        """
        from .athena_data import AthenaData

        for n in ns:
            if n not in self.ns:
                self.ns.append(n)
                filename = f"{path}/{basename}.{n:05d}.{dtype}"
                if info:
                    print(f"Loading {filename}")
                self.ads[n] = AthenaData(n).load(filename, **kwargs)

        return self
    
    def __call__(self, n):
        """
        Get a specific data file.
        
        Parameters
        ----------
        n : int
            File number
            
        Returns
        -------
        AthenaData
            The data object
        """
        return self.ads[n]