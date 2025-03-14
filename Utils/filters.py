from neuronumba.tools import filters


class BandPassFilter(filters.BandPassFilter):
    """
    Convenience class to avoid the .T).T requirement, as neuronumba filters work in a
    (time, RoI) fashion while the rest of the library work with (RoI, time)...
    """
    def filter(self, signal):
        return super().filter(signal.T).T

