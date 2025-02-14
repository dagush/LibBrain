# =====================================================================================
# Base class for parcellation data
#
# =====================================================================================
class Parcellation:
    def get_coords(self):
        raise NotImplemented('Should have been implemented by subclass!')

    def get_region_labels(self):
        raise NotImplemented('Should have been implemented by subclass!')

    def get_region_short_labels(self):
        raise NotImplemented('Should have been implemented by subclass!')

    def get_cortices(self):
        raise NotImplemented('Should have been implemented by subclass!')

    def get_lobes(self):
        raise NotImplemented('Should have been implemented by subclass!')

    def get_RSN(self, useLR=False):
        raise NotImplemented('Should have been implemented by subclass!')

    def get_atlas(self):
        raise NotImplemented('Should have been implemented by subclass!')

    def get_data(self, attribute):
        if attribute == 'coords':
            return self.get_coords()
        elif attribute == 'labels':
            return self.get_region_labels()
        elif attribute == 'short_labels':
            return self.get_region_short_labels()
        elif attribute == 'cortices':
            return self.get_cortices()
        elif attribute == 'RSN':
            return self.get_RSN()
        elif attribute == 'atlas':
            return self.get_atlas()
        else:
            return None  # if the attribute is not one of the ones defined above
