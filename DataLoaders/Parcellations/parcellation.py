# =====================================================================================
# Base class for parcellation data
#
# =====================================================================================
class Parcellation:
    def __init__(self):
        print("Initializing Parcellation")

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

    def get_IDs(self):
        raise NotImplemented('Should have been implemented by subclass!')

    def get_RSN(self, useLR=False):
        raise NotImplemented('Should have been implemented by subclass!')

    def get_atlas_MNI(self):
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
            return self.get_atlas_MNI()
        elif attribute == 'IDs':
            return self.get_IDs()
        else:
            return None  # if the attribute is not one of the ones defined above

    def get_label_ID(self, label=None, ID=None):
        if label is None and ID is None:
            raise Exception('Label or ID must be specified!')
        else:
            ids = self.get_IDs()
            ls = self.get_region_labels()
            if label is not None:
                return ids[ls.index(label)]
            else:
                return ls[ids.index(ID)]

# =====================================================================================
# =====================================================================================
# =====================================================================================EOF