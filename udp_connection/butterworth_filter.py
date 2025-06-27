import numpy as np
from scipy.signal import butter, lfilter

class ButterworthFilter:
    def __init__(self, order=2, cutoff=0.15):
        self.order = order
        self.cutoff = cutoff
        self.b, self.a = butter(self.order, self.cutoff, btype='low', analog=False)
        self.zi = {}

    def filter(self, key, data):
        data = np.asarray(data)
        if key not in self.zi:
            self.zi[key] = [np.zeros((self.a.size - 1,)) for _ in range(data.shape[0])]
        filtered = []
        for i, val in enumerate(data):
            val = np.array([val])
            filtered_val, self.zi[key][i] = lfilter(self.b, self.a, val, zi=self.zi[key][i])
            filtered.append(filtered_val[0])
        return np.array(filtered)
