class valIter:

    '''
    X is the batch of input images (shape= (batch_size, n_classes, n_rows, n_cols), dtype=float32) 
    Y the batch of target segmentation maps (shape=(batch_size, n_rows, n_cols), dtype=int32) 
    '''
    def __init__(self, data, labels, numBatches):
        self.data = data
        self.labels = labels
        self.curr = 0
        self.batches = numBatches-1
        self.shape = (numBatches,len(data[0]), len(data[0][0]))
    
    def __iter__(self):
        return self
    
    def next(self):
        if self.curr < self.batches:
            curr = self.curr
            self.curr += 1
            return self.data[curr], self.labels[curr]
        return None
    def get_n_classes(self):
        return 6
    def get_n_samples(self):
        return sum([len(i) for i in self.data])
    def get_n_batches(self):
        return self.batches
    def get_void_labels(self):
        return [255]