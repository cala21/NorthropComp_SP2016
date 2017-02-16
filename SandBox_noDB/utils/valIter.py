class valIter:

    '''
    X is the batch of input images (shape= (batch_size, n_classes, n_rows, n_cols), dtype=float32) 
    Y the batch of target segmentation maps (shape=(batch_size, n_rows, n_cols), dtype=int32) 
    '''
    def __init__(self, data, labels, numBatches):
        self.data = data
        self.labels = labels
        self.curr = 0
        self.batches = numBatches
    
    def __iter__(self):
        return self
    
    def next(self):
        if self.curr < self.batches:
            curr = self.curr
            self.curr += 1
            return self.data[curr], self.labels[curr]
        else:   
            raise StopIteration()

    def get_n_classes(self):
        return 6
    def get_n_samples(self):
        return len(self.data[self.curr])
    def get_n_batches(self):
        return self.batches
    def get_void_labels(self):
        return [5]