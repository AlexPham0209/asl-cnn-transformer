class EarlyStopping:
    def __init__(self, patience=1, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_valid_loss = float("inf")

    def early_stop(self, valid_loss):
        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            self.counter = 0

        elif valid_loss > (self.min_valid_loss + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False
