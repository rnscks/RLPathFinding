class DQNHyperparameters:
    def __init__(self, learning_rate = 1e-4, netWork_updating_rate = 0.005,discount_factor = 0.99, batch_size = 128, epsilon_start = 0.9, eplison_decay = 1000, epsilon_end = 0.05):
        self.learning_rate = learning_rate
        self.netWork_updating_rate = netWork_updating_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.eplison_decay = eplison_decay
        self.epsilon_end = epsilon_end
        