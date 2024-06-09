class Agent:
    def get_hyperparams(self):
        pass

    def update_target_model(self):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def act(self, state):
        pass

    def replay(self):
        pass

    def get_weights(self):
        pass

    def load(self, name):
        pass

    def save(self, name):
        pass

    def get_name(self):
        pass
