class Task:

    def __init__(self, func, minibatch_id=-1, args=[]):
        self.minibatch_id = minibatch_id
        self.func = func
        self.args = args

    def execute(self):
        return self.func(self.minibatch_id, self.args)
