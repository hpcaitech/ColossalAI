class Task:

    def __init__(self, func, minibatch_id=-1, *args, **kwargs):
        self.minibatch_id = minibatch_id
        self.func = func

        if not args:
            self.args = ()
        else:
            self.args = args

        if not kwargs:
            self.kwargs = {}
        else:
            self.kwargs = kwargs

    def execute(self):
        return self.func(self.minibatch_id, *self.args, **self.kwargs)
