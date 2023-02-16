class Task:

    def __init__(self, func, *args, **kwargs):
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
        return self.func(*self.args, **self.kwargs)
