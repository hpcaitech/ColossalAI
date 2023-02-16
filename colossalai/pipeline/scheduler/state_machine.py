class WorkerState:

    def __init__(self, name, transitions, entry_action=None, exit_action=None):
        self.name = name
        self.transitions = transitions
        self.entry_action = entry_action or (lambda: None)
        self.exit_action = exit_action or (lambda: None)

    def get_next_state(self, event):
        return self.transitions.get(event)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, WorkerState):
            return __o.name == self.name

        return False


class StateMachine:

    def __init__(self):
        self.current_state: WorkerState = None
        self.states = {}

    def add_state(self, state: WorkerState):
        self.states[state.name] = state

    def get_state(self, state_name):
        return self.states.get(state_name)

    def set_initial_state(self, state_name):
        self.current_state = self.states[state_name]

    def get_next_state(self, event):
        return self.current_state.get_next_state(event)

    def get_current_state(self):
        return self.current_state.name

    def set_current_state(self, state_name):
        self.current_state = self.get_state(state_name)

    def run(self):
        pass
