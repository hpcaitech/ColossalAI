from abc import ABC

from chatgpt.experience_maker import Experience


class Callback(ABC):
    """
        Base callback class. It defines the interface for callbacks.
    """

    def on_fit_start(self) -> None:
        pass

    def on_fit_end(self) -> None:
        pass

    def on_episode_start(self, episode: int) -> None:
        pass

    def on_episode_end(self, episode: int) -> None:
        pass

    def on_make_experience_start(self) -> None:
        pass

    def on_make_experience_end(self, experience: Experience) -> None:
        pass

    def on_learn_epoch_start(self, epoch: int) -> None:
        pass

    def on_learn_epoch_end(self, epoch: int) -> None:
        pass

    def on_learn_batch_start(self) -> None:
        pass

    def on_learn_batch_end(self, metrics: dict, experience: Experience) -> None:
        pass
