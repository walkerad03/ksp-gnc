from __future__ import annotations


class State[C]:
    def on_enter(self, _context: C) -> None:
        pass

    def on_exit(self, _context: C) -> None:
        pass

    def tick(self, _context: C) -> type[State[C]] | None:
        return None


class StateMachine[C]:
    def __init__(self, initial_state_cls: type[State[C]], context: C) -> None:
        self.context = context
        self.current_state: State[C] = initial_state_cls()

        self._history: list[tuple[str | None, str]] = [
            (None, type(self.current_state).__name__)
        ]
        self.current_state.on_enter(self.context)

    @property
    def history(self) -> list[tuple[str | None, str]]:
        return list(self._history)

    def tick(self) -> None:
        next_state_cls = self.current_state.tick(self.context)

        if next_state_cls is not None and next_state_cls is not type(
            self.current_state
        ):
            self.current_state.on_exit(self.context)

            from_name = type(self.current_state).__name__
            self.current_state = next_state_cls()
            to_name = type(self.current_state).__name__

            self._history.append((from_name, to_name))
            self.current_state.on_enter(self.context)
