# MIT License
# Copyright 2023 Ryan Hausen and contributers

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import os
from itertools import count
from typing import Tuple

import ray.util.queue as queue
from tqdm import tqdm


class OutputManager:
    """Manages all FitsMap console output for tasks."""

    SENTINEL = -1

    __instance = None

    @staticmethod
    def pbar_disabled():
        return bool(os.getenv("DISBALE_TQDM", False))

    def check_for_updates(func):
        def f(*args, **kwargs):
            func(*args, **kwargs)
            if OutputManager.__instance and not OutputManager.pbar_disabled():
                OutputManager.__instance.check_for_updates()

        return f

    @staticmethod
    @check_for_updates
    def write(pbar_ref: Tuple[int, queue.Queue], message: str) -> None:
        idx, q = pbar_ref

        def write(pbar):
            pbar.clear()
            pbar.display(message)

        q.put([idx, write])

    @staticmethod
    @check_for_updates
    def update(pbar_ref: Tuple[int, queue.Queue], value: int) -> None:
        idx, q = pbar_ref
        q.put([idx, lambda pbar: pbar.update(value)])

    @staticmethod
    @check_for_updates
    def update_done(pbar_ref: Tuple[int, queue.Queue]) -> None:
        OutputManager.update(pbar_ref, OutputManager.SENTINEL)

    @staticmethod
    @check_for_updates
    def set_description(pbar_ref: Tuple[int, queue.Queue], desc: str) -> None:
        idx, q = pbar_ref

        def write(pbar):
            pbar.clear()
            pbar.set_description(desc)

        q.put([idx, write])

    @staticmethod
    @check_for_updates
    def set_units_total(
        pbar_ref: Tuple[int, queue.Queue], unit: str, total: int
    ) -> None:
        idx, q = pbar_ref

        def setup(pbar):
            pbar.unit = unit
            pbar.reset(total=total)

        q.put([idx, setup])

    def __init__(self):
        self.progress_bars = dict()
        self.in_progress = dict()
        self.q = queue.Queue()
        self.idx = count()
        OutputManager.__instance = self

    def make_bar(self) -> Tuple[int, queue.Queue]:
        for idx in self.idx:
            self.progress_bars[idx] = tqdm(
                position=idx, disable=OutputManager.pbar_disabled(), leave=True
            )
            if not OutputManager.pbar_disabled():
                self.progress_bars[idx].display("Preparing...")
            self.in_progress[idx] = True
            yield tuple([idx, self.q])

    def check_for_updates(self):
        if not self.q.empty():
            idx, f = self.q.get(block=True)
            running = not f == OutputManager.SENTINEL
            self.in_progress[idx] = running

            if running and not OutputManager.pbar_disabled():
                f(self.progress_bars[idx])

    def close_up(self):
        list(
            map(
                lambda key: self.progress_bars[key].close(),
                sorted(self.progress_bars.keys()),
            )
        )

    @property
    def jobs_running(self):
        return any(self.in_progress.values())
