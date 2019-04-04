# Copyright 2017 Mark van der Wilk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import time


class ElapsedTracker:
    def __init__(self, elapsed=0):
        self._elapsed = elapsed

    def start(self):
        pass

    def stop(self):
        pass

    def add(self, time):
        self._elapsed += time

    @property
    def elapsed(self):
        return self._elapsed


class Stopwatch(ElapsedTracker):
    def __init__(self, elapsed=0.0):
        super().__init__(elapsed)
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            self.stop()
        self._start_time = time.time()
        return self

    def stop(self):
        self._elapsed = self.elapsed
        self._start_time = None

    @property
    def running(self):
        return self._start_time is not None

    @property
    def elapsed(self):
        if self.running:
            return self._elapsed + time.time() - self._start_time
        else:
            return self._elapsed

    @contextlib.contextmanager
    def pause(self):
        self.stop()
        yield
        self.start()
