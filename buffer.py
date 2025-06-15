from collections import deque

class CandleBuffer:
    def __init__(self, maxlen=12):
        self.buffer = deque(maxlen=maxlen)

    def add(self, candle):
        self.buffer.append(candle)

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen

    def get_features(self):
        return [c.close for c in self.buffer]
