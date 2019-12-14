from ml.flmodel import Metric


class Comparison:
    def __init__(self, metric: Metric, threshold: float):
        self.metric = metric
        self.threshold = threshold

    def satisfied(self, prev, new):
        if self.metric is Metric.LOSS:
            prev, new = prev[0], new[0]
            if prev <= self.threshold:
                return prev >= new
            return prev >= new + self.threshold
        elif self.metric is Metric.ACC:
            prev, new = prev[1], new[1]
            if new + self.threshold > 1.0:
                return prev <= new
            return prev <= new + self.threshold
        else:
            raise KeyError
