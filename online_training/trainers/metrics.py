"""
학습 중 기록할 metric 들을 담을 유용한 클래스, 매서드 정의. 
"""
import numpy as np
from collections import defaultdict

POSSIBLE_CRITERION = ["min", "max"]

class MetricsMeter():
    """
    AverageMeter 클래스를 default로 하는 dictionary를 담은 클래스.
    새로운 metric을 추가해도 딕셔너리처럼 새로운 key를 설정하여 추가하면 됨.
    """
    def __init__(self):
        self._metrics = defaultdict(AverageMeter)
    
    def __setitem__(self, key, value):
        if isinstance(value, AverageMeter):
            raise TypeError("value must be instance of AverageMeter")
        self._metrics[key] = value
    
    def __getitem__(self, key):
        return self._metrics[key]

    def items(self):
        for key, value in self._metrics.items():
            yield (key, value)
    
    def avg_items(self):
        for key, value in self._metrics.items():
            yield (key, value.avg)
            
    def __iter__(self):
        return iter(self._metrics)
        
    def reset_all(self):
        for _, metric in self._metrics.items():
            metric.reset()
        

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MetricLogs():
    """
    AverageMeter 클래스를 default로 하는 dictionary를 담은 클래스.
    새로운 metric을 추가해도 딕셔너리처럼 새로운 key를 설정하여 추가하면 됨.
    """
    def __init__(self, rolling_window_size):
        self.rolling_window_size = rolling_window_size
        self._metrics = {}
    
    def __setitem__(self, key, value):
        if isinstance(value, MetricLog):
            raise TypeError("value must be instance of MetricLog")
        self._metrics[key] = value
    
    def __getitem__(self, key):
        return self._metrics[key]
    
    def add_metric_log(self, key, criterion):
        self._metrics[key] = MetricLog(self.rolling_window_size, criterion)

    def items(self):
        for key, value in self._metrics.items():
            yield (key, value)
    
    def avg_items(self):
        for key, value in self._metrics.items():
            yield (key, value.avg)
            
    def __iter__(self):
        return iter(self._metrics)
        
    def reset_all(self):
        for _, metric in self._metrics.items():
            metric.reset()

class MetricLog():
    def __init__(self, rolling_window_size, criterion):
        self.rolling_window_size = rolling_window_size
        self.criterion = criterion
        self.result_log = []
        self.moving_avg_log = []

        if self.criterion not in POSSIBLE_CRITERION:
            raise ValueError(f"criterion should be in {POSSIBLE_CRITERION}\n Now get {self.criterion}")

        if self.criterion == "min":
            self.best_value = np.inf
            self.best_moving_avg = np.inf
        elif self.criterion == "max":
            self.best_value = -np.inf
            self.best_moving_avg = -np.inf

    def reset(self):
        self.result_log = []
        self.moving_avg_log = []
    
    def update(self, val, n_repeat_for_eval: int =None):
        """
        n_repeat: eval step과 training의 steps 수를 맞춰주기 위한 장ㅊ;
        """
        if n_repeat_for_eval is not None:
            self.result_log.append(val)
            self.moving_avg_log.append(np.mean(self.result_log[-self.rolling_window_size]))
            self.update_best_value()
        else:
            self.result_log.append(val)
            self.moving_avg_log.append(np.mean(self.result_log[-self.rolling_window_size]))
            self.update_best_value()

    def update_best_value(self):
        """
        현재까지의 best metric (best) 와 비교 값(compare)을 비교하여 best 값 반환
        criterion: best metric의 기준 (min or max) 
        """
        if self.criterion == "min":
            self.best_value = min(self.best_value, self.result_log[-1])
            self.best_moving_avg = min(self.best_moving_avg, self.moving_avg_log[-1])
        elif self.criterion == "max":
            self.best_value = max(self.best_value, self.result_log[-1])
            self.best_moving_avg = max(self.best_moving_avg, self.moving_avg_log[-1])