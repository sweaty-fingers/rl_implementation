"""
학습 중 기록할 metric 들을 담을 유용한 클래스, 매서드 정의. 
"""
from collections import defaultdict

class MetricsMeter():
    """
    AverageMeter 클래스를 default로 하는 dictionary를 담은 클래스.
    새로운 metric을 추가해도 딕셔너리처럼 새로운 key를 설정하여 추가하면 됨.
    """
    def __init__(self):
        self._metrics = defaultdict(AverageMeter)
    
    def __setitem__(self, key, value):
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
        for key, metric in self._metrics.items():
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
