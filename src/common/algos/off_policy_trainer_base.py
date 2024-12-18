
@dataclass
class OffPolicyBase(ABC):
    """
    Base class for off-policy trainers
    """
    # from env
    device: str = "cpu" # 학습 장치 
    state_dim: int = 0 # 상태 차원
    action_dim: int = 0 # 액션 차원
    action_type: str = "" # 액션 타입

    env: gym.Env | str = None # 환경
    agent: BaseAgent = None # agent

    @abstractmethod
    def build(self):
        """
        build trainer

        agent 초기화
        """
        pass

    @abstractmethod
    def environment(self, env: str):
        """
        set environment
        """
        pass