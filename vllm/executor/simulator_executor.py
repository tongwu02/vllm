from typing import List, Optional, Set, Tuple

from vllm.config import VllmConfig
from vllm.executor.executor_base import ExecutorBase
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import ExecuteModelRequest
from vllm.sim.simulator import Simulator


class SimulatorExecutor(ExecutorBase):
    """Dummy executor used in simulator mode (no real model execution)."""

    uses_ray: bool = False

    def __init__(self, vllm_config: VllmConfig, simulator: Simulator) -> None:
        self.simulator = simulator
        super().__init__(vllm_config=vllm_config)

    def _init_executor(self) -> None:
        # No-op initialization; no workers to spawn.
        return

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        # Provide a generous default to enable prefix caching logic.
        num_gpu = self.cache_config.num_gpu_blocks or 1024
        num_cpu = self.cache_config.num_cpu_blocks or 0
        return int(num_gpu), int(num_cpu)

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        # Nothing to initialize in simulator mode.
        return

    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> Optional[List[SamplerOutput]]:
        # Execution is handled directly inside LLMEngine for simulator mode.
        return None

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return True

    def remove_lora(self, lora_id: int) -> bool:
        return True

    def pin_lora(self, lora_id: int) -> bool:
        return True

    def list_loras(self) -> Set[int]:
        return set()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        return True

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return True

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return True

    def list_prompt_adapters(self) -> Set[int]:
        return set()

    def check_health(self) -> None:
        return
