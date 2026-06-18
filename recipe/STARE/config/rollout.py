# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig
from verl.workers.config import MultiTurnConfig, RolloutConfig
from verl.trainer.config import AlgoConfig
from verl.workers.config import ActorConfig, FSDPActorConfig

__all__ = [
    "Agent_MultiTurnConfig",
    "Agent_RolloutConfig",
    "Agent_AlgoConfig",
    "Agent_FSDPActorConfig",
    "Adaptive_Entropy_Config",
    "STAREConfig",
]


@dataclass
class Agent_MultiTurnConfig(MultiTurnConfig):
    val_max_user_turns: int = 1
    val_max_assistant_turns: int = 1
    format_real: str = "hermes"
    function_name: str = "code_interpreter"




@dataclass
class Agent_RolloutConfig(RolloutConfig):
    stop_tokens: Optional[list[str]] = None
    LRUCache_server_weight_prompt_length: int = 4096
    LRUCache_server_weight_prompt_length_enable: bool = False
    LRUCache_server_weight_score: int = 1
    free_cache_engine_sleep: bool = False
    enable_expert_parallel: bool = False
    val_temperature_lists: Optional[list[float]] = None
    partial_rollout_max_split: int = 1
    tool_calls_no_limit: bool = False

@dataclass
class Agent_AlgoConfig(AlgoConfig):
    partial_rollout_max_split: int = 1
    mask_invalid_json: bool = False
    mask_invalid_answer: bool = False
    break_json_format_error: bool = True
    mask_repeat_response: bool = False
    truncat_repeat_response: bool = False
    mask_positive_samples: bool = False
    json_format_try_n: int = 1


@dataclass
class Adaptive_Entropy_Config(BaseConfig):
    enabled: bool = False
    max_ent_coef: float = 0.5
    min_ent_coef: float = 0
    delta_ent_coef: float = 0.001
    target_entropy: float = 0

@dataclass
class STAREConfig(BaseConfig):
    """Configuration for STARE (Surprisal-guided Token-level Advantage Reweighting).

    Defaults follow the paper's main configuration: Variant O1 (one-sided
    amplification), batch-level closed-loop target-entropy gating, and fixed
    weights with ``W = 1.1``, ``M = 0.9``, ``H_tgt = 0.3``, ``P% = 10%``.
    """

    # Master switch. When False, the stare policy loss reduces to vanilla GRPO.
    enabled: bool = False
    # "O1": one-sided amplification of L+ (default). "C2": also attenuate L-.
    variant: str = "O1"
    # Top-P% high-surprisal selection ratio within each advantage subset.
    top_p_ratio: float = 0.1
    # Reweight factor W (> 1) applied to L+ (positive-advantage high-surprisal tokens).
    reweight_w: float = 1.1
    # Reweight factor M (< 1) applied to L- (negative-advantage high-surprisal, C2 only).
    reweight_m: float = 0.9
    # Closed-loop target entropy H_tgt (Section 4.3 batch-level gate).
    target_entropy: float = 0.3
    # Adaptive weight schedule (Section 4.4). With adaptive=False weights stay fixed.
    adaptive: bool = False
    alpha: float = 0.01
    w_max: float = 1.5
    m_min: float = 0.5

@dataclass
class Agent_FSDPActorConfig(FSDPActorConfig):
    use_teacher_gt_kl_loss: bool = False
    teacher_gt_kl_loss_coef: float = 0
    use_partial_rollout_old_log_probs: bool = False
    adaptive_entropy: Adaptive_Entropy_Config = field(default_factory=Adaptive_Entropy_Config)
    stare: STAREConfig = field(default_factory=STAREConfig)
