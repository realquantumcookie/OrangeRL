from .agent import Tensor_Or_TensorDict
from .agent_util import NNAgentCritic, NNAgentInputMapper, NNAgentCriticMapper, BatchedNNCriticOutput, NNAgentNetworkOutput
from orangerl import AgentStage
import torch
import torch.nn as nn
from typing import Any, Optional, Type, Dict, List

class NNAgentCriticEnsembleImpl(NNAgentCritic):
    def __init__(
        self,
        critic_input_mapper: NNAgentInputMapper,
        critic_networks : List[nn.Module],
        is_sequence_model : bool,
        empty_state : Optional[Tensor_Or_TensorDict],
        num_subsample : int = -1,
        subsample_aggregate_method : str = "min", # "min", "max", "mean"
        is_discrete : bool = False,
    ):
        NNAgentCritic.__init__(self)
        self.critic_input_mapper = critic_input_mapper
        self._critic_networks = critic_networks
        self._is_sequence_model = is_sequence_model
        self.empty_state = empty_state
        self._is_discrete = is_discrete
        self._critic_params, self._critic_buffers = torch.func.stack_module_state(
            critic_networks
        )
        self._vmap_func = torch.func.vmap(
            self._wrapper_call,
            (0, 0, None, None)
        )
        self.num_subsample = num_subsample
        self.subsample_aggregate_method = subsample_aggregate_method

    def _wrapper_call(self, params, buffers, args, kwargs):
        return torch.func.functional_call(self._critic_networks[0], (params, buffers), *args, **kwargs)

    @property
    def is_discrete(self) -> bool:
        return self._is_discrete

    @property
    def is_sequence_model(self) -> bool:
        return self._is_sequence_model

    def forward(
        self,
        obs_batch: Tensor_Or_TensorDict,
        act_batch: Optional[Tensor_Or_TensorDict],
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_update = False,
        stage : AgentStage = AgentStage.ONLINE,
        **kwargs: Any,
    ) -> BatchedNNCriticOutput:
        nn_input_args, nn_input_kwargs = self.critic_input_mapper.forward(
            obs_batch,
            act_batch,
            masks,
            state,
            is_seq=self.is_sequence_model,
            is_update=is_update,
            stage=stage
        )
        ensemble_idxes = torch.randperm(len(self._critic_networks))[:self.num_subsample] if self.num_subsample > 0 else torch.arange(len(self._critic_networks))
        ensemble_len = len(ensemble_idxes)

        network_raw_outputs = self._vmap_func(
            self._critic_params[ensemble_idxes], 
            self._critic_buffers[ensemble_idxes], 
            nn_input_args,
            nn_input_kwargs
        )
        network_outputs = [
            self.critic_input_mapper.map_net_output(
                network_raw_outputs[i],
                masks=masks,
                state=state,
                is_seq=self.is_sequence_model,
                is_update=is_update,
                stage=stage
            )
            for i in range(ensemble_len)
        ]
        network_output = torch.stack([
            network_outputs[i].output
            for i in range(ensemble_len)
        ], dim=0) # (ensemble_len, B, [S])

        if self.is_sequence_model:
            network_output = network_output.flatten(start_dim=2) # (ensemble_len, B, S)
        else:
            network_output = network_output.flatten(start_dim=1) # (ensemble_len, B)

        if self.subsample_aggregate_method == "min":
            mapped_output = network_output.min(dim=0)
        elif self.subsample_aggregate_method == "max":
            mapped_output = network_output.max(dim=0)
        elif self.subsample_aggregate_method == "mean":
            mapped_output = network_output.mean(dim=0)
        else:
            raise NotImplementedError(f"subsample_aggregate_method {self.subsample_aggregate_method} not implemented")
        
        mapped_std = network_output.std(dim=0)

        return BatchedNNCriticOutput(
            critic_estimates=mapped_output,
            distributions=None,
            log_stds=torch.log(mapped_std),
            final_states=None,
            masks=masks,
            is_seq=self.is_sequence_model,
            is_discrete=self.is_discrete,
        )

    def forward_all(
        self,
        obs_batch: Tensor_Or_TensorDict,
        act_batch: Optional[Tensor_Or_TensorDict],
        masks: Optional[torch.Tensor] = None,
        state: Optional[Tensor_Or_TensorDict] = None,
        is_update = False,
        stage : AgentStage = AgentStage.ONLINE,
        **kwargs: Any,
    ) -> List[BatchedNNCriticOutput]:
        ensemble_len = len(self._critic_networks)
        nn_input_args, nn_input_kwargs = self.critic_input_mapper.forward(
            obs_batch,
            act_batch,
            masks,
            state,
            is_seq=self.is_sequence_model,
            is_update=is_update,
            stage=stage
        )
        network_raw_outputs = self._vmap_func(
            self._critic_params, 
            self._critic_buffers, 
            nn_input_args,
            nn_input_kwargs
        )
        network_outputs = [
            self.critic_input_mapper.map_net_output(
                network_raw_outputs[i],
                masks=masks,
                state=state,
                is_seq=self.is_sequence_model,
                is_update=is_update,
                stage=stage
            )
            for i in range(ensemble_len)
        ]
        network_output = torch.stack([
            network_outputs[i].output
            for i in range(ensemble_len)
        ], dim=0) # (ensemble_len, B, [S])

        return [
            BatchedNNCriticOutput(
                critic_estimates=network_output[i],
                distributions=None,
                log_stds=None,
                final_states=None,
                masks=masks,
                is_seq=self.is_sequence_model,
                is_discrete=self.is_discrete
            )
            for i in range(ensemble_len)
        ]