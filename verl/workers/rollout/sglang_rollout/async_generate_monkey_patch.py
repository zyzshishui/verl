import asyncio
from typing import AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union
from sglang.srt.entrypoints.verl_engine import VerlEngine
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj


async def async_generate(
    self,
    # The input prompt. It can be a single prompt or a batch of prompts.
    prompt: Optional[Union[List[str], str]] = None,
    sampling_params: Optional[Union[List[Dict], Dict]] = None,
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None,
    # The image input. It can be a file name, a url, or base64 encoded string.
    # See also python/sglang/srt/utils.py:load_image.
    image_data: Optional[Union[List[str], str]] = None,
    return_logprob: Optional[Union[List[bool], bool]] = False,
    logprob_start_len: Optional[Union[List[int], int]] = None,
    top_logprobs_num: Optional[Union[List[int], int]] = None,
    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
    lora_path: Optional[List[Optional[str]]] = None,
    custom_logit_processor: Optional[Union[List[str], str]] = None,
) -> Union[Dict, AsyncIterator[Dict]]:
    """
    The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
    Please refer to `GenerateReqInput` for the documentation.
    """
    if self._tp_rank == 0:
        output = self._engine.async_generate(
            prompt=prompt,
            sampling_params=sampling_params,
            input_ids=input_ids,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
        )
    else:
        output = None

    # Most naive implementation, can extract tensor and send via gloo if too slow
    [output] = broadcast_pyobj(
        data=[output],
        rank=self._tp_rank,
        dist_group=self._device_mesh_cpu.get_group(),
        src=self._device_mesh_cpu.mesh[0].item(),
    )

    return output


def apply_sgl_verl_engine_monkey_patch(engine: VerlEngine):
    engine.async_generate = async_generate
