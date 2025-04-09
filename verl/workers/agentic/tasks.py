import asyncio

import aiohttp
import traceback

import torch

from verl.utils.swedev_utils import *


async def dummy(*_, **__):
    pass


def _pre_process_inputs(pad_token_id, token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = token_ids[non_pad_index:].tolist()
    return token_ids


# TODO: this is just a temporary approach for dr getting reward. should be moved to a backend.
async def swedev_start(instance_id, input_ids, tokenizer):
    try:
        result = await initialize_runtime(instance_id.item())
        print(result)
        return {
            "prompt_ids": _pre_process_inputs(tokenizer.pad_token_id, input_ids),
            "sid": result["sid"],
            "sids": int(
                result["sid"]
            ),  # will be treated as a obs metric, thus, will be gathered into batch, and later used in reward acquisition
        }
    except Exception as e:
        # TODO: return true for handle api instead of raising an error
        print(f"Error processing instance: {e}")
        # in original logic, mismatched sids count and instance_ids count will cause error eventually, better raise now
        raise


async def openai_chat_start(index, name, url):
    # TODO: exception handling in this function is tricky
    print(f"starting session {index=} @ {torch.distributed.get_rank()=}")
    if isinstance(index, torch.Tensor):
        index = index.item()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.post(url + "/start_sample", json={
                "index": index,
                "name": name,
        }) as response:
            ret = await response.json()
    ret["sid"] = response.headers["session_id"]
    return ret


async def openai_chat_obs(message, sid, url, **_):
    payload = {"messages": [message]}
    header = {"session_id": str(sid)}
    metrics = {"failed_times": 0, "observations_times": 1}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(url + "/interact", json=payload, headers=header) as response:
                assert response.status == 200, f"Wrong status code: {sid=} {response.status=} {await response.text()}"
                ret = await response.json()
                metrics["failed_times"] = 0
    except Exception as e:
        print(f"API call failed: {e}")
        traceback.print_exc()
        ret = {"messages": [{"role": "user", "content": "Connection Error"}], "finish": False, "reward": -1}
        metrics["failed_times"] += 1
    ret["metrics"] = ret.get("metrics", {}) | metrics
    if torch.distributed.get_rank() == 0:
        print(f"Observation: {ret}")
    return ret


async def openai_chat_end(sid, done, url):
    if done:
        return
    payload = {"session_id": sid, "done": done}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url + "/cancel", json=payload) as response:
                pass
    except Exception as e:
        print(f"API call failed when ending: {e}")


async def swe_dev_obs(action_ids, sid, tokenizer, **kwargs):
    action = tokenizer.decode(action_ids, skip_special_tokens=False)
    if is_stop(action):
        print(f"Action stop: {action}")
        return {"done": True, "ids": [], "observation_times": 0}

    result = call_observation_api(sid, action)
    # TODO(haoran): handle here
    try:
        obs = result["content"]
    except:
        obs = "Error"
    return {"done": False, "ids": tokenizer.encode(obs), "observation_times": 1}


async def swe_dev_end(sid, _done):
    await asyncio.to_thread(call_postprocess_api, sid)
