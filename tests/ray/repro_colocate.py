# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import ray

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.single_controller.ray.base import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup, create_colocated_worker_cls

from verl import DataProto
from time import sleep

@ray.remote
class Actor(Worker):

    def __init__(self) -> None:
        super().__init__()
        sleep(60)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def add(self, data: DataProto):
        data.batch['a'] += self.rank
        sleep(60)
        return data


@ray.remote
class Critic(Worker):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        sleep(60)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def sub(self, data: DataProto):
        data.batch['a'] -= self.config['b']
        sleep(60)
        return data

@ray.remote
class Reference(Worker):

    def __init__(self) -> None:
        super().__init__()
        sleep(60)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def add(self, data: DataProto):
        data.batch['a'] -= self.rank
        sleep(60)
        return data

@ray.remote
class Reward(Worker):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        sleep(60)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def sub(self, data: DataProto):
        data.batch['a'] += self.config['b']
        sleep(60)
        return data

@ray.remote
def test_colocated_workers():
    import torch
    data = DataProto.from_dict({'a': torch.zeros(10)})
    # create separate workers on the same resource pool
    actor_cls = RayClassWithInitArgs(cls=Actor)
    critic_cls = RayClassWithInitArgs(cls=Critic, config={'b': 10})
    rm_cls = RayClassWithInitArgs(cls=Reward, config={'b': 20})
    ref_cls = RayClassWithInitArgs(cls=Reference)

    process_on_nodes = [2]

    resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, max_colocate_count=1)

    # create colocated workers
    all_wg = {}
    cls_dict = {'actor': actor_cls, 'critic': critic_cls, 'ref': ref_cls, 'reward': rm_cls}
    ray_cls_with_init = create_colocated_worker_cls(cls_dict)
    wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    spawn_wg = wg_dict.spawn(prefix_set=cls_dict.keys())
    all_wg.update(spawn_wg)

    colocated_actor_wg = all_wg['actor']
    colocated_critic_wg = all_wg['critic']
    colocated_ref_wg = all_wg['ref']
    colocated_reward_wg = all_wg['reward']

    for i in range(20):
        print(f'iteration {i}')
        actor_output = colocated_actor_wg.add(data)
        critic_output = colocated_critic_wg.sub(data)
        ref_output = colocated_ref_wg.add(data)
        rm_output = colocated_reward_wg.sub(data)
        sleep(5)

    all_meta = {}
    for wg in spawn_wg:
        all_meta[wg] = spawn_wg[wg].get_meta()

    # 4 role
    assert len(all_meta) == len(cls_dict)
    # each role locate on both processes
    assert len(all_meta['actor']) == 2
    assert len(all_meta['critic']) == 2
    assert len(all_meta['ref']) == 2
    assert len(all_meta['reward']) == 2
    # pid is int
    assert isinstance(all_meta['critic'][0][-1], int)
    # actor name is str
    assert isinstance(all_meta['actor'][0][-2], str)
    assert len(all_meta['actor'][0][-2]) > 0
    # 4 pids
    all_pids = []
    for wg in all_meta:
        for meta in all_meta[wg]:
            all_pids.append(meta[-1])
    assert len(set(all_pids)) == process_on_nodes[0]


def main():
    ray.init()

    ray.get(test_colocated_workers.remote())
    
    ray.shutdown()

if __name__ == '__main__':
    main()