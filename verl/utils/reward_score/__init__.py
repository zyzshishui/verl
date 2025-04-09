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
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, question="", tokenizer=None):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['hotpotqa', 'hotpotQA']:
        from . import hotpotqa
        res = hotpotqa.compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            question=question,
            # extractor_urls=["http://172.18.75.153:8000/v1"],
            # checker_urls=["http://172.18.75.109:8000/v1"],
            # extractor_urls=["http://172.20.68.119:8000/v1"],
            # checker_urls=["http://172.20.69.226:8000/v1"],
            extractor_urls=["http://172.21.1.152:8000/v1"],
            checker_urls=["http://172.21.1.97:8000/v1"],
            tokenizer=tokenizer
        )
        # lurui: must return a float
        print(f"judgement by hotpotqa: {res}")
        if isinstance(res, float):
            return float(res)
        else:
            raise NotImplementedError
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    else:
        print(f"Unknown data source: {data_source}")
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
