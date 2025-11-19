import numpy as np
import os
import re
import time
import uuid
from typing import Any, Dict, List

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import jupyter tool functionality
from .jupyter_tool import SEMAPHORE, tool_registry
from .statistic_metrics import get_init_metrics

debug = os.getenv("SLIME_DEBUG", "False").lower() == "true"

def format_prompt(state: GenerateState, prompt: str, tool_specs: List[Dict[str, Any]]) -> str:
    # if prompt has already applied chat template
    if isinstance(prompt, str) and "<|im_start|>" in prompt:
        return prompt
    
    REASON_CONTENT = "Please reason step by step, and put your final answer within \\boxed{}."
    
    if isinstance(prompt, str):
        if not prompt.strip().lower().endswith(REASON_CONTENT.lower()):
            prompt = prompt.strip() + "\n" + REASON_CONTENT
        formatted_prompt = state.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tools=tool_specs or [], 
            add_generation_prompt=True, tokenize=False
        )
        return formatted_prompt
    elif isinstance(prompt, list):
        formatted_prompt = state.tokenizer.apply_chat_template(
            prompt, tools=tool_specs or [], add_generation_prompt=True, tokenize=False
        )
        return formatted_prompt
    else:
        raise ValueError(f"Invalid prompt type ({type(prompt)}). "
                         f"Sample prompt must be either a string or a list of message dicts.")

def postprocess_predictions(prediction: str):
    """Extract action and content from prediction string"""
    # Check for \boxed{...} format (only format we need for math_dapo)
    # Use a more robust regex that handles nested braces
    answer_pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    answer_match = re.search(answer_pattern, prediction, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        return "answer", content

    # Then check for <tool_call> tags (new format from Jinja2 template)
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if tool_call_match:
        try:
            import json

            # Clean up the JSON string by removing newlines and extra
            # whitespace
            json_str = tool_call_match.group(1)
            # Replace newlines in string values with \n
            # json_str = json_str.replace("\n", "\\n")
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})

            if tool_name == "python":
                if set(arguments.keys()) != {'code'}:
                    return "tool_call_error", f"Invalid function argument, only 'code' is supported."
                code = arguments.get("code", "")
                if code.strip():
                    return "code", code
            else:
                return "tool_call_error", f"Unsupported function name '{tool_name}', only 'python' is supported."
        except (json.JSONDecodeError, KeyError, AttributeError):
            return "tool_call_error", "Invalid function call format."

    return None, ""

async def execute_predictions(session_id: str, state: GenerateState, prediction: str, log_dict: dict) -> str:
    """Execute predictions and return results"""
    tool_wait_lock_time, tool_execution_time, tool_wait_lock_and_execution_times = 0.0, 0.0, 0.0
    
    action, content = postprocess_predictions(prediction)

    if action == "tool_call_error":
        next_obs = state.tokenizer.apply_chat_template([{"role": "tool", "content": content}], 
                                                       add_generation_prompt=True, tokenize=False)
        done = False
    elif action == "code":
        # Content is already the Python code (extracted by postprocess_predictions)
        code = content.strip()
        if code:
            tool_wait_lock_start_time = time.time()
            async with SEMAPHORE:
                tool_wait_lock_end_time = time.time()
                tool_wait_lock_time = tool_wait_lock_end_time - tool_wait_lock_start_time
                
                tool_execution_start_time = time.time()
                result = await tool_registry.execute_tool("python", {"code": code}, session_id)
                tool_execution_end_time = time.time()
                
                tool_execution_time = tool_execution_end_time - tool_execution_start_time
                tool_wait_lock_and_execution_times = tool_execution_end_time - tool_wait_lock_start_time
        else:
            result = "Error: No Python code found"

        next_obs = state.tokenizer.apply_chat_template([{"role": "tool", "content": result}], 
                                                       add_generation_prompt=True, tokenize=False)
        done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = ""
        done = True

    log_dict["tool_wait_lock_times"].append(tool_wait_lock_time)
    log_dict["tool_execution_times"].append(tool_execution_time)
    log_dict["tool_wait_lock_and_execution_times"].append(tool_wait_lock_and_execution_times)

    return next_obs, done

def postprocess_sample(sample: Sample, prompt_token_ids: List[int], response_token_ids: List[int], 
                       loss_masks: List[int], response: str, max_new_tokens: int, tokenizer, log_dict: dict) -> Sample:
    if len(response_token_ids) > max_new_tokens:
        response_token_ids = response_token_ids[:max_new_tokens]
        response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        loss_masks = loss_masks[:max_new_tokens]
        sample.status = Sample.Status.TRUNCATED
    
    # Set sample attributes
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    sample.train_metadata = postprocess_log_dict(sample, log_dict)
    print(f"[session_id: {log_dict.get('session_id')}, group idx: {sample.group_index}, sample idx: {sample.index}] "\
          f"debug log dict: {sample.train_metadata}")
    return sample

def postprocess_log_dict(sample: Sample, log_dict: dict) -> dict:
    print_prefix = f"[session_id: {log_dict.get('session_id')}, group idx: {sample.group_index}, sample idx: {sample.index}]"

    log_dict["total_length"] = len(sample.prompt) + len(sample.response)
    log_dict["total_token_length"] = len(sample.tokens)
    log_dict["response_length"] = len(sample.response)
    log_dict["response_token_length"] = sample.response_length
    log_dict["tool_call_count"] = sample.response.count("<tool_response>")
    
    eps = 1e-8  # to avoid division by zero
    # round statistics
    if len(log_dict["round_total_times"]) > len(log_dict["sgl_generation_times"]):
        log_dict["sgl_generation_times"].extend(
            [0.0] * (len(log_dict["round_total_times"]) - len(log_dict["sgl_generation_times"]))
        )
    assert len(log_dict["round_total_times"]) == len(log_dict["sgl_generation_times"]), \
        f"{print_prefix} len(round_total_times): {len(log_dict['round_total_times'])}, len(sgl_generation_times): "\
        f"{len(log_dict['sgl_generation_times'])} are not equal."
    # 每轮SGLang生成时间占该轮生成回复总时间的比例
    log_dict["sgl_generation_time_ratios"] = (
        np.array(log_dict["sgl_generation_times"]) / (np.array(log_dict["round_total_times"]) + eps)
    ).tolist()

    assert len(log_dict["tool_execution_times"]) == len(log_dict["tool_wait_lock_times"]) == len(log_dict["tool_wait_lock_and_execution_times"]), \
        f"{print_prefix} len(tool_execution_times): {len(log_dict['tool_execution_times'])}, len(tool_wait_lock_times): {len(log_dict['tool_wait_lock_times'])}, "\
        f"len(tool_wait_lock_and_execution_times): {len(log_dict['tool_wait_lock_and_execution_times'])} are not equal."
    if len(log_dict["round_total_times"]) > len(log_dict["tool_execution_times"]):
        extend_len = len(log_dict["round_total_times"]) - len(log_dict["tool_execution_times"])
        log_dict["tool_execution_times"].extend([0.0] * extend_len)
        log_dict["tool_wait_lock_times"].extend([0.0] * extend_len)
        log_dict["tool_wait_lock_and_execution_times"].extend([0.0] * extend_len)
    assert len(log_dict["round_total_times"]) == len(log_dict["tool_wait_lock_and_execution_times"]), \
        f"{print_prefix} len(round_total_times): {len(log_dict['round_total_times'])}, len(tool_wait_lock_and_execution_times): "\
        f"{len(log_dict['tool_wait_lock_and_execution_times'])} are not equal."

    # 每轮tool调用时间占该轮总tool调用时间的比例
    log_dict["tool_execution_time_ratios_for_tool_time"] = (
        np.array(log_dict["tool_execution_times"]) / (np.array(log_dict["tool_wait_lock_and_execution_times"]) + eps)
    ).tolist()
    # 每轮等待获取执行tool许可信号量的总时间占所有tool调用时间的比例
    log_dict["tool_wait_lock_time_ratios_for_tool_time"] = (
        np.array(log_dict["tool_wait_lock_times"]) / (np.array(log_dict["tool_wait_lock_and_execution_times"]) + eps)
    ).tolist()
    # 每轮等待获取执行tool许可信号量+执行tool时间占该轮生成回复总时间的比例
    log_dict["tool_wait_lock_and_execution_time_ratios"] = (
        np.array(log_dict["tool_wait_lock_and_execution_times"]) / (np.array(log_dict["round_total_times"]) + eps)
    ).tolist()

    # sample statistics
    log_dict["total_tool_execution_time"] = sum(log_dict["tool_execution_times"])
    log_dict["total_tool_wait_lock_time"] = sum(log_dict["tool_wait_lock_times"])
    log_dict["total_tool_wait_lock_and_execution_time"] = sum(log_dict["tool_wait_lock_and_execution_times"])
    log_dict["total_sgl_generation_time"] = sum(log_dict["sgl_generation_times"])
    log_dict["total_time"] = sum(log_dict["round_total_times"])
    # 执行tool消耗的总时间占所有tool调用时间的比例
    log_dict["total_tool_execution_time_ratio_for_total_tool_time"] = (
        log_dict["total_tool_execution_time"] / (log_dict["total_tool_wait_lock_and_execution_time"] + eps)
    )
    # 等待获取执行tool许可信号量的总时间占所有tool调用时间的比例
    log_dict["total_tool_wait_lock_time_ratio_for_total_tool_time"] = (
        log_dict["total_tool_wait_lock_time"] / (log_dict["total_tool_wait_lock_and_execution_time"] + eps)
    )
    # 等待获取执行tool许可信号量+执行tool的总时间占生成回复总时间的比例
    log_dict["total_tool_wait_lock_and_execution_time_ratio"] = (
        log_dict["total_tool_wait_lock_and_execution_time"] / (log_dict["total_time"] + eps)
    )
    # SGLang生成的总时间占生成回复总时间的比例
    log_dict["total_sgl_generation_time_ratio"] = (
        log_dict["total_sgl_generation_time"] / (log_dict["total_time"] + eps)
    )
    
    return log_dict

def report_wandb(log_dict: dict):
    report_dict = {
        f"debug/{k}": v for k, v in log_dict.items() if not isinstance(v, list)
    }
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(report_dict)
    except ImportError:
        pass  # wandb not available

async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls"""
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    session_id = "generate_" + uuid.uuid4().hex

    # Set up the initial prompt with system prompt and tools (outside the loop)
    tool_specs = tool_registry.get_tool_specs()
    # Count available tools (from tool_specs)
    available_tools = len(tool_specs)

    prompt = format_prompt(state, sample.prompt, tool_specs)
    if debug:
        print(f"[session_id: {session_id}] sample.prompt:\n {sample.prompt}\nFormatted prompt:\n {prompt}")
    # convert sample.prompt to formatted prompt
    sample.prompt = prompt

    prompt_token_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0  # Track actual tool call rounds
    max_new_tokens = sampling_params["max_new_tokens"]
    turn = 0

    debug_log_dict = get_init_metrics()
    debug_log_dict["available_tools"] = available_tools
    debug_log_dict["session_id"] = session_id

    try:
        while True:
            round_start_time = time.time()
            sampling_params["max_new_tokens"] = max_new_tokens - len(response_token_ids)

            if sampling_params["max_new_tokens"] <= 0:
                print(f"[session_id: {session_id}] Response longer than expected. Max new tokens: {max_new_tokens}, "
                    f"total tokens: {len(prompt_token_ids) + len(response_token_ids)}, prompt tokens: "
                    f"{len(prompt_token_ids)}, response tokens: {len(response_token_ids)}, prompt: {sample.prompt}")
                
                sample.status = Sample.Status.TRUNCATED
                return postprocess_sample(sample, prompt_token_ids, response_token_ids, loss_masks, response, 
                                          max_new_tokens, state.tokenizer, debug_log_dict)

            # Prepare payload for sglang server
            payload = {
                "input_ids": prompt_token_ids + response_token_ids,
                "sampling_params": sampling_params,
            }

            # # Log payload to wandb for debugging
            # try:
            #     import wandb

            #     if wandb.run is not None:
            #         # Count tools used in the current response
            #         tools_used = response.count("<tool_call>")

            #         wandb.log(
            #             {
            #                 "debug/total_length": len(prompt + response),
            #                 "debug/total_token_length": len(prompt_token_ids + response_token_ids),
            #                 "debug/response_length": len(response),
            #                 "debug/response_token_length": len(response_token_ids),
            #                 "debug/available_tools": available_tools,
            #                 "debug/tools_used": tools_used,
            #                 "debug/turn": turn,
            #             }
            #         )
            # except ImportError:
            #     pass  # wandb not available

            sgl_generation_start_time = time.time()
            output = await post(url, payload)
            _log_duration_time(sgl_generation_start_time, debug_log_dict, "sgl_generation_times")

            # Handle abort
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                _log_duration_time(round_start_time, debug_log_dict, "round_total_times")
                sample.status = Sample.Status.ABORTED
                return postprocess_sample(sample, prompt_token_ids, response_token_ids, loss_masks, response, 
                                          max_new_tokens, state.tokenizer, debug_log_dict)

            cur_response_token_ids = output["output_ids"]
            cur_response = state.tokenizer.decode(cur_response_token_ids, skip_special_tokens=False)
            if debug:
                print(f"[session_id: {session_id}] Current response:\n{cur_response}")

            response += cur_response
            response_token_ids += cur_response_token_ids
            loss_masks += [1] * len(cur_response_token_ids)

            # Check length limit
            if output["meta_info"]["finish_reason"]["type"] == "length":
                _log_duration_time(round_start_time, debug_log_dict, "round_total_times")
                break

            next_obs, done = await execute_predictions(session_id, state, cur_response, debug_log_dict)
            if done:
                _log_duration_time(round_start_time, debug_log_dict, "round_total_times")
                break
            if debug:
                # 观察tool_response apply_chat_template后的输出结果
                print(f"[session_id: {session_id}] Next observation: {next_obs}")

            # Count tool calls (when we get tool response output, it means a tool was called)
            if "<tool_response>" in next_obs:
                tool_call_count += 1

            assert next_obs != "", "Next observation should not be empty."
            obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
            response += next_obs
            response_token_ids += obs_tokens_ids
            loss_masks += [0] * len(obs_tokens_ids)
            turn += 1

            debug_log_dict["turn"] = turn
            _log_duration_time(round_start_time, debug_log_dict, "round_total_times")

        # Set status
        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED
        
        sample = postprocess_sample(sample, prompt_token_ids, response_token_ids, loss_masks, response, 
                                    max_new_tokens, state.tokenizer, debug_log_dict)
        if debug:
            print(f"[session_id: {session_id}] sample: {sample}")
    finally:
        # close jupyter session
        result = await tool_registry.jupyter_client.end_session(session_id)
        if debug:
            print(f"[session_id: {session_id}] End session result: {result}")

    return sample

# 记录持续时间
def _log_duration_time(start_time: float, log_dict: dict, key: str) -> float:
    end_time = time.time()
    duration = end_time - start_time

    if log_dict.get(key) is None:
        pass
    elif isinstance(log_dict.get(key), list):
        log_dict[key].append(duration)
    else:
        log_dict[key] += duration

    return end_time