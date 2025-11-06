import os
import re
import uuid
from typing import Any, Dict, List

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import reward models
try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError:
    raise ImportError("MathDapo is not installed")

# Import jupyter tool functionality
from .jupyter_tool import SEMAPHORE, TOOL_CONFIGS, tool_registry


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

async def execute_predictions(session_id: str, state: GenerateState, prediction: str) -> str:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)

    if action == "tool_call_error":
        next_obs = state.tokenizer.apply_chat_template([{"role": "tool", "content": content}], 
                                                       add_generation_prompt=True, tokenize=False)
        done = False
    if action == "code":
        # Content is already the Python code (extracted by postprocess_predictions)
        code = content.strip()
        if code:
            async with SEMAPHORE:
                result = await tool_registry.execute_tool("python", {"code": code}, session_id)
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

    return next_obs, done

def postprocess_sample(sample: Sample, prompt_token_ids: List[int], 
                       response_token_ids: List[int], loss_masks: List[int], response: str) -> Sample:
    # Set sample attributes
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # # Store payload information for wandb logging
    # sample.payload_text = prompt + response
    # sample.payload_has_system = "<|im_start|>system" in prompt + response
    # sample.payload_has_tools = "# Tools" in prompt + response

    # # Store tool call count for reward calculation
    # sample.tool_call_count = tool_call_count
    return sample

async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls"""
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    session_id = "generate_" + uuid.uuid4().hex
    debug = os.getenv("SLIME_DEBUG", "False").lower() == "true"

    # Set up the initial prompt with system prompt and tools (outside the loop)
    tool_specs = tool_registry.get_tool_specs()

    if isinstance(sample.prompt, str):
        # if prompt has already applied chat template
        if "<|im_start|>" in sample.prompt:
            prompt = sample.prompt
        else:
            prompt = state.tokenizer.apply_chat_template(
                [{"role": "user", "content": sample.prompt}], tools=tool_specs or [], 
                add_generation_prompt=True, tokenize=False
            )
    elif isinstance(sample.prompt, list):
        prompt = state.tokenizer.apply_chat_template(
            sample.prompt, tools=tool_specs or [], add_generation_prompt=True, tokenize=False
        )
    else:
        raise ValueError(f"Invalid prompt type ({type(sample.prompt)}). "
                         f"Sample prompt must be either a string or a list of message dicts.")
    if debug:
        print(f"sample.prompt:\n {sample.prompt}\nFormatted prompt:\n {prompt}")

    prompt_token_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0  # Track actual tool call rounds
    max_new_tokens = sampling_params["max_new_tokens"]
    turn = 0

    try:
        while True:
            sampling_params["max_new_tokens"] = max_new_tokens - len(response_token_ids)

            if sampling_params["max_new_tokens"] <= 0:
                print(f"[session_id: {session_id}] Response longer than expected. Max new tokens: {max_new_tokens}, "
                    f"total tokens: {len(prompt_token_ids) + len(response_token_ids)}, prompt tokens: "
                    f"{len(prompt_token_ids)}, response tokens: {len(response_token_ids)}, prompt: {sample.prompt}")
                
                sample.status = Sample.Status.TRUNCATED
                return postprocess_sample(sample, prompt_token_ids, response_token_ids, loss_masks, response)

            # Prepare payload for sglang server
            payload = {
                "input_ids": prompt_token_ids + response_token_ids,
                "sampling_params": sampling_params,
            }

            # Log payload to wandb for debugging
            try:
                import wandb

                if wandb.run is not None:
                    # Count available tools (from tool_specs)
                    available_tools = len(tool_specs)
                    # Count tools used in the current response
                    tools_used = response.count("<tool_call>")

                    wandb.log(
                        {
                            "debug/payload_length": len(prompt + response),
                            "debug/payload_token_length": len(prompt_token_ids + response_token_ids),
                            "debug/response_length": len(response),
                            "debug/response_token_length": len(response_token_ids),
                            "debug/available_tools": available_tools,
                            "debug/tools_used": tools_used,
                            "debug/turn": turn,
                        }
                    )
            except ImportError:
                pass  # wandb not available

            output = await post(url, payload)

            # Handle abort
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                sample.status = Sample.Status.ABORTED
                return postprocess_sample(sample, prompt_token_ids, response_token_ids, loss_masks, response)

            cur_response_token_ids = output["output_ids"]
            cur_response = state.tokenizer.decode(cur_response_token_ids, skip_special_tokens=False)
            if debug:
                print(f"[session_id: {session_id}] Current response:\n{cur_response}")

            response += cur_response
            response_token_ids += cur_response_token_ids
            loss_masks += [1] * len(cur_response_token_ids)

            # Check length limit
            if output["meta_info"]["finish_reason"]["type"] == "length":
                break

            next_obs, done = await execute_predictions(session_id, state, cur_response)
            if done:
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

        sample = postprocess_sample(sample, prompt_token_ids, response_token_ids, loss_masks, response)

        # Set status
        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED

    finally:
        # close jupyter session
        result = await tool_registry.jupyter_client.end_session(session_id)
        if debug:
            print(f"[session_id: {session_id}] End session result: {result}")

    return sample
