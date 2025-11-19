import sys
from typing import List
from slime.utils.types import Sample
from slime.utils.metric_utils import compute_statistics, dict_add_prefix

def get_init_metrics():
    return {
        "total_length": 0,
        "total_token_length": 0,
        "response_length": 0,
        "response_token_length": 0,
        "available_tools": 0,
        "tool_call_count": 0,
        "turn": 0,
        "session_id": "",

        # sample statistics
        # 该条sample执行tool消耗的总时长
        "total_tool_execution_time": 0.0,
        # 该条sample等待获取 执行tool semaphore的总时长
        "total_tool_wait_lock_time": 0.0,
        # 该条sample等待获取 执行tool semaphore+执行tool的总时长
        "total_tool_wait_lock_and_execution_time": 0.0,
        # 该条sample 调用SGLang的总时长
        "total_sgl_generation_time": 0.0,
        # 该条sample生成完整trajectory的总时长
        "total_time": 0.0,

        # 执行tool消耗的总时长占tool总调用时长的比例
        "total_tool_execution_time_ratio_for_total_tool_time": 0.0,
        # 等待获取执行tool semaphore的总时长占tool总调用时长的比例
        "total_tool_wait_lock_time_ratio_for_total_tool_time": 0.0,
        # 等待获取执行tool semaphore+执行tool消耗的总时长占生成完整trajectory总时长的比例
        "total_tool_wait_lock_and_execution_time_ratio": 0.0,
        # 调用SGLang的总时长占生成完整trajectory总时长的比例
        "total_sgl_generation_time_ratio": 0.0,

        # round statistics
        # 每轮执行tool消耗的时长
        "tool_execution_times": [],
        # 每轮等待获取 执行tool semaphore的时长
        "tool_wait_lock_times": [],
        # 每轮等待获取 执行tool semaphore+执行tool的时长
        "tool_wait_lock_and_execution_times": [],
        # 每轮调用SGLang的时长
        "sgl_generation_times": [],
        # 每轮总时长
        "round_total_times": [],

        # 每轮执行tool消耗的时长占该轮tool调用总时长的比例
        "tool_execution_time_ratios_for_tool_time": [],
        # 每轮等待获取 执行tool semaphore时长占该轮tool调用总时长的比例
        "tool_wait_lock_time_ratios_for_tool_time": [],
        # 每轮等待获取 执行tool semaphore+执行tool时长占该轮生成完整trajectory总时长的比例
        "tool_wait_lock_and_execution_time_ratios": [],
        # 每轮调用SGLang的时长占该轮生成完整trajectory总时长的比例
        "sgl_generation_time_ratios": [],
    }

# 记录一组rollout或一轮rollout的TIR统计指标
def log_tir_stat_metrics(samples: List[Sample]):
    total_lengths = [
        sample.train_metadata["total_length"] for sample in samples 
        if sample.train_metadata and "total_length" in sample.train_metadata
    ]
    total_length_stats = compute_statistics(total_lengths) if total_lengths else {}

    total_token_lengths = [
        sample.train_metadata["total_token_length"] for sample in samples 
        if sample.train_metadata and "total_token_length" in sample.train_metadata
    ]
    total_token_length_stats = compute_statistics(total_token_lengths) if total_token_lengths else {}

    turns = [
        sample.train_metadata["turn"] for sample in samples 
        if sample.train_metadata and "turn" in sample.train_metadata
    ]
    turn_stats = compute_statistics(turns) if turns else {}

    tool_call_counts = [
        sample.train_metadata["tool_call_count"] for sample in samples 
        if sample.train_metadata and "tool_call_count" in sample.train_metadata
    ]
    tool_call_count_stats = compute_statistics(tool_call_counts) if tool_call_counts else {}

    tool_execution_times = [
        sample.train_metadata["total_tool_execution_time"] for sample in samples 
        if sample.train_metadata and "total_tool_execution_time" in sample.train_metadata
    ]
    tool_execution_time_stats = compute_statistics(tool_execution_times) if tool_execution_times else {}

    tool_wait_lock_times = [
        sample.train_metadata["total_tool_wait_lock_time"] for sample in samples 
        if sample.train_metadata and "total_tool_wait_lock_time" in sample.train_metadata
    ]
    tool_wait_lock_time_stats = compute_statistics(tool_wait_lock_times) if tool_wait_lock_times else {}

    tool_wait_lock_and_execution_times = [
        sample.train_metadata["total_tool_wait_lock_and_execution_time"] for sample in samples 
        if sample.train_metadata and "total_tool_wait_lock_and_execution_time" in sample.train_metadata
    ]
    tool_wait_lock_and_execution_time_stats = compute_statistics(tool_wait_lock_and_execution_times) if tool_wait_lock_and_execution_times else {}

    sgl_generation_times = [
        sample.train_metadata["total_sgl_generation_time"] for sample in samples 
        if sample.train_metadata and "total_sgl_generation_time" in sample.train_metadata
    ]
    sgl_generation_time_stats = compute_statistics(sgl_generation_times) if sgl_generation_times else {}

    rollout_times = [
        sample.train_metadata["total_time"] for sample in samples 
        if sample.train_metadata and "total_time" in sample.train_metadata
    ]
    rollout_time_stats = compute_statistics(rollout_times) if rollout_times else {}

    tool_execution_time_ratio_for_tool_time = (
        (sum(tool_execution_times) / sum(tool_wait_lock_and_execution_times)) 
        if sum(tool_wait_lock_and_execution_times) > 0 else 0.0
    )
    tool_wait_lock_time_ratio_for_tool_time = (
        (sum(tool_wait_lock_times) / sum(tool_wait_lock_and_execution_times)) 
        if sum(tool_wait_lock_and_execution_times) > 0 else 0.0
    )
    tool_wait_lock_and_execution_time_ratio = (
        (sum(tool_wait_lock_and_execution_times) / sum(rollout_times)) 
        if sum(rollout_times) > 0 else 0.0
    )
    sgl_generation_time_ratio = (
        (sum(sgl_generation_times) / sum(rollout_times)) 
        if sum(rollout_times) > 0 else 0.0
    )

    tir_stat_metrics_dict = {
        "total_rollout_time": min(sum(rollout_times), sys.float_info.max),
        "total_sgl_generation_time": min(sum(sgl_generation_times), sys.float_info.max),
        "total_tool_execution_time": min(sum(tool_execution_times), sys.float_info.max),
        "total_tool_wait_lock_time": min(sum(tool_wait_lock_times), sys.float_info.max),
        "total_tool_wait_lock_and_execution_time": min(sum(tool_wait_lock_and_execution_times), sys.float_info.max),
        "total_turn": min(sum(turns), sys.maxsize),
        "total_tool_call_count": min(sum(tool_call_counts), sys.maxsize),

        "tool_execution_time_ratio_for_tool_time": tool_execution_time_ratio_for_tool_time,
        "tool_wait_lock_time_ratio_for_tool_time": tool_wait_lock_time_ratio_for_tool_time,
        "tool_wait_lock_and_execution_time_ratio": tool_wait_lock_and_execution_time_ratio,
        "sgl_generation_time_ratio": sgl_generation_time_ratio,

        **dict_add_prefix(total_length_stats, "total_length/"),
        **dict_add_prefix(total_token_length_stats, "total_token_length/"),
        **dict_add_prefix(rollout_time_stats, "rollout_time/"),
        **dict_add_prefix(sgl_generation_time_stats, "sgl_generation_time/"),
        **dict_add_prefix(tool_execution_time_stats, "tool_execution_time/"),
        **dict_add_prefix(tool_wait_lock_time_stats, "tool_wait_lock_time/"),
        **dict_add_prefix(tool_wait_lock_and_execution_time_stats, "tool_wait_lock_and_execution_time/"),
        **dict_add_prefix(turn_stats, "turn/"),
        **dict_add_prefix(tool_call_count_stats, "tool_call_count/"),
    }
    return tir_stat_metrics_dict