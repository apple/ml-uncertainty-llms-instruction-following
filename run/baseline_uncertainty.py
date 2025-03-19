#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import re
import json
import pickle
import argparse
import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    AutoModelForCausalLM,
)
from huggingface_hub import login
import sys

sys.path.append("../")


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
overall_instruction = """\
You are a helpful assistant.
"""


#
# Entry function
#
def get_verbalized_confidence(q, history, model, tokenizer, generation_config):
    # // Prepare prompt
    prompt = q
    print("=" * 20)
    print("Prompt:")
    print(prompt)
    print("=" * 20)

    # // Tokenizer -> optional: padding=False, truncation=False, add_special_tokens=False
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    if torch.cuda.is_available():
        attention_mask = input_ids["attention_mask"].cuda()
        input_ids = input_ids["input_ids"].cuda()

    # // Generate
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

    # // Decode output generation
    s = outputs.sequences[0][input_ids.shape[1] :]
    response = tokenizer.decode(
        s, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    print("*" * 20)
    print("Response:")
    print(response)
    print("*" * 20)
    return response, outputs


#
# Entry function
#
def get_perplexity_seqprob_entropy_from_response_fast(
    model, tokenizer, full_prompt, response_text
):
    """
    Compute the perplexity, sequence probability, and entropy of a specified response part in a full prompt.
    """

    # // Tokenize the full prompt and the response
    full_tokens = tokenizer.encode(full_prompt.strip())
    response_tokens = tokenizer.encode(response_text.strip())[1:]

    # // Find start and end of the response in tokens
    i = 1
    start_index, end_index = None, None
    while start_index is None or end_index is None:
        start_index, end_index = get_indices_from_response(
            full_tokens, response_tokens[i:-i]
        )
        i += 1
        if i > 20:
            break

    if start_index is None or end_index is None:
        raise ValueError("Response text not found in the prompt.")

    # // Make sure everything is clean going in
    for module in model.model.layers:
        module._forward_hooks.clear()

    # // Register a forward hook to capture outputs
    outputs = []

    def hook_function(module, mod_input, mod_output):
        mod_output = mod_output[0]  # // <-- output is tuple (tensor)
        # print(mod_output[0].keys())
        outputs.append(mod_output[0][:].detach())

    hook = model.model.layers[-1].register_forward_hook(hook_function)

    # // Perform a forward pass to collect outputs
    inputs = tokenizer(full_prompt, return_tensors="pt")
    with torch.no_grad():
        _ = model(**inputs)

    # // Remove the hook
    hook.remove()

    # // Extract hidden_state for the response part (token_len, hidden_dim)
    select_outputs = outputs[0][start_index : end_index + 1, :]

    # // Change it to token prob (token_len, 32000)
    logits = model.lm_head(select_outputs)

    # // Calculate probabilities using softmax
    probs = F.softmax(logits, dim=-1)
    actual_tokens = full_tokens[
        start_index + 1 : end_index + 2
    ]  # Shift for next token prediction

    neg_log_likelihoods = []
    entropies = []
    for token_idx in range(len(logits)):
        token_prob = probs[token_idx, actual_tokens[token_idx]].item()
        neg_log_likelihood = -torch.log(torch.tensor(token_prob))
        neg_log_likelihoods.append(neg_log_likelihood)
        entropy = -torch.sum(
            probs[token_idx] * torch.log(probs[token_idx].clamp(min=1e-10))
        ).item()  # // <-- Stable entropy calculation
        entropies.append(entropy)

    # Calculate perplexity, sequence probability, and average entropy
    perplexity = torch.tensor(neg_log_likelihoods).mean()
    sequence_prob = torch.tensor(neg_log_likelihoods).sum()
    average_entropy = torch.tensor(entropies).mean()

    return perplexity.item(), sequence_prob.item(), average_entropy.item()


#
# Entry function
#
def get_ptrue(instruction, response, model, tokenizer):
    """
    Instruction-generation pairs
    Return Softmax instead of log softmax
    """
    # // Make a prompt for p(true)
    prompt_ptrue = "You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to score a given output for the given instruction.\nQuestion: {q}\nResponse: {a}\nDoes the response:\n (A) Follow instructions\n (B) Not follow instructions\nThe answer is: ("
    prompt_ptrue = prompt_ptrue.format(q=instruction, a=response)
    print("=" * 20)
    print("Prompt p(true): ")
    print(prompt_ptrue)
    print("=" * 20)

    # // Tokenizer -> optional: padding=False, truncation=False, add_special_tokens=False
    input_ids = tokenizer(prompt_ptrue, return_tensors="pt", add_special_tokens=False)
    if torch.cuda.is_available():
        attention_mask = input_ids["attention_mask"].cuda()
        input_ids = input_ids["input_ids"].cuda()

    generation_config = GenerationConfig(
        do_sample=False,
        temperature=1,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_scores=True,
        output_logits=True,
        pad_token_id=tokenizer.eos_token_id,
        min_new_tokens=1,
        max_new_tokens=1,
        num_beams=1,
    )

    # // Generate
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
    logits = torch.stack(outputs.logits, dim=1)
    logits_prob = logits.softmax(-1)
    expected_token = tokenizer("A")["input_ids"][-1]
    false_token = tokenizer("B")["input_ids"][-1]
    probs = logits_prob[:, -1, expected_token].cpu().numpy()
    false_probs = logits_prob[:, -1, false_token].cpu().numpy()

    # // Decode output generation
    s = outputs.sequences[0][input_ids.shape[1] :]
    response = tokenizer.decode(
        s, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    print("*" * 20)
    print("Response p(true):")
    print(response)
    print("*" * 20)

    return probs, false_probs


#
# Helper functions
#
def get_indices_from_response(full_tokens, response_tokens):
    """Find the start and end indices of the response tokens in the full prompt tokens"""
    response_length = len(response_tokens)
    for i in range(len(full_tokens) - response_length + 1):
        if full_tokens[i : i + response_length] == response_tokens:
            return i, i + response_length - 1
    return None, None  # If not found


def format_prompt(query, history=[], input=None):
    prompt = ""
    if len(history) == 0:
        prompt += f"<s>{B_INST} {B_SYS} {overall_instruction} {E_SYS} {query} {E_INST} "
    else:
        for i, (old_query, response) in enumerate(history):
            prompt += f"{old_query} {response}</s>"
        prompt += f"<s>{B_INST} {query} {E_INST}"
    return prompt


def format_prompt_pairs(instruction, response):
    """
    Instruction-generation pairs
    """
    # // Load basic prompt format
    with open("./run/prompt.txt", "r") as fin:
        prompt = fin.read()

    # // Put instructions and generations
    pairs = dict(input=instruction, output=response)
    prompt = prompt.format_map(pairs)

    # // Change it to dict
    channels = prompt_to_chatml(prompt)
    assert channels[0]["role"] == "system"

    prompt = "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n".format(channels[0]["content"])
    for index, channel in enumerate(channels[1:]):
        if index % 2 == 0:
            assert channel["role"] == "user"
            prompt += "{} [/INST]".format(channel["content"])
            if index < len(channels) - 2:
                prompt += " "
        else:
            assert channel["role"] == "assistant"
            prompt += "{} </s><s>[INST] ".format(channel["content"])
    return prompt


def readjsonl(datapath):
    res = []
    with open(datapath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            res.append(json.loads(line))
    return res


def read_prompt_to_response_dict(input_jsonl_filename):
    """Creates dictionary matching prompt and response."""
    return_dict = {}
    with open(input_jsonl_filename, "r") as f:
        for l in f:
            example = json.loads(l)
            return_dict[example["prompt"]] = example["output"]
    return return_dict


def prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"):
    prompt = prompt.strip()
    assert prompt.endswith(end_token)
    assert prompt.startswith(start_token)

    def change_string_to_dict(to_convert):
        return {s.split("=", 1)[0]: s.split("=", 1)[1] for s in to_convert.split(" ") if len(s) > 0}

    contents = []
    for p in prompt.split("<|im_start|>")[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system"):
            if role != "system":
                _params = change_string_to_dict(role.split("system", 1)[-1])
                role = "system"
        else:
            _params = dict()

        contents.append(dict(content=content, role=role, **_params))

    return contents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        default="",
        help="Path to the data file.",
    )
    parser.add_argument(
        "--model_name_hf",
        type=str,
        required=True,
        default="",
        help="Path to the model weights.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        default="IFEval",
        help="ifeval, counterfact, biasbios, json, pronounce, json_counterfact",
    )
    parser.add_argument("--hf_use_auth_token", type=str, default=None)
    args = parser.parse_args()

    res_path = os.path.join(args.data_path, args.model_name_hf.split("/")[-1])
    os.makedirs(res_path, exist_ok=True)

    # // Load model
    login(token=args.hf_use_auth_token)

    if "Llama-2" in args.model_name_hf:
        model_config = transformers.AutoConfig.from_pretrained(args.model_name_hf)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_hf,
            device_map="auto",
            use_auth_token=args.hf_use_auth_token,
            config=model_config,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_hf,
            device_map="auto",
            use_auth_token=args.hf_use_auth_token,
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_hf, use_auth_toke=args.hf_use_auth_token
    )
    model.eval()
    print(model)

    generation_config = GenerationConfig(
        do_sample=False,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_scores=True,
        output_logits=True,
        repetition_penalty=1.0,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
    )
    print("tokenizer.eos_token_id: ", tokenizer.eos_token_id)

    # // Save prompt, response, and activations
    eval_response_path = os.path.join(res_path, args.task_type)
    os.makedirs(eval_response_path, exist_ok=True)
    response_file_name = os.path.join(
        eval_response_path, "all_eval_response_and_baseline.jsonl"
    )
    output_file = open(response_file_name, "a", encoding="utf-8")

    # // Load response file
    if args.task_type == "controlled_ver":
        response_path = f"{args.data_path}/controlled_ver.jsonl"
        inst_response_key = "output"
    elif args.task_type == "realistic_ver":
        response_path = f"{args.data_path}/realistic_ver.jsonl"
        inst_response_key = "response"
    else:
        raise NotImplementedError

    response_data = readjsonl(response_path)
    for idx, res_data in enumerate(response_data):
        instruction = res_data["prompt"]
        inst_response = res_data[inst_response_key]
        eval_prompt = format_prompt_pairs(instruction, inst_response)
        following_label = res_data["following_label"]
        instruction_id_list = res_data["instruction_id_list"]

        # // Get act
        eval_response, outputs = get_verbalized_confidence(
            eval_prompt, [], model, tokenizer, generation_config
        )

        # // Compute baseline
        perplexity, maximum_seq_prob, entropy = (
            get_perplexity_seqprob_entropy_from_response_fast(
                model, tokenizer, eval_prompt, inst_response
            )
        )
        p_true, unexpected_p_true = get_ptrue(
            instruction, inst_response, model, tokenizer
        )
        p_true, p_false = p_true[0], unexpected_p_true[0]
        normalized_p_true = p_true / (p_true + p_false)
        print(
            "perplexity, maximum_seq_prob, entropy: ",
            perplexity,
            maximum_seq_prob,
            entropy,
        )
        print("p_true: ", p_true)

        #  // Compute verbalized confidence
        try:
            eval_reponse_score = [int(s) for s in re.findall(r"\d+", eval_response)][0]
        except:
            eval_reponse_score = None
        save = dict(
            instructions=eval_prompt,
            response=inst_response,
            eval_response=eval_response,
            verbalized_confidence=eval_reponse_score,
            following_label=following_label,
            instruction_id_list=instruction_id_list,
            p_true=str(p_true),
            p_false=str(p_false),
            normalized_p_true=str(normalized_p_true),
            maximum_seq_prob=str(maximum_seq_prob),
            entropy=str(entropy),
            perplexity=str(perplexity),
        )
        output_file.write(json.dumps(save, ensure_ascii=False) + "\n")

    output_file.close()
