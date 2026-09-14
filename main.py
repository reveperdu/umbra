import argparse
import json
import logging
import os
import re
import readline  # noqa: F401
import subprocess
import sys

import requests

LOG_LEVEL = logging.INFO
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()
config_path = "config.json"
prompt_macro_path = "prompt_macro.json"
if args.config is not None:
    config_path = args.config
with open(config_path) as f:
    config = json.load(f)
with open(prompt_macro_path) as f:
    prompt_macro = json.load(f)
if args.verbose:
    LOG_LEVEL = logging.DEBUG
# the core idea is, the system prompt is, concating a series of text files.
# using md for convention, but i don't expect ##s and **s in the prompt itself.
current_state = {
    "should_ask_input": True,
    "context": [],
    "token_stats": {"input": 0, "cache": 0, "output": 0, "total": 0, "last_context": 0},
}

if "API_KEY" in os.environ:
    apikey = os.environ["API_KEY"]
else:
    raise RuntimeWarning("No API Key Found.")


def construct_system_prompt():
    sysprompt = ""
    for fname in config["agent"]["sysprompt_files"]:
        with open(fname) as f:
            contents = f.read()
            sysprompt = sysprompt + f"\n# {fname}\n" + contents
    return sysprompt


def record_usage(usage_data: dict):
    tstat = current_state["token_stats"]
    cache = usage_data["prompt_tokens_details"]["cached_tokens"]
    tstat["cache"] += cache
    tstat["input"] += usage_data["prompt_tokens"] - cache
    tstat["output"] += usage_data["completion_tokens"]
    tstat["total"] = tstat["input"] + tstat["cache"] + tstat["output"]
    tstat["last_context"] = usage_data["prompt_tokens"]


def generate(context: list[dict[str, str]], config: dict) -> str:
    header = {
        "Authorization": "Bearer " + apikey,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    data = {"messages": context} | config["model_config"]
    resp = requests.post(config["url"], headers=header, json=data)
    resp.raise_for_status()
    data = resp.json()
    logger.debug(data)
    record_usage(data["usage"])
    return data["choices"][0]["message"]["content"]


def process_model_output(output: str):
    # after the model repeatedly mix the shell command
    # and freeform text, (particularly on non-cot)
    # pairing [/RUN] is added to address this.
    mo = re.search(r"(?s)\[RUN\](.*?)\[/RUN\]", output)
    if mo:
        msg_head, cmd, msg_tail = output[: mo.start()], mo[1], output[mo.end() :]
        if msg_head:
            print(msg_head)
        logger.info("shell:" + cmd)
        if msg_tail:
            print(msg_tail)
        # NOTE using check=true will raise exception on error, making shell_result_obj invalid.
        # but the agent needs to see the error message, so check=false makes this more intuitive.
        shell_result = subprocess.run(
            cmd, shell=True, capture_output=True, check=False, text=True
        )
        shell_output = shell_result.stdout + shell_result.stderr
        if shell_output == "":
            # letting the agent know the output is empty,
            # rather than leaving an ambiguous `[SHELL]` in the context
            shell_output = "(no output)"
        if shell_result.returncode != 0:
            logger.warning("shell command returned non-zero")
        logger.debug("shell output:\n" + shell_output)
        current_state["context"].append(
            {"role": "user", "content": "[SHELL]" + shell_output}
        )
    else:
        current_state["should_generate"] = False
        out_str = "\n" + output + "\n"
        print(out_str)


def handle_user_command(cmd: str):
    match cmd.split():
        case ["save"]:
            with open("session-log.json", "w") as f:
                json.dump(current_state["context"], f, ensure_ascii=False, indent=4)
        case ["exit"]:
            sys.exit(0)
        case ["token"]:
            print(current_state["token_stats"])
        case _:
            if cmd in prompt_macro:
                current_state["context"].append(
                    {"role": "user", "content": prompt_macro[cmd]}
                )
                current_state["should_generate"] = True
            else:
                logger.info("command not found:" + cmd)


def mainloop():
    if not current_state["should_generate"]:
        user_input = input("> ")
        if user_input.startswith("/"):
            handle_user_command(user_input.removeprefix("/"))
        elif not user_input.strip():
            logger.warning("input is empty. doing nothing")
        else:
            current_state["context"].append({"role": "user", "content": user_input})
            current_state["should_generate"] = True
    if current_state["should_generate"]:
        model_output = generate(current_state["context"], config["api"])
        current_state["context"].append({"role": "assistant", "content": model_output})
        process_model_output(model_output)


# === main section ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s:%(message)s")
current_state["context"] = [{"role": "system", "content": construct_system_prompt()}]
current_state["should_generate"] = False
while True:
    try:
        mainloop()
    except KeyboardInterrupt:
        current_state["should_generate"] = False
