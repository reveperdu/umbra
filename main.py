import argparse
import json
import logging
import os
import re
import readline  # noqa: F401
import subprocess
import sys

import requests

RUN_SHELL_PREFIX = "[RUN]"
SHELL_OUTPUT_PREFIX = "[SHELL]"
LOG_LEVEL = logging.INFO
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
args = parser.parse_args()
config_path = "config.json"
if args.config is not None:
    config_path = args.config
with open(config_path) as f:
    config = json.load(f)
# the core idea is, the system prompt is, concating a series of text files.
# using md for convention, but i don't expect ##s and **s in the prompt itself.
current_state = {"should_ask_input": True, "context": []}

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


def generate(context: list[dict[str, str]], config: dict) -> str:
    header = {
        "Authorization": "Bearer " + apikey,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    data = {"messages": context} | config["modelConfig"]
    resp = requests.post(config["url"], headers=header, json=data)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def process_model_output(output: str):
    # considerations for deciding shell or user message is that,
    # the model might output a user message followed by a shell command
    # eg "okay i'll do it\n[RUN]...", or a prefix inside a sentence
    # without intent to run shell, eg "i know. i should output [RUN] when ..."
    # these are the two cases that need to be handled correctly.
    if re.search(r"(?m)^\[RUN\]", output):
        user_msg, _, cmd = output.partition(RUN_SHELL_PREFIX)
        if user_msg:
            print(user_msg)
        logger.info("shell:" + cmd)
        # NOTE using check=true will raise exception on error, making shell_result_obj invalid.
        # but the agent needs to see the error message, so check=false makes this more intuitive.
        shell_result = subprocess.run(
            cmd, shell=True, capture_output=True, check=False, text=True
        )
        shell_output = shell_result.stdout + shell_result.stderr
        if shell_result.returncode != 0:
            logger.warning("shell command returned non-zero")
        logger.debug("shell output:\n" + shell_output)
        current_state["context"].append(
            {"role": "user", "content": SHELL_OUTPUT_PREFIX + shell_output}
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
