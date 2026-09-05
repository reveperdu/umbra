import logging
import os
import readline  # noqa: F401
import subprocess

import requests

RUN_SHELL_PREFIX = "[RUN]"
SHELL_OUTPUT_PREFIX = "[SHELL]"
LOG_LEVEL = logging.INFO
config_api = {
    "url": "https://api.deepseek.com/chat/completions",
    "modelConfig": {
        "model": "deepseek-v4-flash",
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    },
}
current_state = {"should_ask_input": True,"context":[]}
# the core idea is, the system prompt is, concating a series of text files.
# using md for convention, but i don't expect ##s and **s in the prompt itself.
config_agent = {"sysprompt_files": ["test.md"]}
if "API_KEY" in os.environ:
    apikey = os.environ["API_KEY"]
else:
    raise RuntimeError("No API Key Found.")


def construct_system_prompt():
    sysprompt = ""
    for fname in config_agent["sysprompt_files"]:
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
    if RUN_SHELL_PREFIX in output:
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
        logger.debug("shell output:\n"+shell_output)
        current_state["context"].append(
            {"role": "user", "content": SHELL_OUTPUT_PREFIX + shell_output}
        )
    else:
        current_state["should_ask_input"] = True
        print(output)


def mainloop():
    if current_state["should_ask_input"]:
        user_input = input("> ")
        if user_input.strip():
            current_state["should_ask_input"] = False
            current_state["context"].append({"role": "user", "content": user_input})
        else:
            logger.warning("input is empty. doing nothing")
            return
    model_output = generate(current_state["context"], config_api)
    current_state["context"].append({"role": "assistant", "content": model_output})
    process_model_output(model_output)


# === main section ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s:%(message)s")
current_state["context"] = [{"role": "system", "content": construct_system_prompt()}]
current_state["should_ask_input"] = True
while True:
    try:
        mainloop()
    except KeyboardInterrupt:
        current_state["should_ask_input"] = True
