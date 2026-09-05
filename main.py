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
global_state = {"should_ask_input": True}
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


def generate(context: list[dict[str, str]], config: dict):
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


def mainloop():
    if global_state["should_ask_input"]:
        user_input = input("> ")
        if user_input.strip():
            global_state["should_ask_input"] = False
            current_context.append({"role": "user", "content": user_input})
        else:
            logger.warning("input is empty. doing nothing")
    current_output = generate(current_context, config_api)
    current_context.append({"role": "assistant", "content": current_output})
    if current_output.startswith(RUN_SHELL_PREFIX):
        command = current_output.removeprefix(RUN_SHELL_PREFIX)
        logger.info("\nshell:" + command)
        # NOTE using check=true will raise exception on error, making shell_result_obj invalid.
        # but the agent needs to see the error message, so check=false makes this more intuitive.
        shell_result = subprocess.run(
            command, shell=True, capture_output=True, check=False,text=True
        )
        shell_output = shell_result.stdout+shell_result.stderr
        if shell_result.returncode!=0:
            logger.warning("shell command returned non-zero")
        current_context.append(
            {"role": "user", "content": SHELL_OUTPUT_PREFIX + shell_output}
        )
    else:
        global_state["should_ask_input"] = True
        print(current_output)


# === main section ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s:%(message)s")
current_context = [{"role": "system", "content": construct_system_prompt()}]
global_state["should_ask_input"] = True
while True:
    try:
        mainloop()
    except KeyboardInterrupt:
        global_state["should_ask_input"] = True
