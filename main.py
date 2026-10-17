import argparse
import base64
import json
import logging
import os
import re
import readline  # noqa: F401
import subprocess
import sys
from typing import Any
import requests
from dataclasses import dataclass


def init_context() -> list[dict[str, str]]:
    sysprompt = ""
    for fname in config["agent"]["sysprompt_files"]:
        with open(fname) as f:
            contents = f.read()
            sysprompt = sysprompt + f"\n# {fname}\n" + contents
    init_messages = [
        {"role": "system", "content": sysprompt},
        {
            "role": "user",
            "content": '\n[SYSTEM] This is the "session header" section. By default this is intended to store summaries of truncated contexts.\n[EMPTY]',
        },
    ]
    return init_messages


def record_usage(usage_data: dict):
    tstat = cur_state.token_stats
    cache = usage_data["prompt_tokens_details"]["cached_tokens"]
    tstat["cache"] += cache
    tstat["input"] += usage_data["prompt_tokens"] - cache
    tstat["output"] += usage_data["completion_tokens"]
    tstat["total"] = tstat["input"] + tstat["cache"] + tstat["output"]
    tstat["last_context"] = usage_data["prompt_tokens"]


def generate(context: list[dict], config: dict) -> str:
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


def exec_shell(cmd: str):
    shell_result = subprocess.run(
        cmd, shell=True, capture_output=True, check=False, text=True
    )
    shell_output = shell_result.stdout + shell_result.stderr
    if shell_output == "":
        # letting the agent know the output is empty,
        # rather than leaving an ambiguous `[SHELL]` in the context
        shell_output = "(no output)"
    if len(shell_output) > SHELL_RESULT_TRUNC:
        shell_output = shell_output[:SHELL_RESULT_TRUNC] + "...(too long, truncated)"
        logger.info("shell command returned very long output(truncated): " + cmd)
    elif len(shell_output) > SHELL_RESULT_TRUNC / 4:
        logger.info("shell command returned very long output(not truncated): " + cmd)
    if shell_result.returncode != 0:
        logger.warning("shell command returned non-zero")
    logger.debug("shell output:\n" + shell_output)
    return shell_output


def process_model_output(output: str):
    # after the model repeatedly mix the shell command
    # and freeform text, (particularly on non-cot)
    # pairing [/RUN] is added to address this.
    mo = re.search(r"(?s)\[RUN\](.*?)\[/RUN\]", output)
    if mo:
        msg_head, cmd, msg_tail = output[: mo.start()], mo[1], output[mo.end() :]
        if msg_head:
            print(msg_head)
        logger.info("shell: " + cmd)
        if msg_tail:
            print(msg_tail)
        # NOTE using check=true will raise exception on error, making shell_result_obj invalid.
        # but the agent needs to see the error message, so check=false makes this more intuitive.
        if ALLOW_SHELL:
            shell_output = exec_shell(cmd)
        else:
            shell_output = "sorry, shell is not permitted in this environment."
        cur_state.context.append({"role": "user", "content": "[SHELL]" + shell_output})
    else:
        cur_state.should_generate = False
        out_str = "\n" + output + "\n"
        print(out_str)


def trunc_context(mode="keep_last_n", **kw):
    # a very crude impl to truncate the context, but any act of modifying the context should require caution.
    if mode == "keep_last_n":
        n_keep = kw.get("n_keep", 10)
        if n_keep <= 1:
            logger.warning(
                "invalid value for truncation. at least 2 items should be kept (1 user + 1 assistant)"
            )
            return
        if n_keep % 2 != 0:
            logger.warning("n_keep for keep_last_n truncation is not even.")
        system_msg = [msg for msg in cur_state.context if msg["role"] == "system"]
        keep_msg = cur_state.context[-n_keep:]
        notice_msg = [
            {"role": "user", "content": "[SYSTEM] earlier context has been discarded."}
        ]
        stash_msg = [cur_state.context[cur_state.idx_stash]]
        cur_state.context = system_msg + stash_msg + notice_msg + keep_msg
        t = str(keep_msg[0]["content"])
        if len(t) > 100:
            t = t[:100] + "..."
        logger.info("context truncated to message:\n" + t)
    elif mode == "trunc_shell":
        for msg in cur_state.context:
            if msg["content"].startswith("[SHELL]"):
                msg["content"] = "[SHELL] <truncated>"
                logger.info("all shell outputs removed.")
    else:
        logger.warning("invalid mode for trunc command: " + mode)


def handle_user_command(cmdtext: str):
    if not cmdtext.strip():
        logger.warning("empty command. doing nothing.")
        return
    cmd, *args = cmdtext.split()
    match cmd:
        case "load":
            # load from the direct output of context,overwriting whatever current context.
            fname = "session-log.json" if len(args) == 0 else args[0]
            with open(fname, "r") as f:
                cur_state.context = json.load(f)
        case "save":
            fname = "session-log.json" if len(args) == 0 else args[0]
            with open(fname, "w") as f:
                json.dump(cur_state.context, f, ensure_ascii=False, indent=4)
        case "exit":
            sys.exit(0)
        case "token":
            print(cur_state.token_stats)
        case "mline":
            # multiline input
            print("please input heredoc-like text, end with EOF")
            user_input_mline = ""
            line = input()
            while line != "EOF":
                user_input_mline += line + "\n"
                line = input()
            cur_state.context.append({"role": "user", "content": user_input_mline})
            cur_state.should_generate = True
        case "trunc":
            if len(args) == 0:
                logger.warning("cannot parse.")
                return
            if args[0] == "shell":
                trunc_context("trunc_shell")
            else:
                if args[0].isdigit():
                    trunc_context(n_keep=int(args[0]))
                else:
                    logger.warning("cannot parse.")
                    return
        case "image":
            # in doc examples the message included a text query accompanying the image,
            # but possibly not need
            # msg="[image]"
            # also, this action is supposed to be "included an external data to chat",
            # so "should generate" is not set to true after.
            fname = "image.png" if len(args) == 0 else args[0]
            with open(fname, "rb") as f:
                base64_str = base64.b64encode(f.read()).decode("utf-8")
            cur_state.context.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_str}"},
                        }
                    ],
                }
            )
        case "stash":
            tmp = cur_state.context[cur_state.idx_stash]["content"]
            # tmp used for clarity, but note it does not hold a reference due to how python works
            if tmp.endswith("[EMPTY]"):
                cur_state.context[cur_state.idx_stash]["content"] = tmp.removesuffix(
                    "[EMPTY]"
                )
            stash_str = f"\n{cur_state.context[-1]['role']}:{cur_state.context[-1]['content']}\n"
            cur_state.context[cur_state.idx_stash]["content"] += stash_str
            logger.info(
                "stashed last message. note changing the stash may invalidate input cache. use with trunc."
            )

        case _:
            if cmd in prompt_macro:
                cur_state.context.append({"role": "user", "content": prompt_macro[cmd]})
                cur_state.should_generate = True
            else:
                logger.info("command not found:" + cmd)


def mainloop():
    if not cur_state.should_generate:
        user_input = input("> ")
        if user_input.startswith("/"):
            handle_user_command(user_input.removeprefix("/"))
        elif not user_input.strip():
            logger.warning("input is empty. doing nothing")
        else:
            cur_state.context.append({"role": "user", "content": user_input})
            cur_state.should_generate = True
    if cur_state.should_generate:
        model_output = generate(cur_state.context, config["api"])
        cur_state.context.append({"role": "assistant", "content": model_output})
        process_model_output(model_output)


# finally, a dataclass to prevent a portion of possible errors.
# note that idx_stash points to the index of the stash section
# in the context and should be used to access it, normally its just 1 (the 2nd item)
# right after system prompt, but again, caution is a must when handling context,
# which is and will remain one of the most important variables in this code.
# also, the stash may be used as a generic informative session header.
@dataclass
class CurrentState:
    context: list[dict[str, str | Any]]
    token_stats: dict[str, int]
    idx_stash: int = 1
    should_generate: bool = False


if __name__ == "__main__":
    # setting up
    LOG_LEVEL = logging.INFO
    # note for latin text (most shell) where roughly 4 char=1 token,
    # but for cjk text about 1.5 char=1 token
    SHELL_RESULT_TRUNC = 20000
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    config_path = "config.json"
    if args.config is not None:
        config_path = args.config
    with open(config_path) as f:
        config = json.load(f)
    if "prompt_macro_path" in config["agent"]:
        with open(config["agent"]["prompt_macro_path"]) as f:
            prompt_macro = json.load(f)
    else:
        prompt_macro = {}
    if args.verbose:
        LOG_LEVEL = logging.DEBUG
    ALLOW_SHELL = True
    if "allow_shell" in config["agent"] and config["agent"]["allow_shell"] is False:
        ALLOW_SHELL = False

    # the core idea is, the system prompt is, concating a series of text files...
    # but that was the idea that the prompt should be tuned by the ai itself as appropriate,
    # however it is unnecessary if the prompt is maintained manually...
    # using md for convention, but i don't expect ##s and **s in the prompt itself.
    init_tstats = {
        "input": 0,
        "cache": 0,
        "output": 0,
        "total": 0,
        "last_context": 0,
    }
    cur_state = CurrentState(context=init_context(), token_stats=init_tstats)
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s:%(message)s")
    if "API_KEY" in os.environ:
        apikey = os.environ["API_KEY"]
    else:
        logger.warning("No API Key Found.")
    while True:
        try:
            mainloop()
        except KeyboardInterrupt:
            cur_state.should_generate = False
        except requests.HTTPError as e:
            if len(e.response.text) < 100:
                rtext = e.response.text
            else:
                rtext = e.response.text[:100] + "..."
            cur_state.should_generate = False
            logger.warning(
                f"received http code {e.response.status_code},"
                + f"the response reads:\n{rtext}"
            )
        except requests.ConnectionError as e:
            cur_state.should_generate = False
            logger.warning(e)
        except FileNotFoundError as e:
            cur_state.should_generate = False
            logger.warning(e)
