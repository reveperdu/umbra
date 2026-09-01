import os
import readline  # noqa: F401
import subprocess

import requests

API_URL="https://api.deepseek.com/chat/completions"
DIRECT_OUTPUT_PREFIX="[TO_USER]"
SHELL_OUTPUT_PREFIX="[SHELL]"
config_api = {
    "model": "deepseek-v4-flash",
    "thinking": {"type": "enabled"},
    "reasoning_effort": "high"
}
global_state={"should_ask_input":True}
#the core idea is, the system prompt is, concating a series of text files.
#using md for convention, but i don't expect ##s and **s in the prompt itself.
config_agent={
    "sysprompt_files":["test.md"]
}
if "API_KEY" in os.environ:
    apikey = os.environ["API_KEY"]
else:
    raise RuntimeError("No API Key Found.")

def construct_system_prompt():
    sysprompt=""
    for fname in config_agent["sysprompt_files"]:
        with open(fname) as f:
            contents=f.read()
            sysprompt=sysprompt+f"\n# {fname}\n"+contents
    return sysprompt
def generate(context: list[dict[str, str]], config: dict[str, str]):
    header = {
        "Authorization": "Bearer " + apikey,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    data={"messages":context}|config
    resp=requests.post(API_URL,headers=header,json=data)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def mainloop():
    if global_state["should_ask_input"]:
        user_input=input("> ")
        global_state["should_ask_input"]=False
        current_context.append({"role":"user","content":user_input})
    current_output=generate(current_context,config_api)
    current_context.append({"role":"assistant","content":current_output})
    if current_output.startswith(DIRECT_OUTPUT_PREFIX):
        global_state["should_ask_input"]=True
        print(current_output.removeprefix(DIRECT_OUTPUT_PREFIX))
    else:
        shell_result_obj=subprocess.run(current_output,shell=True,capture_output=True,check=True)
        shell_output=shell_result_obj.stdout.decode("utf-8")
        current_context.append({"role":"user","content":SHELL_OUTPUT_PREFIX+shell_output})

# === main section ===
current_context=[{"role":"system","content":construct_system_prompt()}]
should_ask_input=True
while True:
    try:
        mainloop()
    except KeyboardInterrupt:
        global_state["should_ask_input"]=True
