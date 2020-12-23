import json
import os

PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "libdata")
ANS_CONVERT = json.load(open(os.path.join(PATH, "convert_answers.json")))


def process_answer_default(ans):
    if len(ans) == 0:
        return ""
    ans = ans.lower()
    if ans[-1] == ".":
        ans = ans[:-1].strip()
    if ans.startswith("a "):
        ans = ans[2:].strip()
    if ans.startswith("an "):
        ans = ans[3:].strip()
    if ans.startswith("the "):
        ans = ans[4:].strip()
    if ans in ANS_CONVERT:
        ans = ANS_CONVERT[ans]
    return ans
