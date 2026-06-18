import os
import random
import re
from mathruler.grader import extract_boxed_content, grade_answer
import traceback

def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    pattern = r'^<think>(.*?)</think>(.+)$'  # Note the .+ for non-empty after content
    match = re.match(pattern, solution_str, re.DOTALL)

    if not match:
        return None

    inside_content = match.group(1)
    after_content = match.group(2)

    if not after_content.strip():
        return None

    if not inside_content.strip():
        return None

    if '<think>' in inside_content or '</think>' in inside_content:
        return None

    if '<think>' in after_content or '</think>' in after_content:
        return None

    return after_content

def pred_ans_extract(pred_response):
    if "<answer>" in pred_response:
        ans_1 = pred_response.split('<answer>')[-1]
        ans = ans_1.split("</answer>")[0]
    elif '</thinking>' in pred_response:
        ans = pred_response.split('</thinking>')[-1]
    else:
        if '</think>' in pred_response:
            ans = pred_response.split('</think>')[-1]
        else:
            ans = pred_response

    if '<!-- Final Answer -->' in ans:
        ans = ans.split('<!-- Final Answer -->')[1].strip()
        ans = ans.split('<!-- Finish Answer -->')[0].strip()
    elif '<!-- Summarize Answer -->' in ans:
        ans = ans.split('<!-- Summarize Answer -->')[-1].strip()
        ans = ans.split('<!-- Finish Response -->')[0].strip()
    else:
        if "最终答案" in ans:
            ans = ans.split(r"最终答案")[-1].replace("：", "").replace("**", "").strip()
        elif "**答案**" in ans:
            ans = ans.split(r"**答案**")[-1].replace("：", "").replace("**", "").strip()
        elif "Final Answer" in ans:
            ans = ans.split(r"Final Answer")[-1].replace(":", "").replace("**", "").strip()
        elif "答案：" in ans:
            ans = ans.split(r"答案：")[-1].replace("：", "").replace("**", "").strip()
        # elif "答案" in ans:
        #     ans = ans.split(r"答案")[-1].replace("：", "").replace("**", "").strip()
        elif "**Answer**" in ans:
            ans = ans.split(r"**Answer**")[-1].replace(":", "").replace("**", "").strip()
        elif "Answer:" in ans:
            ans = ans.split(r"Answer:")[-1].replace(":", "").replace("**", "").strip()
        # elif "Answer" in ans:
        #     ans = ans.split(r"Answer")[-1].replace(":", "").replace("**", "").strip()
        elif "总结" in ans:
            ans = ans.split(r"总结")[-1].replace("：", "").replace("**", "").strip()
        elif "Conclusion" in ans:
            ans = ans.split(r"Conclusion")[-1].replace(":", "").replace("**", "").strip()
        else:
            ans = "\n\n".join(ans.split("\n\n")[-10:])
            ans = ans[-300:].strip()
    if ans != None:
        ans = ans.replace("<|end_of_solution|>", "")
        ans = ans.replace("<|begin_of_solution|>", "")
        ans = ans.replace("<|end_of_thought|>", "")
        ans = ans.replace("<|begin_of_thought|>", "")

    if ans == None:
        ans = pred_response[-300:]
    return ans

def pred_ans_extract_train(pred_response):

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, pred_response, re.S)
    if match:
        has_answer = True
        content = match.group(1)
    else:
        has_answer = False
        content = None

    return content

def is_valid_think_format(text):
    # Check basic pattern first
    pattern = r'^<think>(.*?)</think>(.+)$'  # Note the .+ for non-empty after content
    match = re.match(pattern, text, re.DOTALL)

    if not match:
        return False

    # Extract the content inside and after the think tags
    inside_content = match.group(1)
    after_content = match.group(2)

    # Verify inside content is not empty
    if not inside_content.strip():
        return False

    # Verify neither part contains additional think tags
    if '<think>' in inside_content or '</think>' in inside_content:
        return False

    if '<think>' in after_content or '</think>' in after_content:
        return False

    return True

def is_valid_DS_format(text):

    if "</think>" in text:
        return True
    else:
        return False

def is_valid_Boxed_format(text):

    if "boxed" in text:
        return True
    else:
        return False

def compute_score_format(solution_str):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    if solution_str is None:
        return 0.0
    
    try:
        if "<think>" in solution_str and "<answer>" in solution_str:
            assistant_blocks = re.findall(r'<think>(.*?)</think>\n<answer>(.*?)</answer>', solution_str, re.DOTALL)
            if len(assistant_blocks) == 0:
                return 0.0
            else:
                return 1.0

        format_reward = 0.0
        last_assistant_block = solution_str
        think_answer_match = is_valid_think_format(last_assistant_block) or is_valid_DS_format(last_assistant_block) or is_valid_Boxed_format(last_assistant_block)
        if think_answer_match:
            format_reward = 1.0
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")
        return 0.0
    
    return format_reward


def math_verify_reward_function(solution_str, ground_truth):
    from math_verify import parse, verify
    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth

    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return 0.0

    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return 0.0

    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return 1.0

    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                    parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                    math_verify_parsed,
                    timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue

    # Very unlikely to be correct after the above matches
    return 0.0

def compute_score_mathverify_judge(solution_str: str, ground_truth: str):
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[1]
    else:
        # print("No </think> tag found")
        solution_str = solution_str

    try:
        return math_verify_reward_function(solution_str, ground_truth)
    except:
        traceback.print_exc(10)
        return False

def compute_score_answer(solution_str, ground_truth, is_train=False) -> float:

    if is_train == False:
        answer = pred_ans_extract(solution_str)
    else:
        answer = pred_ans_extract_train(solution_str)
        if answer == None:
            return 0.0

    verify_0 = compute_score_mathverify_judge(answer, ground_truth)
    answer = extract_boxed_content(answer)

    verify_2 = compute_score_mathverify_judge(answer, ground_truth)
    verify_1 = 1.0 if grade_answer(answer, ground_truth) else 0.0

    if verify_0 == 1.0 or verify_1 == 1.0 or verify_2 == 1.0:
        return 1.0
    else:
        return 0.0






def compute_score_format_answer(solution_str, ground_truth, is_train=False):
    """The scoring function for format reward.

    Args:
        solution_str: the solution text
    
    """
    if solution_str is None or ground_truth is None:
        return 0.0

    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth, is_train=is_train)

        if format_reward == 1.0:
            return answer_reward
        else:
            return -1.0
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return -1.0




def stare_compute_score(solution_str, ground_truth, is_train=False):
    score = compute_score_format_answer(solution_str, ground_truth, is_train=is_train)
    acc = compute_score_answer(solution_str, ground_truth, is_train=is_train)
    format = compute_score_format(solution_str)
    pred_text = pred_ans_extract(solution_str)
    pred_ans = extract_boxed_content(pred_text)
    if pred_ans is None:
        pred_ans = pred_text
    return {
        "score": float(score),
        "acc": float(acc),
        "format": float(format),
        "pred": pred_ans
    }