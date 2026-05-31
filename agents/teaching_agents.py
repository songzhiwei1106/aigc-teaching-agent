import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()

# 清除 Python 请求时可能读取到的系统代理，避免 ProxyError
for proxy_key in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy"
]:
    os.environ.pop(proxy_key, None)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"


def call_deepseek(prompt, temperature=0.3):
    """
    调用 DeepSeek API。
    """

    if not DEEPSEEK_API_KEY:
        raise ValueError("没有找到 DEEPSEEK_API_KEY，请检查 .env 文件。")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业、严谨、适合中小学场景的AI教学诊断助手。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature
    }

    session = requests.Session()
    session.trust_env = False

    response = session.post(
        DEEPSEEK_API_URL,
        headers=headers,
        json=payload,
        timeout=90
    )

    if response.status_code != 200:
        raise RuntimeError(f"DeepSeek API 调用失败：{response.text}")

    result = response.json()
    return result["choices"][0]["message"]["content"]


def extract_json(text):
    """
    尽量从大模型返回内容中提取 JSON。
    """

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return {
        "raw_output": text
    }


def run_single_question_diagnosis(question, standard_answer, student_answer, knowledge_point):
    """
    v0.1：单题诊断，保留备用。
    """

    prompt = f"""
你是一个由多个教学智能体组成的系统，包括：
1. 批改智能体
2. 错因诊断智能体
3. 订正辅导智能体
4. 变式练习智能体

请对下面这一道题进行完整学习诊断。

题目：{question}
知识点：{knowledge_point}
标准答案：{standard_answer}
学生答案：{student_answer}

要求：
1. 只输出 JSON。
2. 不要输出 Markdown。
3. 不要输出多余解释。
4. 错误类型只能从以下五类选择：
   知识点漏洞、计算失误、审题不清、表达不规范、无明显错误。

请严格按照下面格式输出：

{{
  "grading_agent": {{
    "is_correct": false,
    "score": 0,
    "grading_result": "错误",
    "grading_reason": "说明为什么对或错"
  }},
  "diagnosis_agent": {{
    "error_type": "计算失误",
    "error_location": "错误位置",
    "diagnosis": "错因解释",
    "weak_knowledge_point": "薄弱知识点"
  }},
  "correction_agent": {{
    "correction_steps": [
      "第一步……",
      "第二步……",
      "第三步……"
    ],
    "key_reminder": "重点提醒",
    "student_friendly_explanation": "给学生看的简短解释"
  }},
  "exercise_agent": {{
    "exercises": [
      {{
        "difficulty": "基础",
        "question": "题目内容",
        "answer": "标准答案",
        "purpose": "设计意图"
      }},
      {{
        "difficulty": "巩固",
        "question": "题目内容",
        "answer": "标准答案",
        "purpose": "设计意图"
      }},
      {{
        "difficulty": "提升",
        "question": "题目内容",
        "answer": "标准答案",
        "purpose": "设计意图"
      }}
    ]
  }}
}}
"""

    output = call_deepseek(prompt, temperature=0.3)
    return extract_json(output)


def question_diagnosis_agent(student_name, grade, subject, chapter, questions):
    """
    逐题批改与错因诊断智能体。
    一次性处理多道题，减少 API 调用次数。
    """

    prompt = f"""
你是“逐题批改与错因诊断智能体”。

现在你需要批改一名学生的一整份作业，并对每一道错题进行诊断。

学生姓名：{student_name}
年级：{grade}
学科：{subject}
章节或知识范围：{chapter}

作业题目数据：
{json.dumps(questions, ensure_ascii=False, indent=2)}

请完成：
1. 判断每道题是否正确。
2. 给每道题打分，分数使用 0-100。
3. 判断每道题的错误类型。
4. 分析错误原因。
5. 指出薄弱知识点。
6. 给出该题的订正建议。

错误类型只能从以下五类中选择：
知识点漏洞、计算失误、审题不清、表达不规范、无明显错误。

要求：
1. 只输出 JSON。
2. 不要输出 Markdown。
3. 不要输出多余解释。
4. 每道题都必须返回结果。
5. 如果答案正确，error_type 写“无明显错误”。

请严格按照下面格式输出：

{{
  "question_results": [
    {{
      "question_number": 1,
      "is_correct": false,
      "score": 0,
      "grading_result": "错误",
      "grading_reason": "批改原因",
      "error_type": "计算失误",
      "error_location": "错误位置",
      "diagnosis": "错因分析",
      "weak_knowledge_point": "薄弱知识点",
      "correction_advice": "订正建议"
    }}
  ]
}}
"""

    output = call_deepseek(prompt, temperature=0.2)
    result = extract_json(output)

    if "question_results" not in result:
        result["question_results"] = []

    return result


def exercise_agent(student_name, grade, subject, chapter, question_results):
    """
    个性化变式练习智能体。
    根据学生薄弱知识点生成练习。
    """

    prompt = f"""
你是“个性化变式练习智能体”。

请根据学生的逐题诊断结果，为学生生成个性化变式练习。

学生姓名：{student_name}
年级：{grade}
学科：{subject}
章节或知识范围：{chapter}

逐题诊断结果：
{json.dumps(question_results, ensure_ascii=False, indent=2)}

要求：
1. 优先针对错误题目和薄弱知识点生成练习。
2. 生成 3 道变式练习题。
3. 难度从基础、巩固、提升递进。
4. 每道题都要有答案。
5. 每道题都要说明设计意图。
6. 只输出 JSON，不要输出 Markdown。

请严格按照下面格式输出：

{{
  "personalized_exercises": [
    {{
      "difficulty": "基础",
      "target_knowledge_point": "针对的知识点",
      "question": "题目内容",
      "answer": "标准答案",
      "purpose": "设计意图"
    }},
    {{
      "difficulty": "巩固",
      "target_knowledge_point": "针对的知识点",
      "question": "题目内容",
      "answer": "标准答案",
      "purpose": "设计意图"
    }},
    {{
      "difficulty": "提升",
      "target_knowledge_point": "针对的知识点",
      "question": "题目内容",
      "answer": "标准答案",
      "purpose": "设计意图"
    }}
  ]
}}
"""

    output = call_deepseek(prompt, temperature=0.6)
    result = extract_json(output)

    if "personalized_exercises" not in result:
        result["personalized_exercises"] = []

    return result


def student_report_agent(student_name, grade, subject, chapter, question_results, exercises):
    """
    个人学习诊断报告智能体。
    生成整体分析、薄弱点、学习建议。
    """

    prompt = f"""
你是“个人学习诊断报告智能体”。

请根据逐题诊断结果和个性化练习，为学生生成一份个人学习诊断报告。

学生姓名：{student_name}
年级：{grade}
学科：{subject}
章节或知识范围：{chapter}

逐题诊断结果：
{json.dumps(question_results, ensure_ascii=False, indent=2)}

个性化练习：
{json.dumps(exercises, ensure_ascii=False, indent=2)}

请输出：
1. 总体表现总结。
2. 正确率。
3. 主要错误类型。
4. 薄弱知识点列表。
5. 学习建议。
6. 给老师看的辅导建议。
7. 给学生看的鼓励性反馈。

要求：
1. 只输出 JSON。
2. 不要输出 Markdown。
3. 语言清晰，适合教学场景。
4. correct_rate 使用百分比字符串，例如 "66.7%"。

请严格按照下面格式输出：

{{
  "summary": {{
    "overall_comment": "总体表现总结",
    "correct_count": 2,
    "total_count": 3,
    "correct_rate": "66.7%",
    "main_error_types": ["计算失误", "审题不清"],
    "weak_knowledge_points": ["两位数乘一位数", "平均分应用题"]
  }},
  "learning_advice": [
    "学习建议1",
    "学习建议2",
    "学习建议3"
  ],
  "teacher_suggestion": [
    "老师辅导建议1",
    "老师辅导建议2"
  ],
  "student_feedback": "给学生看的鼓励性反馈"
}}
"""

    output = call_deepseek(prompt, temperature=0.4)
    result = extract_json(output)

    if "summary" not in result:
        result["summary"] = {}

    if "learning_advice" not in result:
        result["learning_advice"] = []

    if "teacher_suggestion" not in result:
        result["teacher_suggestion"] = []

    if "student_feedback" not in result:
        result["student_feedback"] = ""

    return result


def run_student_report(student_name, grade, subject, chapter, questions):
    """
    v0.2 总控智能体：
    协调逐题批改与错因诊断智能体、个性化练习智能体、个人报告智能体。
    """

    diagnosis_result = question_diagnosis_agent(
        student_name=student_name,
        grade=grade,
        subject=subject,
        chapter=chapter,
        questions=questions
    )

    question_results = diagnosis_result.get("question_results", [])

    exercise_result = exercise_agent(
        student_name=student_name,
        grade=grade,
        subject=subject,
        chapter=chapter,
        question_results=question_results
    )

    exercises = exercise_result.get("personalized_exercises", [])

    report_result = student_report_agent(
        student_name=student_name,
        grade=grade,
        subject=subject,
        chapter=chapter,
        question_results=question_results,
        exercises=exercises
    )

    return {
        "student_info": {
            "student_name": student_name,
            "grade": grade,
            "subject": subject,
            "chapter": chapter
        },
        "question_results": question_results,
        "personalized_exercises": exercises,
        "student_report": report_result
    }