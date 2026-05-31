from flask import Flask, render_template, request, jsonify
from agents.teaching_agents import run_single_question_diagnosis, run_student_report

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    """
    v0.1 单题诊断接口，保留备用。
    """
    try:
        data = request.get_json()

        question = data.get("question", "").strip()
        standard_answer = data.get("standard_answer", "").strip()
        student_answer = data.get("student_answer", "").strip()
        knowledge_point = data.get("knowledge_point", "").strip()

        if not question or not standard_answer or not student_answer or not knowledge_point:
            return jsonify({
                "success": False,
                "message": "请完整填写题目、标准答案、学生答案和知识点。"
            }), 400

        result = run_single_question_diagnosis(
            question=question,
            standard_answer=standard_answer,
            student_answer=student_answer,
            knowledge_point=knowledge_point
        )

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"系统出错：{str(e)}"
        }), 500


@app.route("/api/student_report", methods=["POST"])
def student_report():
    """
    v0.2 单个学生多题诊断报告接口。
    """
    try:
        data = request.get_json()

        student_name = data.get("student_name", "").strip()
        grade = data.get("grade", "").strip()
        subject = data.get("subject", "").strip()
        chapter = data.get("chapter", "").strip()
        questions = data.get("questions", [])

        if not student_name:
            return jsonify({
                "success": False,
                "message": "请填写学生姓名。"
            }), 400

        if not subject:
            return jsonify({
                "success": False,
                "message": "请填写学科。"
            }), 400

        if not chapter:
            return jsonify({
                "success": False,
                "message": "请填写章节或知识范围。"
            }), 400

        if not questions or not isinstance(questions, list):
            return jsonify({
                "success": False,
                "message": "请至少填写一道题。"
            }), 400

        cleaned_questions = []
        for index, item in enumerate(questions, start=1):
            question_text = item.get("question_text", "").strip()
            knowledge_point = item.get("knowledge_point", "").strip()
            standard_answer = item.get("standard_answer", "").strip()
            student_answer = item.get("student_answer", "").strip()

            if not question_text or not knowledge_point or not standard_answer or not student_answer:
                return jsonify({
                    "success": False,
                    "message": f"第 {index} 题信息不完整，请填写题目、知识点、标准答案和学生答案。"
                }), 400

            cleaned_questions.append({
                "question_number": index,
                "question_text": question_text,
                "knowledge_point": knowledge_point,
                "standard_answer": standard_answer,
                "student_answer": student_answer
            })

        result = run_student_report(
            student_name=student_name,
            grade=grade,
            subject=subject,
            chapter=chapter,
            questions=cleaned_questions
        )

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"系统出错：{str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)