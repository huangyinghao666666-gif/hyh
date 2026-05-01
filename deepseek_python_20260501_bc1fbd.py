import os
import time
import argparse
import openai
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

class IndustryReportAgent:
    """产业链深度研报生成智能体"""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature

    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        """调用语言模型"""
        try:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM 调用异常] {e}")
            return ""

    def generate_outline(self, topic: str) -> list:
        """
        根据主题生成报告大纲（章节标题列表）
        """
        system = (
            "你是一位资深的行业研究分析师，擅长撰写产业链深度研究报告。"
            "请只输出干净的大纲，不要额外解释。"
        )
        user = f"""请为“{topic}”的产业链深度研究报告生成一个详细大纲。
要求：
1. 报告标题可以自由拟定。
2. 必须包含且不限于以下维度：产业链全景图、上游分析、中游分析、下游分析、竞争格局、重点企业、核心技术/技术壁垒、政策环境、市场规模与预测、风险与机遇、结论与建议。
3. 总共 8～12 个章节。
4. 输出格式：每行一个章节标题，以“数字. ”开头，例如：
1. 全球光伏产业链全景图
2. 上游：多晶硅与硅片供给分析
...
"""
        text = self._call_llm(system, user, max_tokens=600)
        # 解析章节标题
        sections = []
        for line in text.split("\n"):
            line = line.strip()
            # 匹配 "1. 标题" 或 "1、标题" 或 "1) 标题"
            if line and (line[0].isdigit() or line.startswith("- ")):
                # 去掉编号和可能的符号
                title = line.lstrip("0123456789. -、) ").strip()
                if title:
                    sections.append(title)
        # 如果解析失败，使用默认稳健大纲
        if len(sections) < 6:
            print("大纲自动解析不够完整，使用默认大纲。")
            sections = [
                "产业链全景图",
                "上游供应环节分析",
                "中游制造与核心技术",
                "下游应用与市场需求",
                "竞争格局与重点企业剖析",
                "政策与监管环境",
                "技术迭代与创新趋势",
                "市场规模与增长预测",
                "风险因素与挑战",
                "投资机遇与策略建议"
            ]
        return sections

    def generate_section(self, topic: str, section_title: str, report_so_far: str) -> str:
        """
        撰写单个章节，可参考已生成的全文以保证连贯。
        """
        system = (
            "你是一位资深产业链研究员，正在撰写一份专业、深度的产业链研报。"
            "要求：使用 Markdown 格式，章节标题用 ##；内容要有具体数据/案例/公司分析；"
            "保持客观、严谨、前瞻；不少于 500 字。"
        )
        # 只提取最后 800 字作为上下文，避免超过 token 限制
        context = report_so_far[-800:] if len(report_so_far) > 800 else report_so_far
        user = f"""研究主题：{topic}
当前章节：{section_title}
报告已写部分（供风格/内容衔接参考）：