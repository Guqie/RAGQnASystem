"""咨询师回答所用的提示词，以及没有大模型时的整理式回复。"""

COUNSEL_ONLY_KNOWLEDGE = (
    "本轮规划为直接咨询，没有调用知识图谱。"
    "请围绕用户的情绪和叙述做支持，不要主动堆砌疾病资料，也不要编造药名。"
)

EMPTY_KNOWLEDGE = (
    "知识库中没有找到与当前问题直接对应的医学条目。"
    "请只做心理支持，不要编造疾病结论、药名或剂量。"
)

UNAVAILABLE_KNOWLEDGE = (
    "知识图谱当前没有连上。请说明资料暂时查不到，然后做心理支持，不要编造医学结论。"
)

SYSTEM_PROMPT = """你是一名专业、有耐心的心理咨询师，同时会参考医疗知识图谱。
请用自然的中文回复，并按这个顺序组织，但不要把序号念出来：
1. 先接住用户的情绪，确认他正在经历的感受。
2. 帮他把问题说清楚，必要时指出信息还不够。
3. 如果给了知识图谱资料，用咨询的口气转述，不要罗列成说明书。资料没写的药名、剂量、疗程不要补充。
4. 给出温和、可执行的应对方向，并提醒这不是诊断，不能代替医生面诊。
5. 鼓励对方继续说，留下一个具体的后续问题。

用药信息只转述资料中出现的名称，并说明需要在医生指导下使用。
图谱里的“症状共现”只能说“可能相关”，不能说用户已经被诊断为某种疾病。"""

CRISIS_CAUTION = """
另外，最近几轮对话里出现过危机表述。回复开头先确认对方现在是否安全、身边有没有人，再非常谨慎地谈及其他问题。不要提供剂量，不要把检索结果说成治疗方案。"""


def build_messages(query: str, memory, knowledge: str) -> list:
    system = SYSTEM_PROMPT
    if memory.crisis_turns_left:
        system += CRISIS_CAUTION
    user = f"""对话记忆：
{memory.prompt_block()}

用户当前的话：
{query}

知识图谱检索结果：
{knowledge}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def offline_answer(query: str, memory, knowledge: str, note: str | None = None) -> str:
    parts = ["我在听。你愿意把这些说出来，本身就很重要。"]
    if memory.last_disease and memory.turns:
        parts.append(f"我们前面一直在聊「{'、'.join(memory.focus_diseases)}」。")
    if memory.crisis_turns_left:
        parts.append("我更关心你现在是否安全。如果危险还在，请先联系身边的人或拨打 12356。")
    if knowledge and knowledge not in {COUNSEL_ONLY_KNOWLEDGE, EMPTY_KNOWLEDGE, UNAVAILABLE_KNOWLEDGE}:
        parts.append("我查了知识图谱。下面是可以参考的资料，它不是诊断：")
        parts.append(knowledge)
    elif knowledge == UNAVAILABLE_KNOWLEDGE:
        parts.append("知识图谱这会儿没有连上，我没法引用具体的疾病条目。")
    else:
        parts.append("这一轮我没有检索到可以直接引用的图谱条目，先从支持的角度陪你把事情说清楚。")
    parts.append("这些内容不能代替医生面诊。如果你愿意，可以告诉我这种感受持续了多久，以及对睡眠、学习或工作的影响。")
    parts.append(note or "（当前未配置大模型，以上是根据检索结果整理的回复。）")
    return "\n\n".join(parts)
