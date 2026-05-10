from __future__ import annotations

from pathlib import Path


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
SKILLS_DIR = Path(__file__).parent.parent / "skills"


def load_prompt(relative_path: str) -> str:
    """
    读取 prompts/ 目录下的 prompt 模板文件。
    relative_path 示例：'audio/analyzer_r1_system.md'
    """
    path = PROMPTS_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Prompt 文件不存在：{path}")
    return path.read_text(encoding="utf-8")


def load_skill(skill_name: str) -> str:
    """
    读取 skills/ 目录下的 skill 文件。
    skill_name 示例：'narrative_skill'（不需要带 .md 后缀）
    """
    path = SKILLS_DIR / f"{skill_name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Skill 文件不存在：{path}")
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, **kwargs) -> str:
    """
    用正则替换 {变量名} 占位符，只替换 kwargs 中存在的变量名。
    JSON 示例里的花括号不会被误处理。
    变量名规则：只含字母、数字、下划线。
    """
    import re
    def replacer(m):
        key = m.group(1)
        if key in kwargs:
            return str(kwargs[key])
        return m.group(0)  # 不在 kwargs 里就原样保留
    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", replacer, template)


def load_and_render(relative_path: str, **kwargs) -> str:
    """
    读取并渲染 prompt 模板，一步完成。
    """
    template = load_prompt(relative_path)
    return render_prompt(template, **kwargs)