import enum
from src.behavioral_engine.moe_score_subgraph.judge_prompts import (
    disinformation_deception,
    illegal_goods_services,
    hate_harassment_discrimination,
    non_violent_crime,
    violence,
    sexually_explicit_content,
)


class EvalType(enum.Enum):
    DISINFORMATION_DECEPTION = "disinformation_deception"
    ILLEGAL_GOODS_SERVICES = "illegal_goods_services"
    HATE_HARASSMENT_DISCRIMINATION = "hate_harassment_discrimination"
    NON_VIOLENT_CRIME = "non_violent_crime"
    VIOLENCE = "violence"
    SEXUALLY_EXPLICIT_CONTENT = "sexually_explicit_content"


# Map eval types to their prompts
eval_type_to_prompt = {
    EvalType.DISINFORMATION_DECEPTION: disinformation_deception.JUDGE_PROMPT,
    EvalType.ILLEGAL_GOODS_SERVICES: illegal_goods_services.JUDGE_PROMPT,
    EvalType.HATE_HARASSMENT_DISCRIMINATION: hate_harassment_discrimination.JUDGE_PROMPT,
    EvalType.NON_VIOLENT_CRIME: non_violent_crime.JUDGE_PROMPT,
    EvalType.VIOLENCE: violence.JUDGE_PROMPT,
    EvalType.SEXUALLY_EXPLICIT_CONTENT: sexually_explicit_content.JUDGE_PROMPT,
}




