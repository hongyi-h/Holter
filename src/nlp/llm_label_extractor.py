"""LLM-based label extraction from Chinese Holter ECG report conclusions.

Uses a lightweight Chinese medical LLM for structured classification,
providing higher accuracy than regex for ambiguous or complex reports.
"""
import json
import torch
from typing import Dict, List, Optional

from src.nlp.report_parser import LABEL_NAMES

# Chinese descriptions for each label (used in the prompt)
LABEL_DESC_ZH = {
    "sinus_rhythm": "窦性心律",
    "sinus_tachycardia": "窦性心动过速",
    "sinus_bradycardia": "窦性心动过缓",
    "atrial_premature": "房性早搏/房性期前收缩",
    "ventricular_premature": "室性早搏/室性期前收缩",
    "atrial_fibrillation": "心房颤动/房颤",
    "atrial_flutter": "心房扑动/房扑",
    "svt": "室上性心动过速",
    "ventricular_tachycardia": "室性心动过速/室速",
    "av_block": "房室传导阻滞",
    "st_change": "ST段改变/ST-T改变",
    "long_qt": "QT间期延长",
    "pause": "停搏/长间歇",
}


def build_prompt(conclusion_text: str) -> str:
    """Build a structured prompt for the LLM."""
    label_list = "\n".join(
        f"  {i+1}. {name} ({LABEL_DESC_ZH[name]})"
        for i, name in enumerate(LABEL_NAMES)
    )
    return (
        '你是一位专业的心电图报告分析AI。请根据以下Holter心电图报告结论，'
        '判断每个诊断标签是否存在（1=存在，0=不存在）。\n\n'
        '**重要规则**：\n'
        '- 如果报告中出现"未见"、"未发现"、"无"、"未及"、\n'
        '  "无明显"、"排除"等否定词修饰某个诊断，则该标签应标注为0。\n'
        '- 例如"未见缺血型ST-T改变"→ st_change=0；\n'
        '        "未见房室传导阻滞"→ av_block=0。\n'
        '- 只有报告明确肯定存在该诊断时才标注为1。\n\n'
        f'报告结论：\n{conclusion_text}\n\n'
        f'诊断标签列表：\n{label_list}\n\n'
        '请以JSON格式输出结果，只包含标签名和0/1值，例如：\n'
        '{"sinus_rhythm": 1, "atrial_premature": 0, ...}\n\n'
        'JSON输出：'
    )


class LLMLabelExtractor:
    """Extract labels using a lightweight LLM (e.g., Qwen2.5-1.5B-Instruct)."""

    DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 256,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (
            torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16 if self.device == "cuda"
            else torch.float32
        )
        self.max_new_tokens = max_new_tokens

        print(f"Loading LLM: {self.model_name} on {self.device} ({self.torch_dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.torch_dtype,
            device_map=self.device, trust_remote_code=True,
        )
        self.model.eval()
        print("LLM loaded successfully")

    @torch.no_grad()
    def extract(self, conclusion_text: str) -> Dict[str, int]:
        """Extract binary labels from a single conclusion text."""
        if not conclusion_text or not conclusion_text.strip():
            return {name: 0 for name in LABEL_NAMES}

        prompt = build_prompt(conclusion_text)

        # Use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )

        # Decode only the generated part
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return self._parse_response(response)

    def extract_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, int]]:
        """Extract labels for a batch of texts."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                results.append(self.extract(text))
        return results

    @staticmethod
    def _parse_response(response: str) -> Dict[str, int]:
        """Parse LLM response into label dict, with fallback."""
        # Try to find JSON in the response
        # Look for { ... } pattern
        import re
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                # Validate and normalize
                labels = {}
                for name in LABEL_NAMES:
                    val = parsed.get(name, 0)
                    labels[name] = 1 if val in (1, True, "1", "true", "True") else 0
                return labels
            except json.JSONDecodeError:
                pass

        # Fallback: all zeros
        print(f"  WARNING: Could not parse LLM response, falling back to all-zero")
        return {name: 0 for name in LABEL_NAMES}
