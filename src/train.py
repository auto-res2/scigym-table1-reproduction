"""1 インスタンスを公式 Controller で解かせる。公式の履歴はクラス属性なので 1 件 1 プロセス。"""

import json
import os
import sys
from pathlib import Path

from openai import OpenAI
from scigym.api import LLM
from scigym.controller import Controller
from scigym.data import SBML
from scigym.eval.utils import extract_reaction_hashes


class OpenAICompatible(LLM):
    """公式の scigym.agent.GPT と同じ手順で、OpenAI 互換エンドポイントを呼ぶ。"""

    def initialize(self, base_url):
        self.client = OpenAI(
            api_key=os.environ["VERCEL_AI_GATEWAY_API_KEY"], base_url=base_url, max_retries=5
        )
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_messages(self):
        return self.messages

    def get_response(self, user_message):
        self.add_message("user", user_message)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            max_tokens=self.max_length,
            temperature=self.temperature,
        )
        text = response.choices[0].message.content
        assert isinstance(text, str) and len(text) > 0, "empty response"
        self.add_message("assistant", text)
        usage = response.usage
        self.input_total_tokens += usage.prompt_tokens if usage else 0
        self.output_total_tokens += usage.completion_tokens if usage else 0
        return text, {}


def main():
    cfg = json.loads(sys.argv[1])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    controller = Controller(
        path_to_sbml_cfg=cfg["instance_dir"],
        max_iterations=cfg["max_iterations"],
        test_memorize=False,
        output_directory=str(out),
        experiment_actions_path="prompts/experiment_actions_perturb.md",
        customized_functions_path="prompts/customized_functions_sim.md",
        eval_debug_rounds=cfg["eval_debug_rounds"],
        temperature=cfg["temperature"],
    )
    llm = OpenAICompatible(
        model_name=cfg["model"],
        api_key="",
        system_prompt=controller._create_system_prompt(),
        temperature=cfg["temperature"],
        max_length=cfg["max_tokens"],
        base_url=cfg["base_url"],
    )
    controller.run_benchmark(model=llm)
    # 反応一致を airas-eval の 2 値分類で照合するための集合（modifier なし）
    pred = SBML(controller.final_sbml) if controller.final_evaluation else controller.incomplete_model
    _, pred_rp = extract_reaction_hashes(pred.model)
    (out / "reactions.json").write_text(json.dumps({
        "missing": sorted(controller.evaluator.missing_rp_hashes),
        "added": sorted(pred_rp - controller.evaluator.inco_rp_hashes),
        "input_tokens": llm.input_total_tokens,
        "output_tokens": llm.output_total_tokens,
    }))


if __name__ == "__main__":
    main()
