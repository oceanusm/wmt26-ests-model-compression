from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "submissions" / "ests-gptoss-zho-k26"
sys.path.insert(0, str(PRIMARY))
SPEC = importlib.util.spec_from_file_location("ests_submission_inference", PRIMARY / "inference.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def result(text: str, *, finish_reason: str = "stop"):
    return MODULE.GenerationResult(
        model_hyp_text=text,
        raw_comp_text=text,
        prompt_token_count=10,
        completion_token_count=10,
        finish_reason=finish_reason,
    )


class CallbackGenerator:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def generate(self, prompts, *, seed, max_tokens):
        self.calls.append((list(prompts), seed, max_tokens))
        return [result(self.callback(prompt, seed)) for prompt in prompts]


class ESTSSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.zho = MODULE.load_submission_config(PRIMARY / "submission.json")

    def test_category_inference(self):
        self.assertEqual(MODULE.infer_category('{"title":"hello"}'), "json")
        self.assertEqual(MODULE.infer_category("```json\n{\"x\": \"y\"}\n```"), "json")
        self.assertEqual(MODULE.infer_category("<p>Hello<br>world</p>"), "social")
        self.assertEqual(MODULE.infer_category("<p>Look at #translation</p>"), "social")
        self.assertEqual(
            MODULE.infer_category("<p>A long personal reflection written as an ordinary paragraph without a headline or a final headline marker.</p>"),
            "social",
        )
        self.assertEqual(
            MODULE.infer_category("<p>Central Bank Announces Interest Rate Decision</p><p>Officials released the decision on Friday.</p>"),
            "news",
        )
        self.assertEqual(MODULE.infer_category("A plain spoken transcript."), "speech")

    def test_exact_prompt_templates(self):
        templates = MODULE.instruction_templates(self.zho)
        self.assertTrue(templates["news"].startswith("You are a professional Chinese, Simplified translator"))
        self.assertIn("The original text is a news article.", templates["news"])
        self.assertIn("user-generated content from a social media platform", templates["social"])
        self.assertIn("automatically transcribed from spoken language", templates["speech"])
        self.assertTrue(templates["news"].endswith("(zho_Hans):"))

    def test_checkpoint_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "pruned_gpt_oss",
                        "num_hidden_layers": 24,
                        "num_local_experts_per_layer": [6] * 24,
                        "num_experts_per_tok": 4,
                    }
                ),
                encoding="utf-8",
            )
            value = MODULE.validate_model_checkpoint(path, self.zho)
            self.assertEqual(value["average_drop"], 26)
            self.assertEqual(value["average_keep"], 6)
            self.assertEqual(value["format"], "bf16")

    def test_tokenizers_backend_compatibility_registration(self):
        import transformers

        backend = MODULE.patch_tokenizers_backend_for_vllm()
        self.assertIs(transformers.TokenizersBackend, backend)
        self.assertTrue(hasattr(backend, "from_pretrained"))
        self.assertTrue(hasattr(backend, "all_special_tokens_extended"))

    def test_tokenizers_backend_metadata_is_sanitized_without_model_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            original = {
                "tokenizer_class": "TokenizersBackend",
                "model_max_length": 131072,
            }
            (model / "tokenizer_config.json").write_text(
                json.dumps(original), encoding="utf-8"
            )
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")

            temporary, tokenizer_dir = MODULE.prepare_vllm_tokenizer(model)
            self.assertIsNotNone(temporary)
            try:
                sanitized = json.loads(
                    (tokenizer_dir / "tokenizer_config.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    sanitized["tokenizer_class"], "PreTrainedTokenizerFast"
                )
                self.assertEqual(
                    json.loads(
                        (model / "tokenizer_config.json").read_text(encoding="utf-8")
                    ),
                    original,
                )
            finally:
                temporary.cleanup()

    def test_prepare_model_accepts_prepared_model_without_hub_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PRIMARY / "prepare_model.py"),
                    "--submission-config",
                    str(root / "intentionally-missing-submission.json"),
                    "--cache-dir",
                    str(root / "cache"),
                    "--output",
                    str(model),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("already prepared", completed.stdout)

    def test_whole_document_translation_and_one_line_json(self):
        def callback(prompt, seed):
            source = prompt.rsplit("\n\n", 1)[1]
            if "{{name}}" in source:
                return "<p>你好 {{name}}</p>"
            if "#tag" in source:
                return "<p>你好 #标签</p>"
            return "你好。"

        generator = CallbackGenerator(callback)
        outputs, diagnostics = MODULE.translate_lines(
            generator,
            [
                "Hello speech.",
                "<p>Hello #tag</p>",
                '{"title":"Hello {{name}}","count":3}',
                "",
            ],
            self.zho,
            batch_size=4,
            max_new_tokens=100,
            segment_max_new_tokens=50,
        )
        self.assertEqual(outputs[0], "你好。")
        self.assertEqual(outputs[1], "<p>你好 #标签</p>")
        self.assertEqual(
            json.loads(outputs[2]),
            {"title": "你好 {{name}}", "count": 3},
        )
        self.assertNotIn("\n", outputs[2])
        self.assertEqual(outputs[3], "")
        self.assertEqual(diagnostics["final_whole_document_success"], 3)

    def test_whole_document_retry_uses_next_seed(self):
        def callback(prompt, seed):
            if seed == 0:
                return "<p>only one paragraph</p>"
            return "<p>第一段</p><p>第二段</p>"

        generator = CallbackGenerator(callback)
        outputs, _ = MODULE.translate_lines(
            generator,
            ["<p>Headline One</p><p>Body paragraph.</p>"],
            self.zho,
            batch_size=1,
            max_new_tokens=100,
            segment_max_new_tokens=50,
        )
        self.assertEqual(outputs, ["<p>第一段</p><p>第二段</p>"])
        self.assertEqual([call[1] for call in generator.calls], [0, 1])

    def test_non_json_segment_fallback_reconstructs_source_topology(self):
        def callback(prompt, seed):
            source = prompt.rsplit("\n\n", 1)[1]
            blocks = MODULE.P_BLOCK_RE.findall(source)
            if len(blocks) > 1:
                return "<p>invalid combined output</p>"
            return f"<p>译：{blocks[0]}</p>"

        generator = CallbackGenerator(callback)
        outputs, diagnostics = MODULE.translate_lines(
            generator,
            ["<p>First Headline</p><p>Second body.</p>"],
            self.zho,
            batch_size=1,
            max_new_tokens=100,
            segment_max_new_tokens=50,
        )
        self.assertEqual(outputs, ["<p>译：First Headline</p> <p>译：Second body.</p>"])
        self.assertEqual(diagnostics["final_segmented_success"], 1)
        whole_seeds = [call[1] for call in generator.calls[:10]]
        self.assertEqual(whole_seeds, list(range(10)))

    def test_json_leaf_fallback_preserves_placeholders_and_structure(self):
        def callback(prompt, seed):
            source = prompt.rsplit("\n\n", 1)[1]
            blocks = MODULE.P_BLOCK_RE.findall(source)
            if len(blocks) > 1:
                return "<p>invalid combined output</p>"
            value = blocks[0].replace("Hello", "你好").replace("Bye", "再见")
            return f"<p>{value}</p>"

        generator = CallbackGenerator(callback)
        outputs, diagnostics = MODULE.translate_lines(
            generator,
            ['{"a":"Hello {{user}}","b":"Bye","n":7}'],
            self.zho,
            batch_size=1,
            max_new_tokens=100,
            segment_max_new_tokens=50,
        )
        self.assertEqual(
            json.loads(outputs[0]),
            {"a": "你好 {{user}}", "b": "再见", "n": 7},
        )
        self.assertEqual(diagnostics["final_json_leaf_success"], 1)

    def test_raw_fallback_emits_raw_completion_not_parser_error(self):
        record = MODULE.PreparedRecord(
            index=0,
            source_text="A source sentence.",
            category="speech",
            model_category="speech",
            model_source_text="A source sentence.",
            instruction="Translate:",
            prompt="Translate:\n\nA source sentence.",
        )
        finals = {}
        diagnostics = MODULE.Counter()
        MODULE.apply_raw_fallbacks(
            [record],
            finals,
            {
                0: [
                    {
                        "seed": 0,
                        "model_hyp_text": "ERROR: HIT MAX TOKENS",
                        "raw_comp_text": "النص الخام الفعلي",
                        "failure_reasons": ["hit_max_tokens"],
                        "validation": {},
                    }
                ]
            },
            diagnostics,
        )
        self.assertEqual(
            finals[0],
            ("النص الخام الفعلي", "raw_completion_fallback"),
        )

    def test_all_submission_metadata(self):
        expected = {
            "ests-gptoss-zho-k26": ("eng-zho_Hans", "zho_Hans", 26, 5.15),
            "ests-gptoss-zho-k27": ("eng-zho_Hans", "zho_Hans", 27, 4.85),
            "ests-gptoss-zho-k28": ("eng-zho_Hans", "zho_Hans", 28, 4.55),
            "ests-gptoss-arz-k22": ("eng-ara_EG", "arz_Arab", 22, 6.33),
            "ests-gptoss-arz-k24": ("eng-ara_EG", "arz_Arab", 24, 5.74),
            "ests-gptoss-arz-k26": ("eng-ara_EG", "arz_Arab", 26, 5.15),
        }
        for submission_id, values in expected.items():
            path = ROOT / "submissions" / submission_id / "submission.json"
            raw_config = json.loads(path.read_text(encoding="utf-8"))
            config = MODULE.load_submission_config(path)
            self.assertEqual(config.submission_id, submission_id)
            self.assertEqual(
                (config.lang_pair, config.internal_target_lang, config.expected_average_drop),
                values[:3],
            )
            self.assertEqual(config.model_repo, f"oceanusm/{submission_id}")
            self.assertEqual(raw_config["artifact_format"], "mxfp4")
            self.assertEqual(raw_config["artifact_size_gib"], values[3])

    def test_shared_runtime_files_are_identical(self):
        shared = (
            "inference.py",
            "prepare_model.py",
            "pruned_gptoss.py",
            "wmt26_json_adapter.py",
            "wmt26_robust_inference.py",
            "requirements.txt",
            "setup.sh",
            "run.sh",
        )
        submissions = sorted((ROOT / "submissions").glob("ests-gptoss-*"))
        self.assertEqual(len(submissions), 6)
        for filename in shared:
            expected = (PRIMARY / filename).read_bytes()
            for submission in submissions:
                self.assertEqual(
                    (submission / filename).read_bytes(),
                    expected,
                    f"{submission.name}/{filename} drifted from the shared runtime",
                )


if __name__ == "__main__":
    unittest.main()
