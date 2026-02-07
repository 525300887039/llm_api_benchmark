"""可视化模块的单元测试."""

import os
import json
import unittest
import tempfile
import shutil


class TestLoadResults(unittest.TestCase):
    """测试 load_results 函数."""

    def setUp(self):
        self.results_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.results_dir, ignore_errors=True)

    def _write_json(self, filename, data):
        path = os.path.join(self.results_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def test_load_valid_results(self):
        """测试加载有效的 JSON 结果文件."""
        from llm_api_benchmark.visualize import load_results

        self._write_json("model_a.json", {
            "model": "model-a",
            "first_token_latency": 0.5,
            "token_throughput": 50.0,
        })
        self._write_json("model_b.json", {
            "model": "model-b",
            "first_token_latency": 0.3,
            "token_throughput": 80.0,
        })

        results = load_results(self.results_dir)
        self.assertEqual(len(results), 2)
        models = {r["model"] for r in results}
        self.assertIn("model-a", models)
        self.assertIn("model-b", models)

    def test_skip_invalid_json(self):
        """测试跳过无效的 JSON 文件."""
        from llm_api_benchmark.visualize import load_results

        self._write_json("good.json", {"model": "good-model"})
        # 写入非法 JSON
        bad_path = os.path.join(self.results_dir, "bad.json")
        with open(bad_path, "w") as f:
            f.write("{invalid json")

        results = load_results(self.results_dir)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["model"], "good-model")

    def test_skip_json_without_model(self):
        """测试跳过没有 model 字段的 JSON 文件."""
        from llm_api_benchmark.visualize import load_results

        self._write_json("no_model.json", {"name": "test", "value": 123})
        self._write_json("has_model.json", {"model": "m1"})

        results = load_results(self.results_dir)
        self.assertEqual(len(results), 1)

    def test_empty_directory(self):
        """测试空目录返回空列表."""
        from llm_api_benchmark.visualize import load_results

        results = load_results(self.results_dir)
        self.assertEqual(results, [])


class TestBuildOverviewDf(unittest.TestCase):
    """测试 _build_overview_df 函数."""

    def setUp(self):
        self.results = [
            {
                "name": "Model A",
                "model": "model-a",
                "api_type": "openai",
                "first_token_latency": 0.5,
                "first_token_latency_stats": {"p90": 0.58},
                "token_throughput": 50.0,
                "total_time": 2.0,
            },
            {
                "name": "Model B",
                "model": "model-b",
                "api_type": "claude",
                "first_token_latency": 0.3,
                "first_token_latency_stats": {"p90": 0.35},
                "token_throughput": 80.0,
                "total_time": 1.5,
            },
        ]

    def test_dataframe_shape(self):
        """测试 DataFrame 行列数."""
        from llm_api_benchmark.visualize import _build_overview_df

        df = _build_overview_df(self.results)
        self.assertEqual(len(df), 2)
        self.assertIn("Name", df.columns)
        self.assertIn("Model", df.columns)
        self.assertIn("Throughput (tokens/s)", df.columns)

    def test_dataframe_values(self):
        """测试 DataFrame 数值正确性."""
        from llm_api_benchmark.visualize import _build_overview_df

        df = _build_overview_df(self.results)
        row_a = df[df["Name"] == "Model A"].iloc[0]
        self.assertAlmostEqual(row_a["First Token Latency (s)"], 0.5, places=3)
        self.assertAlmostEqual(row_a["Throughput (tokens/s)"], 50.0, places=2)

    def test_missing_fields_use_defaults(self):
        """测试缺失字段使用默认值."""
        from llm_api_benchmark.visualize import _build_overview_df

        minimal = [{"model": "minimal-model"}]
        df = _build_overview_df(minimal)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["First Token Latency (s)"], 0)
        self.assertEqual(df.iloc[0]["Throughput (tokens/s)"], 0)


if __name__ == '__main__':
    unittest.main()
