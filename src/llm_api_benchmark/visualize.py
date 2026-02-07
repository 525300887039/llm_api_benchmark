"""可视化报告模块，使用 Streamlit 生成交互式 Web 报告."""

import json
import os
import glob


def load_results(results_dir: str) -> list:
    """
    从目录中加载所有 JSON 结果文件.

    Args:
        results_dir: 结果文件目录

    Returns:
        list: 结果字典列表
    """
    results = []
    pattern = os.path.join(results_dir, "*.json")
    for filepath in sorted(glob.glob(pattern)):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if "model" in data:
                    results.append(data)
            except json.JSONDecodeError:
                continue
    return results


def run_dashboard(results_dir: str):
    """
    启动 Streamlit 交互式仪表盘.

    Args:
        results_dir: 结果文件目录
    """
    import streamlit as st
    import pandas as pd

    st.set_page_config(page_title="LLM API Benchmark", layout="wide")
    st.title("LLM API Benchmark Report")

    # 加载数据
    results = load_results(results_dir)
    if not results:
        st.warning(f"未在 `{results_dir}` 中找到测试结果文件。请先运行批量测试。")
        return

    # 构建概览 DataFrame
    overview = _build_overview_df(results)

    # ---- 概览指标卡片 ----
    st.header("Overview")
    _render_metric_cards(st, overview)

    # ---- 对比图表 ----
    st.header("Performance Comparison")
    _render_charts(st, overview)

    # ---- 详细数据表 ----
    st.header("Detail Table")
    st.dataframe(overview, use_container_width=True)

    # ---- 每个模型的详细统计 ----
    st.header("Per-Model Statistics")
    _render_per_model_stats(st, results)


def _build_overview_df(results: list):
    """从结果列表构建概览 DataFrame."""
    import pandas as pd

    rows = []
    for r in results:
        rows.append({
            "Name": r.get("name", r.get("model", "unknown")),
            "Model": r.get("model", ""),
            "API Type": r.get("api_type", "openai"),
            "First Token Latency (s)": round(r.get("first_token_latency", 0), 3),
            "P90 Latency (s)": round(
                r.get("first_token_latency_stats", {}).get("p90", 0), 3
            ),
            "Throughput (tokens/s)": round(r.get("token_throughput", 0), 2),
            "Total Time (s)": round(r.get("total_time", 0), 2),
        })
    return pd.DataFrame(rows)


def _render_metric_cards(st_module, overview):
    """渲染概览指标卡片."""
    cols = st_module.columns(4)
    cols[0].metric("Models Tested", len(overview))
    if not overview.empty:
        cols[1].metric(
            "Best Latency",
            f"{overview['First Token Latency (s)'].min():.3f}s",
        )
        cols[2].metric(
            "Best Throughput",
            f"{overview['Throughput (tokens/s)'].max():.1f} t/s",
        )
        cols[3].metric(
            "Fastest Response",
            f"{overview['Total Time (s)'].min():.2f}s",
        )


def _render_charts(st_module, overview):
    """渲染对比图表."""
    if overview.empty:
        return

    chart_data = overview.set_index("Name")

    col1, col2 = st_module.columns(2)

    with col1:
        st_module.subheader("First Token Latency")
        st_module.bar_chart(
            chart_data[["First Token Latency (s)", "P90 Latency (s)"]],
        )

    with col2:
        st_module.subheader("Token Throughput")
        st_module.bar_chart(
            chart_data[["Throughput (tokens/s)"]],
        )

    st_module.subheader("Total Response Time")
    st_module.bar_chart(
        chart_data[["Total Time (s)"]],
    )


def _render_per_model_stats(st_module, results: list):
    """渲染每个模型的详细统计信息."""
    import pandas as pd

    for r in results:
        name = r.get("name", r.get("model", "unknown"))
        with st_module.expander(f"{name}", expanded=False):
            st_module.markdown(f"**Model:** {r.get('model', '')}  |  "
                               f"**API Type:** {r.get('api_type', 'openai')}  |  "
                               f"**API URL:** `{r.get('api_url', '')}`")

            _render_stats_table(st_module, pd, "First Token Latency (s)",
                                r.get("first_token_latency_stats", {}))
            _render_stats_table(st_module, pd, "Token Throughput (tokens/s)",
                                r.get("token_throughput_stats", {}))
            _render_stats_table(st_module, pd, "Total Response Time (s)",
                                r.get("total_time_stats", {}))


def _render_stats_table(st_module, pd_module, title: str, stats: dict):
    """渲染单个指标的统计表格."""
    if not stats:
        return
    st_module.markdown(f"**{title}**")
    row = {
        "Avg": stats.get("avg", 0),
        "Min": stats.get("min", 0),
        "Max": stats.get("max", 0),
        "Median": stats.get("median", 0),
        "P90": stats.get("p90", 0),
        "P99": stats.get("p99", 0),
        "Std Dev": stats.get("std_dev", 0),
    }
    df = pd_module.DataFrame([row])
    st_module.dataframe(df, use_container_width=True, hide_index=True)


# ---- Streamlit 入口点 ----
# 当通过 `streamlit run visualize.py -- --results_dir ./results` 执行时
if __name__ == "__main__" or "streamlit" in globals().get("__loader__", "").__class__.__name__.lower() if hasattr(globals().get("__loader__", ""), "__class__") else False:
    import argparse as _ap

    _parser = _ap.ArgumentParser()
    _parser.add_argument("--results_dir", default="./results")
    _args, _ = _parser.parse_known_args()
    run_dashboard(_args.results_dir)
