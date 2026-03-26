"""
课程目标达成度分析系统 - Streamlit Web 应用
Course Objective Attainment Analysis System

依赖安装：
    pip install streamlit plotly openpyxl pandas numpy

运行：
    streamlit run app.py
"""

import io
import zipfile
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 页面基础配置
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="课程目标达成度分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义CSS：让界面更紧凑、专业
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .stAlert { margin-top: 0.3rem; margin-bottom: 0.3rem; }
    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.05rem !important; }
    .metric-card {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: bold; color: #1f3d7a; }
    .metric-lbl { font-size: 0.85rem; color: #555; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 核心数据结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AssessmentItem:
    """一个评估项（考试 / 作业 / 实验等）"""
    name: str
    proportion: float
    score_sheet: pd.DataFrame      # 得分表
    distribution: pd.DataFrame     # 分值分配表


@dataclass
class AttainmentResult:
    """达成度计算结果"""
    item_attainment: dict = field(default_factory=dict)   # {name: E_df}
    student_attainment: Optional[pd.DataFrame] = None     # 综合 (学生×子目标)
    objective_attainment: Optional[pd.Series] = None      # 各子目标班级达成度
    course_attainment: Optional[float] = None             # 课程总达成度
    student_total: Optional[pd.Series] = None             # 每个学生综合达成度


# ─────────────────────────────────────────────────────────────────────────────
# 计算逻辑（与独立脚本保持一致）
# ─────────────────────────────────────────────────────────────────────────────

def validate_distribution(dist: pd.DataFrame, item_name: str = "") -> list[str]:
    """校验分值分配表，返回错误列表（空列表=通过）"""
    errors = []
    obj_cols = dist.columns[2:]
    row_sums = dist[obj_cols].sum(axis=1)
    declared = dist.iloc[:, 1]
    mismatch = dist[~np.isclose(row_sums, declared, atol=1e-6)]
    if not mismatch.empty:
        bad = mismatch.iloc[:, 0].tolist()
        errors.append(f"以下题号满分 ≠ 各子目标分值之和：{bad}")
    if (dist.iloc[:, 1] <= 0).any():
        errors.append("存在满分 ≤ 0 的题目")
    return errors


def validate_weights(weights: pd.Series) -> list[str]:
    errors = []
    total = weights.sum()
    if not np.isclose(total, 1.0, atol=1e-6):
        errors.append(f"权重之和 = {total:.4f}，应为 1.0")
    if (weights < 0).any():
        errors.append("存在负权重")
    return errors


def validate_proportions(items: list[AssessmentItem]) -> list[str]:
    errors = []
    total = sum(i.proportion for i in items)
    if not np.isclose(total, 1.0, atol=1e-6):
        errors.append(f"各评估项占比之和 = {total:.4f}，应为 1.0")
    return errors


def compute_item_attainment(
    item: AssessmentItem,
    objective_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    计算单个评估项的子目标达成度矩阵。
    返回 (E_df, warnings_list)
    """
    warn_msgs = []
    dist = item.distribution.copy()
    score = item.score_sheet.copy()

    q_col = dist.columns[0]
    f_col = dist.columns[1]
    questions = dist[q_col].tolist()

    missing = [q for q in questions if q not in score.columns]
    if missing:
        raise ValueError(f"[{item.name}] 得分表缺少列：{missing}")

    student_ids = score.iloc[:, 1].astype(str)
    D = score[questions].values.astype(float)
    f = dist[f_col].values.astype(float)
    D_norm = D / f[np.newaxis, :]

    A = dist[objective_cols].values.astype(float)
    col_sums = A.sum(axis=0)
    zero_mask = col_sums == 0
    if zero_mask.any():
        zc = [objective_cols[i] for i, z in enumerate(zero_mask) if z]
        warn_msgs.append(f"[{item.name}] 子目标 {zc} 在所有题目中分值为0")
        col_sums = np.where(zero_mask, np.nan, col_sums)

    P = A / col_sums[np.newaxis, :]
    E = D_norm @ P
    return pd.DataFrame(E, index=student_ids, columns=objective_cols), warn_msgs


def analyze(
    items: list[AssessmentItem],
    weights: pd.Series,
) -> AttainmentResult:
    """主计算函数"""
    objective_cols = weights.index.tolist()
    result = AttainmentResult()
    combined = None

    for item in items:
        E_item, _ = compute_item_attainment(item, objective_cols)
        result.item_attainment[item.name] = E_item
        if combined is None:
            combined = E_item * item.proportion
        else:
            combined = combined.add(E_item * item.proportion, fill_value=0)

    result.student_attainment = combined
    result.objective_attainment = combined.mean(axis=0)
    result.course_attainment = float(result.objective_attainment @ weights)
    result.student_total = combined @ weights
    return result


def compute_baseline(items: list[AssessmentItem], weights: pd.Series) -> pd.Series:
    """简单加权基准法（对比用）"""
    objective_cols = weights.index.tolist()
    combined = None
    for item in items:
        dist = item.distribution
        score = item.score_sheet
        questions = dist.iloc[:, 0].tolist()
        D = score[questions].values.astype(float)
        f = dist.iloc[:, 1].values.astype(float)
        mean_rate = (D / f[np.newaxis, :]).mean(axis=1)
        E_simple = np.tile(mean_rate[:, np.newaxis], (1, len(objective_cols)))
        E_df = pd.DataFrame(E_simple,
                            index=score.iloc[:, 1].astype(str),
                            columns=objective_cols)
        if combined is None:
            combined = E_df * item.proportion
        else:
            combined = combined.add(E_df * item.proportion, fill_value=0)
    return combined.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Excel 读取工具
# ─────────────────────────────────────────────────────────────────────────────

def read_excel(uploaded_file) -> Optional[pd.DataFrame]:
    """从上传文件读取 Excel，返回 DataFrame 或 None"""
    if uploaded_file is None:
        return None
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"读取文件失败：{e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Plotly 图表函数
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = px.colors.qualitative.Set2
THRESHOLD_COLOR = "rgba(100,100,100,0.6)"


def fig_radar(obj_attainment: pd.Series, item_attainments: dict,
              threshold: float) -> go.Figure:
    """雷达图：各子目标班级达成度"""
    objectives = list(obj_attainment.index)
    m = len(objectives)
    angles = objectives + [objectives[0]]

    fig = go.Figure()

    # 阈值圆
    fig.add_trace(go.Scatterpolar(
        r=[threshold] * (m + 1), theta=angles,
        mode="lines", name=f"阈值 {threshold}",
        line=dict(color=THRESHOLD_COLOR, dash="dash", width=1.5),
        fill=None,
    ))

    # 各评估项
    for idx, (name, E) in enumerate(item_attainments.items()):
        vals = list(E.mean(axis=0).values) + [E.mean(axis=0).values[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=angles, mode="lines+markers",
            name=name, opacity=0.7,
            line=dict(color=PALETTE[idx % len(PALETTE)], width=1.5),
        ))

    # 综合
    vals = list(obj_attainment.values) + [obj_attainment.values[0]]
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=angles, mode="lines+markers+text",
        name="综合", text=[f"{v:.3f}" for v in vals],
        textposition="top center",
        line=dict(color="#1f3d7a", width=2.5),
        fill="toself", fillcolor="rgba(31,61,122,0.08)",
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], tickfont_size=10)),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=40, b=60, l=60, r=60),
        height=400,
    )
    return fig


def fig_boxplot(student_attainment: pd.DataFrame, threshold: float) -> go.Figure:
    """箱线图：各子目标达成度分布"""
    objectives = list(student_attainment.columns)
    fig = go.Figure()
    for i, obj in enumerate(objectives):
        vals = student_attainment[obj].values
        fig.add_trace(go.Box(
            y=vals, name=obj,
            marker_color=PALETTE[i % len(PALETTE)],
            boxmean="sd",
            hovertemplate=f"{obj}<br>达成度: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(y=threshold, line_dash="dash",
                  line_color=THRESHOLD_COLOR,
                  annotation_text=f"阈值 {threshold}",
                  annotation_position="right")
    fig.update_layout(
        yaxis=dict(range=[0, 1.05], title="达成度"),
        xaxis_title="课程子目标",
        showlegend=False,
        height=380,
        margin=dict(t=30, b=40),
    )
    return fig


def fig_heatmap(student_attainment: pd.DataFrame, threshold: float) -> go.Figure:
    """热力图：学生 × 子目标达成度"""
    fig = go.Figure(data=go.Heatmap(
        z=student_attainment.values,
        x=list(student_attainment.columns),
        y=list(student_attainment.index),
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        hovertemplate="学号: %{y}<br>子目标: %{x}<br>达成度: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title="达成度",
            tickvals=[0, threshold, 1],
            ticktext=["0", f"阈值\n{threshold}", "1"],
        ),
    ))
    n = len(student_attainment)
    fig.update_layout(
        xaxis_title="课程子目标",
        yaxis_title="学号",
        height=max(350, n * 14 + 80),
        margin=dict(t=30, b=50, l=80, r=20),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def fig_comparison(obj_attainment: pd.Series,
                   baseline: pd.Series,
                   threshold: float) -> go.Figure:
    """方法对比柱状图"""
    objectives = list(obj_attainment.index)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="本文方法（归一化矩阵）",
        x=objectives, y=obj_attainment.values,
        marker_color=PALETTE[0],
        text=[f"{v:.3f}" for v in obj_attainment.values],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="简单加权法（基准）",
        x=objectives, y=baseline.values,
        marker_color=PALETTE[1],
        text=[f"{v:.3f}" for v in baseline.values],
        textposition="outside",
    ))
    fig.add_hline(y=threshold, line_dash="dash",
                  line_color=THRESHOLD_COLOR,
                  annotation_text=f"阈值 {threshold}")
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.15], title="班级达成度（均值）"),
        xaxis_title="课程子目标",
        legend=dict(orientation="h", y=-0.2),
        height=380,
        margin=dict(t=30, b=70),
    )
    return fig


def fig_scatter(item_attainments: dict, threshold: float) -> Optional[go.Figure]:
    """散点图：两个评估项达成度相关性（如有）"""
    names = list(item_attainments.keys())
    if len(names) < 2:
        return None

    name_x, name_y = names[0], names[1]
    Ex = item_attainments[name_x]
    Ey = item_attainments[name_y]
    objectives = list(Ex.columns)
    m = len(objectives)

    cols = min(m, 3)
    rows = (m + cols - 1) // cols
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=objectives)

    for idx, obj in enumerate(objectives):
        r = idx // cols + 1
        c = idx % cols + 1
        x_vals = Ex[obj].values
        y_vals = Ey[obj].values
        color = PALETTE[idx % len(PALETTE)]

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="markers",
            marker=dict(color=color, size=7, opacity=0.65),
            name=obj,
            hovertemplate=f"学号: %{{text}}<br>{name_x}: %{{x:.3f}}<br>{name_y}: %{{y:.3f}}<extra></extra>",
            text=list(Ex.index),
            showlegend=False,
        ), row=r, col=c)

        # 趋势线
        if len(x_vals) > 2:
            for i, (x, y) in enumerate(zip(x_vals, y_vals)):
                if not (np.isfinite(x) and np.isfinite(y)):
                    print(f"异常数据 index={i}, x={x}, y={y}")
            print("NaN数量:", np.isnan(x_vals).sum(), np.isnan(y_vals).sum())
            print("Inf数量:", np.isinf(x_vals).sum(), np.isinf(y_vals).sum())
            print("唯一值数量:", len(set(x_vals)))
            coef = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
            corr = np.corrcoef(x_vals, y_vals)[0, 1]
            fig.add_trace(go.Scatter(
                x=x_line, y=np.polyval(coef, x_line),
                mode="lines", line=dict(color="black", dash="dot", width=1.2),
                name=f"r={corr:.2f}",
                showlegend=(idx == 0),
            ), row=r, col=c)
            # 在子图中标注 r 值
          
            xref = f"x{idx+1} domain" if idx > 0 else "x domain"
            yref = f"y{idx+1} domain" if idx > 0 else "y domain"
            fig.add_annotation(
                x=0.05,
                y=0.92,
                xref=xref,
                yref=yref,
                text=f"r = {corr:.3f}",
                showarrow=False,
                font=dict(size=11, color="black"),
                )
            # fig.add_annotation(
            #     x=0.05, y=0.92, xref=f"x{idx+1} domain", yref=f"y{idx+1} domain",
            #     text=f"r = {corr:.3f}", showarrow=False,
            #     font=dict(size=11, color="black"),
            # )

        # 阈值线
        fig.add_hline(y=threshold, line_dash="dash",
                      line_color=THRESHOLD_COLOR, row=r, col=c)
        fig.add_vline(x=threshold, line_dash="dash",
                      line_color=THRESHOLD_COLOR, row=r, col=c)

    fig.update_layout(
        height=320 * rows,
        xaxis_title=f"{name_x} 达成度",
        yaxis_title=f"{name_y} 达成度",
        margin=dict(t=50, b=50),
    )
    return fig


def fig_cdf(student_attainment: pd.DataFrame,
            student_total: pd.Series,
            threshold: float) -> go.Figure:
    """CDF 累积分布图"""
    fig = go.Figure()
    thresholds = np.linspace(0, 1, 300)
    objectives = list(student_attainment.columns)

    for idx, obj in enumerate(objectives):
        vals = student_attainment[obj].values
        cdf = [(vals >= t).mean() for t in thresholds]
        pct_at_thr = (vals >= threshold).mean()
        fig.add_trace(go.Scatter(
            x=thresholds, y=cdf, mode="lines",
            name=f"{obj}（{pct_at_thr:.0%}@{threshold}）",
            line=dict(color=PALETTE[idx % len(PALETTE)], width=2),
            hovertemplate=f"{obj}<br>阈值: %{{x:.2f}}<br>达标比例: %{{y:.1%}}<extra></extra>",
        ))

    total_cdf = [(student_total.values >= t).mean() for t in thresholds]
    total_pct = (student_total.values >= threshold).mean()
    fig.add_trace(go.Scatter(
        x=thresholds, y=total_cdf, mode="lines",
        name=f"综合（{total_pct:.0%}@{threshold}）",
        line=dict(color="#1f3d7a", width=2.5, dash="dash"),
        hovertemplate=f"综合<br>阈值: %{{x:.2f}}<br>达标比例: %{{y:.1%}}<extra></extra>",
    ))

    fig.add_vline(x=threshold, line_dash="dash",
                  line_color=THRESHOLD_COLOR,
                  annotation_text=f"阈值 {threshold}",
                  annotation_position="top right")

    fig.update_layout(
        xaxis=dict(range=[0, 1], title="达成度阈值"),
        yaxis=dict(range=[0, 1.05], title="达到阈值的学生比例",
                   tickformat=".0%"),
        legend=dict(orientation="h", y=-0.25),
        height=380,
        margin=dict(t=30, b=80),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 导出工具
# ─────────────────────────────────────────────────────────────────────────────

def export_excel(result: AttainmentResult) -> bytes:
    """导出结果为 Excel，返回字节流"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        result.student_attainment.to_excel(writer, sheet_name="学生子目标达成度")
        result.objective_attainment.to_frame("班级达成度").to_excel(
            writer, sheet_name="子目标班级达成度")
        result.student_total.to_frame("综合达成度").to_excel(
            writer, sheet_name="学生综合达成度")
        for name, E in result.item_attainment.items():
            E.to_excel(writer, sheet_name=f"{name[:20]}_达成度")
        summary = pd.DataFrame({
            "子目标": list(result.objective_attainment.index),
            "班级达成度": list(result.objective_attainment.values),
        })
        summary.loc[len(summary)] = ["课程总达成度", result.course_attainment]
        summary.to_excel(writer, sheet_name="汇总", index=False)
    return buf.getvalue()


def export_figures_zip(figs: dict) -> bytes:
    """将所有 Plotly 图表导出为 PNG 并打包成 ZIP"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for fname, fig in figs.items():
            try:
                img_bytes = fig.to_image(format="png", width=900, height=500, scale=2)
                zf.writestr(fname, img_bytes)
            except Exception:
                # kaleido 未安装时跳过图片导出
                pass
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Session State 初始化
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    if "assessment_items" not in st.session_state:
        # 默认两个评估项
        st.session_state.assessment_items = [
            {"name": "期末考试", "proportion": 70,
             "score_df": None, "dist_df": None,
             "score_name": None, "dist_name": None},
            {"name": "平时成绩", "proportion": 30,
             "score_df": None, "dist_df": None,
             "score_name": None, "dist_name": None},
        ]
    if "weight_df" not in st.session_state:
        st.session_state.weight_df = None
    if "result" not in st.session_state:
        st.session_state.result = None
    if "adjusted_weights" not in st.session_state:
        st.session_state.adjusted_weights = {}
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.6


init_state()


# ─────────────────────────────────────────────────────────────────────────────
# 侧边栏
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 达成度分析系统")
    st.markdown("---")

    # ── ① 评估项管理 ─────────────────────────────────────────
    st.subheader("① 评估项配置")

    # 添加 / 删除评估项
    col_add, col_del = st.columns(2)
    with col_add:
        if st.button("➕ 添加评估项", use_container_width=True):
            st.session_state.assessment_items.append({
                "name": f"评估项{len(st.session_state.assessment_items)+1}",
                "proportion": 0,
                "score_df": None, "dist_df": None,
                "score_name": None, "dist_name": None,
            })
    with col_del:
        if (st.button("➖ 删除最后项", use_container_width=True)
                and len(st.session_state.assessment_items) > 1):
            st.session_state.assessment_items.pop()

    total_prop = sum(i["proportion"]
                     for i in st.session_state.assessment_items)
    prop_ok = np.isclose(total_prop, 100, atol=0.5)
    prop_color = "normal" if prop_ok else "error"
    st.markdown(
        f"占比总和：**{total_prop}%** "
        + ("✅" if prop_ok else "❌ 应为100%"),
        unsafe_allow_html=False,
    )

    for idx, item in enumerate(st.session_state.assessment_items):
        with st.expander(f"📋 {item['name']}", expanded=(idx == 0)):
            # 名称
            item["name"] = st.text_input(
                "评估项名称", value=item["name"],
                key=f"name_{idx}")
            # 占比
            item["proportion"] = st.number_input(
                "占比（%）", min_value=0, max_value=100,
                value=item["proportion"], step=5,
                key=f"prop_{idx}")

            # 上传得分表
            score_file = st.file_uploader(
                "📥 得分表 (.xlsx)",
                type=["xlsx"], key=f"score_{idx}",
                help="列：序号 | 学号 | 姓名 | 题1 | 题2 | …")
            if score_file is not None:
                df = read_excel(score_file)
                if df is not None:
                    item["score_df"] = df
                    item["score_name"] = score_file.name
                    st.success(f"✅ {score_file.name}  ({df.shape[0]}行×{df.shape[1]}列)")
            elif item["score_df"] is not None:
                st.info(f"已加载：{item['score_name']}")

            # 上传分值分配表
            dist_file = st.file_uploader(
                "📥 分值分配表 (.xlsx)",
                type=["xlsx"], key=f"dist_{idx}",
                help="列：题号 | 满分 | 子目标1 | 子目标2 | …")
            if dist_file is not None:
                df = read_excel(dist_file)
                if df is not None:
                    item["dist_df"] = df
                    item["dist_name"] = dist_file.name
                    st.success(f"✅ {dist_file.name}  ({df.shape[0]}行×{df.shape[1]}列)")
            elif item["dist_df"] is not None:
                st.info(f"已加载：{item['dist_name']}")

    st.markdown("---")

    # ── ② 目标权重表 ──────────────────────────────────────────
    st.subheader("② 目标权重表")

    weight_file = st.file_uploader(
        "📥 目标权重表 (.xlsx)", type=["xlsx"],
        key="weight_file",
        help="列：子目标 | 权重（各权重之和应为1）")
    if weight_file is not None:
        df = read_excel(weight_file)
        if df is not None:
            st.session_state.weight_df = df
            # 初始化调整后权重
            obj_col = df.columns[0]
            w_col = df.columns[1]
            for _, row in df.iterrows():
                key = str(row[obj_col])
                if key not in st.session_state.adjusted_weights:
                    st.session_state.adjusted_weights[key] = float(row[w_col])
            st.success(f"✅ {weight_file.name}")

    # 动态权重调整区
    if st.session_state.weight_df is not None:
        wdf = st.session_state.weight_df
        obj_col = wdf.columns[0]
        w_col = wdf.columns[1]
        objectives = [str(r) for r in wdf[obj_col].tolist()]
        m = len(objectives)

        st.markdown("**动态调整权重**")
        new_weights = {}
        for obj in objectives:
            init_val = st.session_state.adjusted_weights.get(obj, 1.0 / m)
            new_weights[obj] = st.slider(
                obj, min_value=0.0, max_value=1.0,
                value=float(init_val), step=0.01,
                key=f"w_{obj}",
            )
        st.session_state.adjusted_weights = new_weights

        w_sum = sum(new_weights.values())
        w_ok = np.isclose(w_sum, 1.0, atol=0.01)
        st.markdown(
            f"权重之和：**{w_sum:.3f}** " + ("✅" if w_ok else "❌ 应为1.0"),
        )
        if not w_ok:
            if st.button("🔧 自动归一化", use_container_width=True):
                if w_sum > 0:
                    for k in new_weights:
                        st.session_state.adjusted_weights[k] = new_weights[k] / w_sum
                    st.rerun()

    st.markdown("---")

    # ── ③ 分析设置 ────────────────────────────────────────────
    st.subheader("③ 分析设置")
    st.session_state.threshold = st.slider(
        "达成度合格阈值", 0.0, 1.0,
        value=st.session_state.threshold, step=0.05,
    )

    st.markdown("---")

    # ── ④ 开始分析按钮 ────────────────────────────────────────
    run_btn = st.button("▶ 开始分析", type="primary",
                        use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 分析触发逻辑
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    errors_all = []

    # 构造 AssessmentItem 列表
    items = []
    for item in st.session_state.assessment_items:
        if item["score_df"] is None:
            errors_all.append(f"【{item['name']}】未上传得分表")
            continue
        if item["dist_df"] is None:
            errors_all.append(f"【{item['name']}】未上传分值分配表")
            continue
        errs = validate_distribution(item["dist_df"], item["name"])
        for e in errs:
            errors_all.append(f"【{item['name']}分值分配表】{e}")
        ai = AssessmentItem(
            name=item["name"],
            proportion=item["proportion"] / 100.0,
            score_sheet=item["score_df"],
            distribution=item["dist_df"],
        )
        items.append(ai)

    # 构造权重 Series（使用调整后的权重）
    weights = None
    if st.session_state.adjusted_weights:
        weights = pd.Series(st.session_state.adjusted_weights)
        w_errs = validate_weights(weights)
        for e in w_errs:
            errors_all.append(f"【目标权重表】{e}")
    else:
        errors_all.append("未加载目标权重表")

    # 比例校验
    if items:
        prop_errs = validate_proportions(items)
        errors_all.extend(prop_errs)

    if errors_all:
        for e in errors_all:
            st.error(e)
    else:
        try:
            with st.spinner("正在计算..."):
                st.session_state.result = analyze(items, weights)
                st.session_state.items_snapshot = items
                st.session_state.weights_snapshot = weights
            st.success("✅ 分析完成！")
        except Exception as e:
            st.error(f"计算出错：{e}")


# ─────────────────────────────────────────────────────────────────────────────
# 主内容区
# ─────────────────────────────────────────────────────────────────────────────

st.title("课程目标达成度分析系统")

result: AttainmentResult = st.session_state.get("result", None)
threshold = st.session_state.threshold

if result is None:
    # 未分析时显示使用说明
    st.info("👈 请在左侧侧边栏上传数据并点击【▶ 开始分析】")

    with st.expander("📖 使用说明 & 输入格式", expanded=True):
        st.markdown("""
### 使用流程
1. **配置评估项**：填写名称（如期末考试、平时成绩），设定占比，分别上传得分表和分值分配表
2. **上传目标权重表**：加载后可在侧边栏实时调整各子目标权重
3. **设置达成度阈值**（默认0.6）
4. 点击 **▶ 开始分析** 查看结果

---
### 输入文件格式

**得分表（Score Sheet）**

| 序号 | 学号 | 姓名 | 题1 | 题2 | … | 题n |
|------|------|------|-----|-----|---|-----|
| 1 | 2024001 | 张三 | 8 | 6 | … | 9 |

**分值分配表（Score Distribution Matrix）**

| 题号 | 满分 | 子目标1 | 子目标2 | … | 子目标m |
|------|------|---------|---------|---|---------|
| 题1 | 10 | 6 | 4 | … | 0 |

> ⚠️ 校验规则：每行「满分」= 各子目标分值之和

**目标权重表（Objective Weight Table）**

| 子目标 | 权重 |
|--------|------|
| 子目标1 | 0.4 |
| 子目标2 | 0.35 |
| 子目标3 | 0.25 |

> ⚠️ 校验规则：各权重之和 = 1.0
""")
else:
    # ── 顶部指标卡片 ────────────────────────────────────────────
    objectives = list(result.objective_attainment.index)
    m = len(objectives)
    n_stu = len(result.student_total)
    threshold_pct = (result.student_total >= threshold).mean()

    metric_cols = st.columns(m + 2)
    with metric_cols[0]:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-val">{result.course_attainment:.3f}</div>'
            f'<div class="metric-lbl">课程总达成度</div></div>',
            unsafe_allow_html=True,
        )
    with metric_cols[1]:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-val">{threshold_pct:.0%}</div>'
            f'<div class="metric-lbl">达标学生比例（≥{threshold}）</div></div>',
            unsafe_allow_html=True,
        )
    for i, obj in enumerate(objectives):
        val = result.objective_attainment[obj]
        color = "#1a7a3a" if val >= threshold else "#b71c1c"
        with metric_cols[i + 2]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-val" style="color:{color}">{val:.3f}</div>'
                f'<div class="metric-lbl">{obj} 达成度</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ────────────────────────────────────────────────────
    tabs = st.tabs([
        "📋 数据预览",
        "🎯 子目标达成度",
        "🔥 热力图",
        "⚖️ 方法对比",
        "🔗 相关性 & CDF",
        "💾 导出",
    ])

    items_snap = st.session_state.get("items_snapshot", [])
    weights_snap = st.session_state.get("weights_snapshot",
                                        pd.Series(dtype=float))

    # ── Tab 1：数据预览 ──────────────────────────────────────────
    with tabs[0]:
        st.subheader("数据预览与校验状态")

        for item in st.session_state.assessment_items:
            with st.expander(f"📋 {item['name']}（占比 {item['proportion']}%）"):
                if item["score_df"] is not None:
                    st.markdown("**得分表**")
                    st.dataframe(item["score_df"], use_container_width=True,
                                 height=200)
                    errs = validate_distribution(item["dist_df"], item["name"]) \
                        if item["dist_df"] is not None else []
                else:
                    st.warning("未上传得分表")

                if item["dist_df"] is not None:
                    st.markdown("**分值分配表**")
                    st.dataframe(item["dist_df"], use_container_width=True,
                                 height=200)
                    errs = validate_distribution(item["dist_df"], item["name"])
                    if errs:
                        for e in errs:
                            st.error(f"❌ {e}")
                    else:
                        st.success("✅ 分值分配表校验通过")
                else:
                    st.warning("未上传分值分配表")

        if st.session_state.weight_df is not None:
            with st.expander("⚖️ 目标权重表"):
                st.dataframe(st.session_state.weight_df,
                             use_container_width=True)
                # 显示当前调整后权重
                adj = st.session_state.adjusted_weights
                adj_df = pd.DataFrame(
                    {"子目标": list(adj.keys()), "调整后权重": list(adj.values())}
                )
                st.markdown("**当前生效权重（调整后）**")
                st.dataframe(adj_df, use_container_width=True)

    # ── Tab 2：子目标达成度 ───────────────────────────────────────
    with tabs[1]:
        st.subheader("各子目标班级达成度")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(
                fig_radar(result.objective_attainment,
                          result.item_attainment, threshold),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                fig_boxplot(result.student_attainment, threshold),
                use_container_width=True,
            )

        # 达成度数值表
        st.markdown("**子目标达成度详情**")
        detail = pd.DataFrame({
            "子目标": list(result.objective_attainment.index),
            "班级达成度（均值）": [f"{v:.4f}" for v in result.objective_attainment.values],
            "中位数": [f"{result.student_attainment[obj].median():.4f}"
                       for obj in result.objective_attainment.index],
            "标准差": [f"{result.student_attainment[obj].std():.4f}"
                       for obj in result.objective_attainment.index],
            f"达标率（≥{threshold}）": [
                f"{(result.student_attainment[obj] >= threshold).mean():.1%}"
                for obj in result.objective_attainment.index
            ],
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)

    # ── Tab 3：热力图 ─────────────────────────────────────────────
    with tabs[2]:
        st.subheader("学生-子目标达成度热力图")
        st.info("悬停可查看具体数值；颜色越绿达成度越高，越红越低")
        st.plotly_chart(
            fig_heatmap(result.student_attainment, threshold),
            use_container_width=True,
        )

    # ── Tab 4：方法对比 ───────────────────────────────────────────
    with tabs[3]:
        st.subheader("本文方法 vs 简单加权基准方法")
        st.markdown("""
**简单加权法**：每个学生的全部题目平均得分率作为所有子目标的统一达成度，
等价于忽略各题目对不同子目标的差异化贡献。
        """)
        baseline = compute_baseline(items_snap, weights_snap)
        st.plotly_chart(
            fig_comparison(result.objective_attainment, baseline, threshold),
            use_container_width=True,
        )

        diff_df = pd.DataFrame({
            "子目标": list(result.objective_attainment.index),
            "本文方法": [f"{v:.4f}" for v in result.objective_attainment.values],
            "简单加权法": [f"{v:.4f}" for v in baseline.values],
            "差值（本文−基准）": [
                f"{a-b:+.4f}"
                for a, b in zip(result.objective_attainment.values, baseline.values)
            ],
        })
        st.dataframe(diff_df, use_container_width=True, hide_index=True)

    # ── Tab 5：相关性 & CDF ───────────────────────────────────────
    with tabs[4]:
        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.subheader("评估项相关性散点图")
            if len(result.item_attainment) >= 2:
                scatter = fig_scatter(result.item_attainment, threshold)
                if scatter:
                    st.plotly_chart(scatter, use_container_width=True)
            else:
                st.info("需要至少2个评估项才能绘制散点图")

        with col_r:
            st.subheader("达成度累积分布（CDF）")
            st.plotly_chart(
                fig_cdf(result.student_attainment,
                        result.student_total, threshold),
                use_container_width=True,
            )

    # ── Tab 6：导出 ────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("导出分析结果")
        col_ex1, col_ex2 = st.columns(2)

        with col_ex1:
            st.markdown("**📊 导出结果 Excel**")
            st.markdown("包含：学生子目标达成度、各评估项达成度、汇总表")
            xlsx_bytes = export_excel(result)
            st.download_button(
                label="⬇️ 下载 Excel",
                data=xlsx_bytes,
                file_name="达成度分析结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with col_ex2:
            st.markdown("**🖼️ 导出图表（ZIP）**")
            st.markdown("包含全部6张图表（需安装 `kaleido`）")
            figs_dict = {
                "fig2_radar.png": fig_radar(result.objective_attainment,
                                            result.item_attainment, threshold),
                "fig3_boxplot.png": fig_boxplot(result.student_attainment, threshold),
                "fig1_heatmap.png": fig_heatmap(result.student_attainment, threshold),
                "fig4_comparison.png": fig_comparison(
                    result.objective_attainment,
                    compute_baseline(items_snap, weights_snap), threshold),
                "fig6_cdf.png": fig_cdf(result.student_attainment,
                                        result.student_total, threshold),
            }
            if len(result.item_attainment) >= 2:
                s = fig_scatter(result.item_attainment, threshold)
                if s:
                    figs_dict["fig5_scatter.png"] = s
            zip_bytes = export_figures_zip(figs_dict)
            st.download_button(
                label="⬇️ 下载图表 ZIP",
                data=zip_bytes,
                file_name="达成度分析图表.zip",
                mime="application/zip",
                use_container_width=True,
            )

        # 学生个人达成度明细
        st.markdown("---")
        st.subheader("学生个人达成度明细")
        detail_df = result.student_attainment.copy()
        detail_df["综合达成度"] = result.student_total
        detail_df["是否达标"] = detail_df["综合达成度"].apply(
            lambda x: "✅" if x >= threshold else "❌"
        )
        st.dataframe(
            detail_df.style.background_gradient(
                subset=list(result.student_attainment.columns) + ["综合达成度"],
                cmap="RdYlGn", vmin=0, vmax=1,
            ),
            use_container_width=True,
            height=400,
        )
