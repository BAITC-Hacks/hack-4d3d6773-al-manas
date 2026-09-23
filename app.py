from collections import defaultdict, deque
from io import BytesIO

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Граф денег — AML-анализ", layout="wide")
st.title("Граф денег: поиск узлов для проверки")
st.caption(
    "Приоритет и роли ниже — аналитические гипотезы по структуре транзакций, "
    "а не доказательство нарушения."
)


def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет и нормализует CSV к source, target, amount, timestamp."""
    required = {"source", "target", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"В CSV не хватает колонок: {', '.join(sorted(missing))}. "
            "Обязательны source, target, amount."
        )

    result = df.copy()
    result["source"] = result["source"].astype("string").str.strip()
    result["target"] = result["target"].astype("string").str.strip()
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce")

    result = result.dropna(subset=["source", "target", "amount"])
    result = result[
        (result["source"] != "")
        & (result["target"] != "")
        & (result["amount"] > 0)
    ].copy()

    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"], errors="coerce", utc=True
        )
    else:
        result["timestamp"] = pd.NaT

    if result.empty:
        raise ValueError("После проверки в CSV не осталось корректных транзакций.")

    return result


def build_graph(transactions: pd.DataFrame) -> nx.DiGraph:
    """Создаёт граф: направление ребра соответствует движению денег."""
    graph = nx.DiGraph()

    grouped = transactions.groupby(["source", "target"], as_index=False).agg(
        amount=("amount", "sum"),
        tx_count=("amount", "size"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
    )

    for row in grouped.itertuples(index=False):
        graph.add_edge(
            str(row.source),
            str(row.target),
            amount=float(row.amount),
            tx_count=int(row.tx_count),
            first_seen=row.first_seen,
            last_seen=row.last_seen,
        )

    return graph


def trace_upstream(graph: nx.DiGraph, seeds: set[str], hops: int):
    """
    Идёт от известных клиентов против направления денег:
    находит счета, переводившие им средства, и их предшественников.
    """
    discovered = set(seeds)
    distance = {seed: 0 for seed in seeds}
    queue = deque((seed, 0) for seed in seeds)

    while queue:
        node, depth = queue.popleft()
        if depth >= hops:
            continue

        for predecessor in graph.predecessors(node):
            if predecessor not in discovered:
                discovered.add(predecessor)
                distance[predecessor] = depth + 1
                queue.append((predecessor, depth + 1))

    return discovered, distance


def scale_feature(values: pd.Series) -> pd.Series:
    """Нормализация признака в диапазон 0–1 после log1p."""
    values = pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0)
    transformed = values.map(lambda x: __import__("math").log1p(float(x)))

    low = transformed.min()
    high = transformed.max()
    if high == low:
        return pd.Series(0.0, index=values.index)

    return (transformed - low) / (high - low)


def analyze(graph: nx.DiGraph, seeds: set[str], hops: int):
    discovered, distance = trace_upstream(graph, seeds, hops)
    subgraph = graph.subgraph(discovered).copy()

    if not subgraph.nodes:
        return pd.DataFrame(), pd.DataFrame(), subgraph

    # Сколько известных клиентов достижимо из каждого узла по направлению денег.
    seed_reach = defaultdict(int)
    for seed in seeds:
        if seed in subgraph:
            for ancestor in nx.ancestors(subgraph, seed):
                seed_reach[ancestor] += 1

    metrics = []
    for node in subgraph.nodes:
        incoming = list(subgraph.in_edges(node, data=True))
        outgoing = list(subgraph.out_edges(node, data=True))

        total_in = sum(data.get("amount", 0) for _, _, data in incoming)
        total_out = sum(data.get("amount", 0) for _, _, data in outgoing)

        metrics.append(
            {
                "node": node,
                "in_degree": len(incoming),
                "out_degree": len(outgoing),
                "total_in": total_in,
                "total_out": total_out,
                "total_volume": total_in + total_out,
                "distance_to_seed": distance.get(node),
                "seeds_reached": seed_reach.get(node, 0),
                "is_seed": node in seeds,
            }
        )

    ranking = pd.DataFrame(metrics).set_index("node")

    # Центральность считается на найденной части графа.
    # На больших графах используется приближение для ускорения.
    if len(subgraph) > 500:
        k = min(200, len(subgraph) - 1)
        centrality = nx.betweenness_centrality(
            subgraph, k=k, normalized=True, seed=42
        )
    else:
        centrality = nx.betweenness_centrality(subgraph, normalized=True)

    ranking["betweenness"] = pd.Series(centrality)
    ranking["seed_coverage"] = ranking["seeds_reached"] / max(len(seeds), 1)
    ranking["proximity"] = ranking["distance_to_seed"].map(
        lambda d: 1 / (1 + d) if pd.notna(d) else 0
    )

    # Прозрачный эвристический балл от 0 до 100.
    # Веса — стартовая настройка, их следует калибровать на размеченных кейсах.
    ranking["score"] = 100 * (
        0.25 * ranking["seed_coverage"]
        + 0.20 * scale_feature(ranking["betweenness"])
        + 0.15 * scale_feature(ranking["in_degree"])
        + 0.15 * scale_feature(ranking["total_volume"])
        + 0.10 * scale_feature(ranking["out_degree"])
        + 0.15 * ranking["proximity"]
    )

    def infer_role(row):
        if row["is_seed"]:
            return "Известный клиент (seed)"

        in_degree = row["in_degree"]
        out_degree = row["out_degree"]

        if in_degree == 0 and out_degree > 0:
            return "Верхний источник в найденной части сети"
        if in_degree >= 2 and out_degree >= 2:
            return "Транзитный узел / возможный распределитель"
        if in_degree >= 2:
            return "Точка консолидации — гипотеза"
        if in_degree >= 1 and out_degree >= 1:
            return "Транзитный узел — гипотеза"
        if out_degree >= 2:
            return "Узел с несколькими исходящими переводами"
        return "Связанный узел — роль не определена"

    def explain(row):
        reasons = []
        if row["is_seed"]:
            reasons.append("входит в список известных клиентов")
        if row["seeds_reached"] > 0:
            reasons.append(
                f"по направлению переводов связан с "
                f"{int(row['seeds_reached'])} известными клиентами"
            )
        if row["in_degree"] >= 2:
            reasons.append(f"получает переводы от {int(row['in_degree'])} узлов")
        if row["out_degree"] >= 2:
            reasons.append(f"переводит средства {int(row['out_degree'])} узлам")
        if row["betweenness"] > 0:
            reasons.append("лежит на транзитных маршрутах графа")
        if not reasons:
            reasons.append("связан с найденной цепочкой")

        return "; ".join(reasons)

    ranking["role_hypothesis"] = ranking.apply(infer_role, axis=1)
    ranking["why"] = ranking.apply(explain, axis=1)
    ranking = ranking.sort_values("score", ascending=False)

    edges = pd.DataFrame(
        [
            {
                "source": source,
                "target": target,
                "amount": data.get("amount", 0),
                "tx_count": data.get("tx_count", 0),
                "first_seen": data.get("first_seen"),
                "last_seen": data.get("last_seen"),
            }
            for source, target, data in subgraph.edges(data=True)
        ]
    )

    if not edges.empty:
        edges = edges.sort_values("amount", ascending=False)

    return ranking, edges, subgraph


def make_graph_plot(subgraph, ranking, seeds, top_n=60):
    """Рисует небольшой обзорный фрагмент графа."""
    if subgraph.number_of_nodes() == 0:
        return None

    selected = set(ranking.head(top_n).index) | set(seeds)
    view = subgraph.subgraph(selected).copy()

    # Убираем из визуализации изолированные узлы, если они появились.
    if view.number_of_nodes() == 0:
        return None

    pos = nx.spring_layout(view, seed=7, k=1.0 / max(len(view) ** 0.5, 1))

    edge_x, edge_y = [], []
    for u, v in view.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="#9aa0a6"),
        hoverinfo="none",
    )

    score_by_node = ranking["score"].to_dict()
    node_x, node_y, labels, colors, sizes, hover = [], [], [], [], [], []

    for node in view.nodes():
        x, y = pos[node]
        row = ranking.loc[node]

        node_x.append(x)
        node_y.append(y)
        labels.append(str(node))
        colors.append("#e45756" if node in seeds else "#4c78a8")
        sizes.append(12 + min(float(row["score"]), 100) * 0.18)
        hover.append(
            f"ID: {node}<br>"
            f"Баллы: {score_by_node.get(node, 0):.1f}<br>"
            f"Роль-гипотеза: {row['role_hypothesis']}<br>"
            f"Причина: {row['why']}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        hovertext=hover,
        hoverinfo="text",
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=1, color="white"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


uploaded = st.file_uploader("Загрузите CSV с транзакциями", type=["csv"])

if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded, dtype={"source": "string", "target": "string"})
        transactions = normalize_transactions(raw_df)

        known_nodes = set(transactions["source"]) | set(transactions["target"])

        st.sidebar.header("Параметры анализа")
        seed_text = st.sidebar.text_area(
            "Известные клиенты — по одному ID на строку",
            placeholder="A001\nB002\nC003",
            height=180,
        )
        hops = st.sidebar.slider(
            "Глубина трассировки вверх по цепочке",
            min_value=1,
            max_value=8,
            value=4,
        )

        seeds = {
            value.strip()
            for value in seed_text.splitlines()
            if value.strip()
        }
        found_seeds = seeds & known_nodes
        missing_seeds = seeds - known_nodes

        if missing_seeds:
            st.warning(
                "Не найдены в транзакциях: "
                + ", ".join(sorted(missing_seeds)[:20])
            )

        if not found_seeds:
            st.info("Введите ID известных клиентов, которые есть в CSV.")
            st.stop()

        graph = build_graph(transactions)
        ranking, edges, subgraph = analyze(graph, found_seeds, hops)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Транзакций в данных", f"{len(transactions):,}")
        col2.metric("Узлов в исходном графе", f"{graph.number_of_nodes():,}")
        col3.metric("Найдено узлов вокруг seeds", f"{subgraph.number_of_nodes():,}")
        col4.metric("Известных клиентов в анализе", f"{len(found_seeds):,}")

        st.subheader("Кого смотреть первым")
        display_cols = [
            "score",
            "role_hypothesis",
            "distance_to_seed",
            "seeds_reached",
            "in_degree",
            "out_degree",
            "total_in",
            "total_out",
            "why",
        ]
        st.dataframe(
            ranking[display_cols].reset_index(names="node"),
            use_container_width=True,
            hide_index=True,
        )

        ranking_csv = ranking.reset_index(names="node").to_csv(
            index=False
        ).encode("utf-8-sig")
        st.download_button(
            "Скачать рейтинг CSV",
            data=ranking_csv,
            file_name="ranked_nodes.csv",
            mime="text/csv",
        )

        st.subheader("Фрагмент сети")
        top_n = st.slider(
            "Число наиболее приоритетных узлов на графе",
            min_value=10,
            max_value=100,
            value=40,
        )
        fig = make_graph_plot(subgraph, ranking, found_seeds, top_n=top_n)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Связи в найденной части сети")
        if edges.empty:
            st.write("Рёбра не найдены.")
        else:
            st.dataframe(edges, use_container_width=True, hide_index=True)
            edges_csv = edges.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Скачать рёбра CSV",
                data=edges_csv,
                file_name="traced_edges.csv",
                mime="text/csv",
            )

        st.caption(
            "Граф направлен: source → target соответствует движению денег. "
            "Трассировка выполняется против направления переводов от известных "
            "клиентов, максимум на заданное число шагов."
        )

    except Exception as exc:
        st.error(f"Не удалось обработать файл: {exc}")
else:
    st.info("Загрузите CSV. Минимальные колонки: source, target, amount.")
