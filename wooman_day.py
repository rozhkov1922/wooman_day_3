#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path

# -------------------------------------------------
# Настройки страницы
# -------------------------------------------------
st.set_page_config(
    page_title="Gender authorship by research areas",
    layout="wide"
)

# -------------------------------------------------
# Загрузка данных
# -------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(base_dir: Path):
    files = {
        2022: "scimagojr 2022.csv",
        2023: "scimagojr 2023.csv",
        2024: "scimagojr 2024.csv",
    }

    dfs = []
    for year, filename in files.items():
        path = base_dir / filename
        if not path.exists():
            st.error(f"Файл не найден: {filename}")
            st.stop()

        df = pd.read_csv(path, sep=";")
        df["Year"] = year
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df["%Female"] = (
        df["%Female"]
        .astype(str)
        .str.replace(",", ".")
        .astype(float)
    )

    df["Areas"] = df["Areas"].str.split(";")
    df = df.explode("Areas")
    df["Areas"] = df["Areas"].str.strip()

    df = df.dropna(subset=["%Female", "Areas", "Year"])
    return df


# -------------------------------------------------
# Boxplot по Areas (топ 10)
# -------------------------------------------------
def plot_boxplot_top_areas(df, year, top_n=10, ascending=False):
    df_year = df[df["Year"] == year]

    areas_sorted = (
        df_year
        .groupby("Areas")["%Female"]
        .median()
        .sort_values(ascending=ascending)
        .head(top_n)
    )

    df_top = df_year[df_year["Areas"].isin(areas_sorted.index)]
    df_top["Areas"] = pd.Categorical(
        df_top["Areas"],
        categories=areas_sorted.index,
        ordered=True
    )

    grouped = [g["%Female"].values for _, g in df_top.groupby("Areas")]
    labels = list(areas_sorted.index)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.boxplot(
        grouped,
        showfliers=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", edgecolor="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5
        )
    )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([""] * len(labels))

    y_min = ax.get_ylim()[0] - 1
    for i, label in enumerate(labels):
        ax.text(
            i + 1,
            y_min,
            textwrap.fill(label, 20),
            ha="center",
            va="top",
            rotation=25,
            fontsize=10
        )

    ax.set_title(f"Топ-10 Areas по медиане %Female, {year}", fontsize=14)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    plt.tight_layout()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.markdown(
            "График показывает распределение доли женщин-авторов "
            "по 10 научным областям. "
            "Красная линия — медиана, точки — выбросы."
        )

    plt.close(fig)


# -------------------------------------------------
# Boxplot по квартилям (ЖЁСТКИЙ ПОРЯДОК Q4 → Q1)
# -------------------------------------------------
def plot_boxplot_by_quartile(df, year, area):
    df_area = df[
        (df["Year"] == year) &
        (df["Areas"] == area) &
        (df["SJR Best Quartile"].isin(["Q1", "Q2", "Q3", "Q4"]))
    ]

    quartile_order = ["Q4", "Q3", "Q2", "Q1"]

    df_area["SJR Best Quartile"] = pd.Categorical(
        df_area["SJR Best Quartile"],
        categories=quartile_order,
        ordered=True
    )

    grouped = [
        df_area[df_area["SJR Best Quartile"] == q]["%Female"].values
        for q in quartile_order
        if q in df_area["SJR Best Quartile"].values
    ]

    labels = [
        q for q in quartile_order
        if q in df_area["SJR Best Quartile"].values
    ]

    if not grouped:
        st.info("Нет данных для выбранной области.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.boxplot(
        grouped,
        showfliers=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", edgecolor="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5
        )
    )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    ax.set_title(f"%Female по квартилям (Q4 → Q1)\n{area}, {year}", fontsize=14)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    plt.tight_layout()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.markdown(
            "График показывает распределение доли женщин-авторов "
            "по квартилям журналов. "
            "Квартиля отсортированы строго по уровню: Q4 → Q1."
        )

    plt.close(fig)


# -------------------------------------------------
# Основное приложение
# -------------------------------------------------
def main():
    st.title("Анализ доли женщин-авторов")

    base_dir = Path(__file__).resolve().parent
    df = load_data(base_dir)

    years_sorted = sorted(df["Year"].unique(), reverse=True)
    year = st.selectbox("Год", years_sorted, index=0)

    order = st.selectbox(
        "Сортировка топ-10 Areas",
        ["По убыванию медианы", "По возрастанию медианы"]
    )
    ascending = order == "По возрастанию медианы"

    plot_boxplot_top_areas(df, year, ascending=ascending)

    areas = sorted(df[df["Year"] == year]["Areas"].unique())
    area = st.selectbox("Область (Area)", areas)

    plot_boxplot_by_quartile(df, year, area)


if __name__ == "__main__":
    main()
