#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
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
            st.write("Файлы в директории приложения:")
            st.write([p.name for p in base_dir.iterdir()])
            st.stop()

        df = pd.read_csv(path, sep=";")
        df["Year"] = year
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["%Female"] = df["%Female"].astype(str).str.replace(",", ".").astype(float)
    df["Areas"] = df["Areas"].str.split(";")
    df = df.explode("Areas")
    df["Areas"] = df["Areas"].str.strip()
    df = df.dropna(subset=["%Female", "Areas", "Year"])
    return df

# -------------------------------------------------
# Вспомогательные функции
# -------------------------------------------------
def wrap_label(label, width=25):
    return textwrap.fill(label, width=width)

def apply_gradient(ax, start_color="#a8cfff", end_color="#08306b"):
    """Градиентный фон слева направо"""
    ax.set_facecolor("none")
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(
        gradient,
        aspect='auto',
        cmap=mpl.colors.LinearSegmentedColormap.from_list("custom_blue", [start_color, end_color]),
        extent=[0, 1, 0, 1],
        transform=ax.transAxes,
        zorder=-1
    )

def plot_boxplot_top_areas(df, year, top_n=10):
    df_year = df[df["Year"] == year]
    top_areas = df_year.groupby("Areas")["%Female"].median().sort_values(ascending=False).head(top_n)
    df_top = df_year[df_year["Areas"].isin(top_areas.index)]
    df_top["Areas"] = pd.Categorical(df_top["Areas"], categories=top_areas.index, ordered=True)

    grouped = [group["%Female"].values for _, group in df_top.groupby("Areas")]
    labels = list(top_areas.index)

    fig, ax = plt.subplots(figsize=(12, 5))
    apply_gradient(ax)

    box_color = "darkblue"

    box = ax.boxplot(
        grouped,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="white", linewidth=2),
        whiskerprops=dict(color="white", linewidth=2),
        capprops=dict(color="white", linewidth=2),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="white", markeredgecolor="white", markersize=4, alpha=1.0),
        manage_ticks=False
    )

    # Белая граница для всех элементов
    for patch in box['boxes']:
        patch.set_edgecolor("white")
        patch.set_linewidth(2)
    for whisker in box['whiskers']:
        whisker.set_color("white")
        whisker.set_linewidth(2)
    for cap in box['caps']:
        cap.set_color("white")
        cap.set_linewidth(2)
    for flier in box['fliers']:
        flier.set_markeredgecolor("white")
        flier.set_markerfacecolor("white")
    for median in box['medians']:
        median.set_color("red")
        median.set_linewidth(2)

    ax.set_title(f"Распределение доли женщин-авторов (%Female)\nТоп-{top_n} Areas по медиане, {year}", color="white", fontsize=14)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(['']*len(labels))  # убираем стандартные подписи

    # Пояснения прямо под ящиками
    y_min = ax.get_ylim()[0] - 5
    for i, label in enumerate(labels):
        ax.text(i+1, y_min, label, ha='center', va='top', rotation=25, color='white', fontsize=10)

    ax.tick_params(axis="y", colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(2)

    plt.tight_layout()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.markdown("**Дополнительно справа:**")
        st.markdown("\n".join([f"- {i+1}. {area}" for i, area in enumerate(labels)]))
    plt.close(fig)

def plot_boxplot_by_quartile(df, year, area):
    df_area = df[(df["Year"] == year) & (df["Areas"] == area)]
    quartile_medians = df_area.groupby("SJR Best Quartile")["%Female"].median().sort_values(ascending=False)
    df_area["SJR Best Quartile"] = pd.Categorical(df_area["SJR Best Quartile"], categories=quartile_medians.index, ordered=True)

    grouped = [group["%Female"].values for _, group in df_area.groupby("SJR Best Quartile")]
    labels = list(quartile_medians.index)

    if not grouped:
        st.info("Нет данных для выбранного Area и года.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    apply_gradient(ax)

    box = ax.boxplot(
        grouped,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="white", linewidth=2),
        whiskerprops=dict(color="white", linewidth=2),
        capprops=dict(color="white", linewidth=2),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="white", markeredgecolor="white", markersize=4, alpha=1.0),
        manage_ticks=False
    )

    # Белая граница для всех элементов
    for patch in box['boxes']:
        patch.set_edgecolor("white")
        patch.set_linewidth(2)
    for whisker in box['whiskers']:
        whisker.set_color("white")
        whisker.set_linewidth(2)
    for cap in box['caps']:
        cap.set_color("white")
        cap.set_linewidth(2)
    for flier in box['fliers']:
        flier.set_markeredgecolor("white")
        flier.set_markerfacecolor("white")
    for median in box['medians']:
        median.set_color("red")
        median.set_linewidth(2)

    ax.set_title(f"%Female по квартилям\n{area}, {year}", color="white", fontsize=14)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(['']*len(labels))  # убираем стандартные подписи

    # Пояснения прямо под ящиками
    y_min = ax.get_ylim()[0] - 5
    for i, label in enumerate(labels):
        ax.text(i+1, y_min, label, ha='center', va='top', rotation=25, color='white', fontsize=10)

    ax.tick_params(axis="y", colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(2)

    plt.tight_layout()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.markdown(f"**Дополнительно справа:**")
        st.markdown("\n".join([f"- {i+1}. {quartile}" for i, quartile in enumerate(labels)]))
    plt.close(fig)

# -------------------------------------------------
# Основное приложение
# -------------------------------------------------
def main():
    st.title("Анализ доли женщин-авторов по областям исследований")

    base_dir = Path(__file__).resolve().parent
    df = load_data(base_dir)

    year = st.selectbox("Выберите год", sorted(df["Year"].unique()))

    st.subheader("Топ Areas по медиане доли женщин")
    plot_boxplot_top_areas(df, year)

    areas_available = sorted(df[df["Year"] == year]["Areas"].unique())
    selected_area = st.selectbox("Выберите Area для детальной разбивки по квартилям", areas_available)

    st.subheader(f"Детализация по квартилям: {selected_area}")
    plot_boxplot_by_quartile(df, year, selected_area)


if __name__ == "__main__":
    main()
