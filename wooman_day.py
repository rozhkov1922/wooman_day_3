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

    # %Female → float
    df["%Female"] = (
        df["%Female"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    # Areas → explode
    df["Areas"] = df["Areas"].str.split(";")
    df = df.explode("Areas")
    df["Areas"] = df["Areas"].str.strip()

    # Чистка
    df = df.dropna(subset=["%Female", "Areas", "Year"])

    return df

# -------------------------------------------------
# Вспомогательные функции
# -------------------------------------------------
def wrap_label(label, width=25):
    return textwrap.fill(label, width=width)

def apply_gradient(ax, cmap_name="Blues"):
    """Применяет горизонтальный градиент на фон осей matplotlib"""
    from matplotlib.colors import LinearSegmentedColormap

    ax.set_facecolor("none")
    fig = ax.figure
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(
        gradient,
        aspect='auto',
        cmap=plt.get_cmap(cmap_name),
        extent=[0, 1, 0, 1],
        transform=ax.transAxes,
        zorder=-1
    )

def plot_boxplot_top_areas(df, year, top_n=10):
    df_year = df[df["Year"] == year]
    top_areas = (
        df_year.groupby("Areas")["%Female"]
        .median()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    df_top = df_year[df_year["Areas"].isin(top_areas)]

    grouped, labels = [], []
    for area, group in df_top.groupby("Areas"):
        grouped.append(group["%Female"].values)
        labels.append(wrap_label(area))

    # Разделение на колонки для графика и пояснения
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(16, 8))
        apply_gradient(ax)

        ax.boxplot(
            grouped,
            labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor="white", color="white"),
            whiskerprops=dict(color="white"),
            capprops=dict(color="white"),
            medianprops=dict(color="red"),
            flierprops=dict(
                marker="o",
                markerfacecolor="white",
                markeredgecolor="white",
                markersize=4,
                alpha=0.8,
            ),
        )

        ax.set_title(
            f"Распределение доли женщин-авторов (%Female)\n"
            f"Топ-{top_n} Areas по медиане, {year}",
            color="white",
            fontsize=15,
        )
        ax.tick_params(axis="x", colors="white", rotation=22)
        ax.tick_params(axis="y", colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("""
        **Пояснения:**
        - Ящики показывают диапазон с 25-го по 75-й процентиль (%Female).  
        - Красная линия – медиана.  
        - Кружки – выбросы.  
        - Топ Areas выбраны по медианной доле женщин-авторов.  
        - Градиентный фон показывает визуально распределение графика.
        """)

def plot_boxplot_by_quartile(df, year, area):
    df_area = df[(df["Year"] == year) & (df["Areas"] == area)]

    grouped, labels = [], []
    for quartile, group in df_area.groupby("SJR Best Quartile"):
        grouped.append(group["%Female"].values)
        labels.append(quartile)

    if not grouped:
        st.info("Нет данных для выбранного Area и года.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        apply_gradient(ax)

        # Только ящички с квартилями
        ax.boxplot(
            grouped,
            labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="blue"),
            whiskerprops=dict(color="blue"),
            capprops=dict(color="blue"),
            medianprops=dict(color="red"),
            flierprops=dict(
                marker="o",
                markerfacecolor="blue",
                markeredgecolor="blue",
                markersize=4,
                alpha=0.7,
            ),
        )

        ax.set_title(f"%Female по квартилям\n{area}, {year}", color="white")
        ax.set_facecolor("none")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("""
        **Пояснения по квартилям:**
        - Ящики показывают диапазон 25%-75% доли женщин-авторов.  
        - Красная линия – медиана.  
        - Кружки – выбросы.  
        - Позволяет сравнить качество журналов по квартилям для выбранной области.
        """)

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
    selected_area = st.selectbox(
        "Выберите Area для детальной разбивки по квартилям",
        areas_available,
    )

    st.subheader(f"Детализация по квартилям: {selected_area}")
    plot_boxplot_by_quartile(df, year, selected_area)


if __name__ == "__main__":
    main()
