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

def wrap_label(label, width=25):
    return textwrap.fill(label, width=width)

# -------------------------------------------------
# Boxplot по топ Areas
# -------------------------------------------------
def plot_boxplot_top_areas(df, year, top_n=10):
    df_year = df[df["Year"] == year]
    top_areas = df_year.groupby("Areas")["%Female"].median().sort_values(ascending=False).head(top_n)
    df_top = df_year[df_year["Areas"].isin(top_areas.index)]
    df_top["Areas"] = pd.Categorical(df_top["Areas"], categories=top_areas.index, ordered=True)

    grouped = [group["%Female"].values for _, group in df_top.groupby("Areas")]
    labels = list(top_areas.index)

    fig, ax = plt.subplots(figsize=(10, 5))

    box = ax.boxplot(
        grouped,
        patch_artist=True,
        showfliers=True,  # показываем все выбросы
        boxprops=dict(facecolor="lightblue", color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=5, alpha=1.0),
        manage_ticks=False
    )

    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(['']*len(labels))  # подписи будем через текст

    # Пояснения под ящиками с переносом строк
    y_min = ax.get_ylim()[0] - 1
    for i, label in enumerate(labels):
        wrapped_label = textwrap.fill(label, width=20)  # перенос после 20 символов
        ax.text(i+1, y_min, wrapped_label, ha='center', va='top', rotation=25, fontsize=10)

    ax.set_title(f"Распределение доли женщин-авторов (%Female)\nТоп-{top_n} Areas, {year}", fontsize=14)
    ax.tick_params(axis="y", colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    plt.tight_layout()

    # График слева, пояснение справа
    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.markdown("**Описание графика:**")
        st.markdown(textwrap.fill(
            f"Этот график показывает распределение доли женщин-авторов (%Female) "
            f"по топ-{top_n} научным областям (Areas) за {year}. "
            "Каждый ящик — это квартильное распределение, красная линия — медиана, "
            "кружки — выбросы.",
            width=40
        ))
    plt.close(fig)

# -------------------------------------------------
# Boxplot по квартилям
# -------------------------------------------------
def plot_boxplot_by_quartile(df, year, area):
    df_area = df[(df["Year"] == year) & (df["Areas"] == area)]
    df_area = df_area[df_area["SJR Best Quartile"].isin(["Q1", "Q2", "Q3", "Q4"])]
    quartile_medians = df_area.groupby("SJR Best Quartile")["%Female"].median().sort_values(ascending=False)
    df_area["SJR Best Quartile"] = pd.Categorical(df_area["SJR Best Quartile"], categories=quartile_medians.index, ordered=True)

    grouped = [group["%Female"].values for _, group in df_area.groupby("SJR Best Quartile")]
    labels = list(quartile_medians.index)

    if not grouped:
        st.info("Нет данных для выбранного Area и года.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    box = ax.boxplot(
        grouped,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor="lightblue", color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=5, alpha=1.0),
        manage_ticks=False
    )

    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(['']*len(labels))

    # Пояснения под ящиками с переносом строк
    y_min = ax.get_ylim()[0] - 1
    for i, label in enumerate(labels):
        wrapped_label = textwrap.fill(label, width=20)
        ax.text(i+1, y_min, wrapped_label, ha='center', va='top', rotation=25, fontsize=10)

    ax.set_title(f"%Female по квартилям\n{area}, {year}", fontsize=14)
    ax.tick_params(axis="y", colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    plt.tight_layout()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.markdown("**Описание графика:**")
        st.markdown(textwrap.fill(
            f"Этот график показывает распределение доли женщин-авторов (%Female) "
            f"по квартилям журналов (Q1-Q4) для области {area} за {year}. "
            "Красная линия — медиана, кружки — выбросы.",
            width=40
        ))
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
