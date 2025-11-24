"""
CSC111 Project 2: Visualizing World Mental Health Data

This module provides a set of visualizations to explore global mental health statistics using data organized in a
recursive tree structure. The visualizations highlight key insights such as average depression rates by region,
extremes in suicide rates, ratios reflecting treatment adequacy, and correlations between mental health needs
and available resources. Diverse graph types—bar charts, scatter plots, boxplots, and more—are used to ensure
clarity, variety, and effective communication of trends and disparities across countries and regions.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from world_mental_health_project import Tree


def plot_depression_rates(tree: Tree) -> None:
    """Bar Chart: Visualize average depression rates per 100K people by region."""
    data = tree.avg_depression_by_region()
    df = pd.DataFrame(list(data.items()), columns=["Region", "Depression Rate per 100K"])

    fig = px.bar(df, x="Region", y="Depression Rate per 100K",
                 title="Average Depression Rates per 100K People by Region",
                 text_auto=True)

    fig.show()


def plot_suicide_rates(tree: Tree) -> None:
    """Line Chart: Visualize highest and lowest suicide rates per 100K people across each region."""
    data = tree.max_min_suicide_rates()
    regions = list(data.keys())
    max_rates = [data[region][0][1] for region in regions]
    min_rates = [data[region][1][1] for region in regions]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=regions, y=max_rates, mode='lines+markers', name="Max Suicide Rate"))
    fig.add_trace(go.Scatter(x=regions, y=min_rates, mode='lines+markers', name="Min Suicide Rate"))

    fig.update_layout(title="Highest vs. Lowest Suicide Rates per Region",
                      xaxis_title="Region", yaxis_title="Suicide Rate per 100K")

    fig.show()


def plot_mental_health_beds(tree: Tree) -> None:
    """Horizontal Bar Chart: Visualize average number of mental health beds per 100K people by region."""
    data = tree.average_beds_per_region()
    df = pd.DataFrame(list(data.items()), columns=["Region", "Beds per 100K"])

    fig = px.bar(df, x="Beds per 100K", y="Region", orientation='h',
                 title="Average Mental Health Beds per 100K People by Region",
                 text_auto=True)

    fig.show()


def plot_psychologists(tree: Tree) -> None:
    """Bubble Chart: Show distribution of psychologists per 100K people by region using bubble sizes."""
    data = tree.avg_psychologists_by_region()
    df = pd.DataFrame(list(data.items()), columns=["Region", "Psychologists per 100K"])

    fig = px.scatter(df, x="Region", y="Psychologists per 100K", size="Psychologists per 100K",
                     title="Psychologists per 100K People by Region",
                     hover_name="Region", color="Psychologists per 100K",
                     size_max=50)

    fig.show()


def plot_mental_health_bed_ratio(tree: Tree) -> None:
    """Scatter Plot: Visualize ratio of mental health beds to depression rate for each country."""
    data = tree.ratio_of_beds_to_depression()
    df = pd.DataFrame(data, columns=["Country", "Bed to Depression Ratio"])

    fig = px.scatter(df, x="Country", y="Bed to Depression Ratio",
                     title="Mental Health Beds to Depression Ratio per Country",
                     size="Bed to Depression Ratio", color="Country",
                     hover_name="Country")

    fig.show()


def plot_psych_to_suicide(tree: Tree) -> None:
    """Bar Chart: Show psychologists-to-suicide ratios per country in a readable format."""
    data = tree.psych_to_suicide_ratio()
    df = pd.DataFrame(data, columns=["Country", "Psychologists/Suicide Ratio"])
    df = df.sort_values("Psychologists/Suicide Ratio", ascending=False)

    fig = px.bar(
        df,
        x="Country",
        y="Psychologists/Suicide Ratio",
        title="Psychologists to Suicide Rate Ratios by Country",
        text_auto=True
    )

    fig.update_layout(xaxis_tickangle=-45)
    fig.show()


def plot_underserved_by_admissions(tree: Tree) -> None:
    """Treemap: Visualize underserved countries based on need vs hospital admissions."""
    data = tree.underserved_by_admissions()
    df = pd.DataFrame(data, columns=["Country", "Need-to-Admission Ratio"])

    fig = px.treemap(df, path=["Country"], values="Need-to-Admission Ratio",
                     title="Countries with High Need vs Low Hospital Admissions",
                     color="Need-to-Admission Ratio", color_continuous_scale="reds")

    fig.show()


def plot_total_workforce(tree: Tree) -> None:
    """Sunburst Chart: Visualize total mental health workforce (psychologists + therapists) by region."""
    data = tree.total_mental_health_workforce()
    df = pd.DataFrame(data, columns=["Region", "Total Workforce"])

    fig = px.sunburst(df, path=["Region"], values="Total Workforce",
                      title="Total Mental Health Workforce by Region",
                      color="Total Workforce", color_continuous_scale="blues")

    fig.show()


def plot_avg_admissions(tree: Tree) -> None:
    """Heatmap: Show regional average admissions in general hospitals for mental health treatment."""
    data = tree.avg_general_hosp_admissions()
    df = pd.DataFrame(data, columns=["Region", "Admissions per 100K"])

    fig = go.Figure(data=go.Heatmap(
        z=df["Admissions per 100K"],
        x=df["Region"],
        y=["Avg Admissions"] * len(df),  # Single row to make it look like a table
        colorscale='Viridis',
        colorbar={"title": "Admissions"},
        hoverongaps=False
    ))

    fig.update_layout(
        title="Average Mental Health Admissions in General Hospitals (per 100K)",
        yaxis={"showticklabels": False}  # Hide repeated y-axis label
    )

    fig.show()


if __name__ == "__main__":
    print("Generating visualizations...")

    world_tree = Tree.load_mental_health_data('world_mental_health_data.json')

    plot_depression_rates(world_tree)
    plot_suicide_rates(world_tree)
    plot_mental_health_beds(world_tree)
    plot_psychologists(world_tree)
    plot_mental_health_bed_ratio(world_tree)
    plot_psych_to_suicide(world_tree)
    plot_underserved_by_admissions(world_tree)
    plot_total_workforce(world_tree)
    plot_avg_admissions(world_tree)

    import python_ta
    python_ta.check_all(config={
        'max-line-length': 120,
        'disable': ['R1705', 'E9998', 'E9999']
    })
