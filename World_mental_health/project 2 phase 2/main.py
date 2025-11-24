"""
main.py

CSC111 Project 2 Main Entry Point

This script loads the global mental health data and runs all the core visualizations to analyze
the relationship between mental health resources and disorder prevalence across countries.
"""

from world_mental_health_project import Tree
import visualizations


def main() -> None:
    """Load the dataset and generate visualizations."""
    # Load the mental health data into a tree structure
    world_tree = Tree.load_mental_health_data('world_mental_health_data.json')

    # Run selected visualizations (uncomment as needed)
    visualizations.plot_depression_rates(world_tree)
    visualizations.plot_suicide_rates(world_tree)
    visualizations.plot_mental_health_beds(world_tree)
    visualizations.plot_psychologists(world_tree)
    visualizations.plot_mental_health_bed_ratio(world_tree)
    visualizations.plot_psych_to_suicide(world_tree)
    visualizations.plot_underserved_by_admissions(world_tree)
    visualizations.plot_total_workforce(world_tree)
    visualizations.plot_avg_admissions(world_tree)


if __name__ == '__main__':
    main()
