"""
CSC111 Project 2 Final Phase: World Mental Health Data Tree

===============================

This module defines a Tree data structure and functions for recursively organizing and displaying
world mental health data. The tree structure represents the hierarchical relationships in the dataset,
with the "World" as the root, continents/regions as subtrees, and individual countries as leaves.
Each country contains detailed mental health statistics stored as part of its node.

The goal of this module is to enable recursive processing and visualization of nested global health data,
demonstrating hierarchical data modeling through recursive trees.
"""

from __future__ import annotations
import json
from typing import Optional, Any, List, Tuple
from statistics import pstdev, mean


class Tree:
    """A recursive tree data structure for the Mental Health Project.

    Representation Invariants:
        - self._root is not None or self._subtrees == []
    """

    _root: Optional[Any]
    _subtrees: List[Tree]

    def __init__(self, root: Optional[Any], subtrees: List[Tree]) -> None:
        """Initialize a new Tree with the given root value and subtrees.

        If root is None, the tree is empty.

        Preconditions:
            - root is not None or subtrees == []
        """
        self._root = root
        self._subtrees = subtrees

    @staticmethod
    def load_mental_health_data(filename: str) -> Tree:
        """Load mental health tree data from a JSON file and return a Tree instance.

        Preconditions:
            - filename is a valid JSON file containing world mental health data.

        >>> world_tree_ex = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> isinstance(world_tree_ex, Tree)
        True
        >>> not world_tree_ex.is_empty()
        True
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        root_name = data['world']['name']
        regions = data['world']['regions']

        region_subtrees = []
        for region_name, countries_dict in regions.items():
            country_subtrees = []
            for country_name, country_data in countries_dict.items():
                # Each country is a leaf with (country name, country data) as root
                country_subtrees.append(Tree((country_name, country_data), []))
            # Each region contains countries as subtrees
            region_subtrees.append(Tree(region_name, country_subtrees))

        return Tree(root_name, region_subtrees)

    def is_empty(self) -> bool:
        """Return whether this tree is empty."""
        return self._root is None

    def __str__(self) -> str:
        """Return a string representation of this tree."""
        return self._str_indented(0)

    def _str_indented(self, depth: int = 0) -> str:
        """Return an indented string representation of this tree (recursive helper)."""
        if self.is_empty():
            return ''
        else:
            str_so_far = '  ' * depth + f'{self._root}\n'
            for subtree in self._subtrees:
                str_so_far += subtree._str_indented(depth + 1)
            return str_so_far

    def avg_depression_by_region(self) -> dict[str, float]:
        """Return a dictionary mapping each region to its average depression rate per 100k people.

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> result = tree.avg_depression_by_region()
        >>> isinstance(result, dict)
        True
        >>> 'North America' in result
        True
        """
        if self.is_empty():
            return {}

        if isinstance(self._root, tuple):
            return {}

        region_depression_rates = {}

        for subtree in self._subtrees:
            region = subtree._root
            total_depression = 0
            country_count = 0

            for country_tree in subtree._subtrees:
                country_data = country_tree._root[1]
                total_depression += country_data.get("depression_rate_per_100k", 0)
                country_count += 1

            region_depression_rates[region] = round(total_depression / country_count, 2) if country_count > 0 else 0.0

        return region_depression_rates

    def max_min_suicide_rates(self) -> dict[str, tuple[tuple[str, float], tuple[str, float]]]:
        """Return a dictionary mapping each region to its country with the highest and lowest suicide rate.

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> result = tree.max_min_suicide_rates()
        >>> isinstance(result, dict)
        True
        >>> isinstance(result['Europe'], tuple)
        True
        """
        if self.is_empty():
            return {}

        if isinstance(self._root, tuple):
            return {}

        region_suicide_rates = {}

        for subtree in self._subtrees:
            region = subtree._root
            max_country, min_country = ('', float('-inf')), ('', float('inf'))

            for country_tree in subtree._subtrees:
                country_name, country_data = country_tree._root
                suicide_rate = country_data.get("suicide_rate_per_100k", 0)

                if suicide_rate > max_country[1]:
                    max_country = (country_name, suicide_rate)
                if suicide_rate < min_country[1]:
                    min_country = (country_name, suicide_rate)

            region_suicide_rates[region] = (max_country, min_country)

        return region_suicide_rates

    def average_beds_per_region(self) -> dict[str, float]:
        """Return a dictionary mapping each region to its average number of beds in mental hospitals.

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> result = tree.average_beds_per_region()
        >>> isinstance(result, dict)
        True
        >>> 'Europe' in result
        True
        """
        if self.is_empty():
            return {}

        if isinstance(self._root, tuple):
            return {}

        region_beds = {}

        for subtree in self._subtrees:
            region = subtree._root
            total_beds = 0
            country_count = 0

            for country_tree in subtree._subtrees:
                country_data = country_tree._root[1]
                total_beds += country_data.get("beds_in_mental_hospitals_per_100k", 0)
                country_count += 1

            region_beds[region] = round(total_beds / country_count, 2) if country_count > 0 else 0.0

        return region_beds

    def avg_psychologists_by_region(self) -> dict[str, float]:
        """Return a dictionary mapping each region to its average number of psychologists per 100k people.

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> result = tree.avg_psychologists_by_region()
        >>> isinstance(result, dict)
        True
        >>> isinstance(result.get('South America'), float)
        True
        """
        if self.is_empty():
            return {}

        if isinstance(self._root, tuple):
            return {}

        region_psychologists = {}

        for subtree in self._subtrees:
            region = subtree._root
            total_psychologists = 0
            country_count = 0

            for country_tree in subtree._subtrees:
                country_data = country_tree._root[1]
                total_psychologists += country_data.get("psychologists_per_100k", 0)
                country_count += 1

            region_psychologists[region] = round(total_psychologists / country_count, 2) if country_count > 0 else 0.0

        return region_psychologists

    def bed_to_depression_ratio(self) -> list[tuple[str, float]]:
        """Return the ratio of mental health beds to depression rates per country.
        The ratio is calculated as:
        beds in mental hospitals / depression rate

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> result = tree.bed_to_depression_ratio()
        >>> isinstance(result, list)
        True
        """
        if self.is_empty():
            return []
        if isinstance(self._root, tuple):
            country_name = self._root[0]
            beds = self._root[1].get("beds_in_mental_hospitals_per_100k", 0)
            depression_rate = self._root[1].get("depression_rate_per_100k", 1)  # Avoid division by zero
            ratio = round(beds / depression_rate, 4) if depression_rate > 0 else 0
            return [(country_name, ratio)]

        ratios = []
        for subtree in self._subtrees:
            ratios.extend(subtree.bed_to_depression_ratio())

        return ratios

    def ratio_of_beds_to_depression(self) -> List[Tuple[str, float]]:
        """
        Return a list of (country_name, ratio) for each country, where:
        ratio = beds_in_mental_hospitals_per_100k / depression_rate_per_100k

        This helps assess how much treatment capacity (beds) exists relative to need (depression).

        Recursively traverses all countries in the tree.

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> len(tree.ratio_of_beds_to_depression()) >= 10
        True
        """
        if self.is_empty():
            return []

        results = []
        if isinstance(self._root, tuple):
            country_name, data = self._root
            dep = data['depression_rate_per_100k']
            beds = data['beds_in_mental_hospitals_per_100k']
            if dep != 0:
                ratio = beds / dep
                results.append((country_name, ratio))

        for subtree in self._subtrees:
            results.extend(subtree.ratio_of_beds_to_depression())

        return results

    def psych_to_suicide_ratio(self) -> List[Tuple[str, float]]:
        """Return a list of (country_name, ratio) for each country, where
        ratio = (psychologists_per_100k) / (suicide_rate_per_100k).

        Recursively visits each country's data in the tree to compute the ratio.

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> isinstance(tree.psych_to_suicide_ratio()[0], tuple)
        True
        """
        if self.is_empty():
            return []

        results = []
        if isinstance(self._root, tuple):
            country_name, data = self._root
            suic = data['suicide_rate_per_100k']
            psych = data['psychologists_per_100k']
            if suic != 0:
                ratio = psych / suic
                results.append((country_name, ratio))

        for subtree in self._subtrees:
            results.extend(subtree.psych_to_suicide_ratio())

        return results

    def _compute_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Return the Pearson correlation coefficient between x_values and y_values.

        If there is insufficient variation or lengths don't match, return 0.0.
        """
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0

        mean_x = mean(x_values)
        mean_y = mean(y_values)
        std_x = pstdev(x_values)
        std_y = pstdev(y_values)

        if std_x == 0 or std_y == 0:
            return 0.0

        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        den = len(x_values) * std_x * std_y
        return num / den

    def corr_depression_outpatient(self, region: Optional[str] = None) -> float:
        """Return the correlation coefficient between depression rates and
        mental health outpatient facilities for the specified region.
        If region is None, compute for ALL countries across the world.

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> corr = tree.corr_depression_outpatient()
        >>> isinstance(corr, float)
        True
        >>> -1.0 <= corr <= 1.0
        True
        >>> corr_region = tree.corr_depression_outpatient('Asia')
        >>> -1.0 <= corr_region <= 1.0
        True
        """
        depression_rates = []
        outpatient_facilities = []

        self.gather_depression_outpatient(depression_rates, outpatient_facilities, region)

        return self._compute_correlation(depression_rates, outpatient_facilities)

    def gather_depression_outpatient(self, dep_list: List[float], out_list: List[float], region: Optional[str]) -> None:
        """Helper method to recursively collect depression rates and outpatient
        facilities for either the entire tree or a specified region."""
        if self.is_empty():
            return

        if isinstance(self._root, str) and (
                self._root == region or region is None or self._root == 'World Mental Health Data'):
            for subtree in self._subtrees:
                subtree.gather_depression_outpatient(dep_list, out_list, region)

        elif isinstance(self._root, tuple):
            if region is None:
                dep_list.append(self._root[1]['depression_rate_per_100k'])
                out_list.append(self._root[1]['mental_health_outpatient_facilities_per_100k'])
            else:
                dep_list.append(self._root[1]['depression_rate_per_100k'])
                out_list.append(self._root[1]['mental_health_outpatient_facilities_per_100k'])

        else:
            pass

    def corr_anxiety_occupational(self, region: Optional[str] = None) -> float:
        """Return the correlation coefficient between anxiety rates and
        occupational therapists for the specified region.
        If region is None, compute for ALL countries across the world.

        Purpose:
            Determine if therapist availability aligns (positively or negatively)
            with anxiety rates.

        Fields used:
            - anxiety_rate_per_100k
            - occupational_therapists_per_100k

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> corr = tree.corr_anxiety_occupational()
        >>> isinstance(corr, float)
        True
        >>> -1.0 <= corr <= 1.0
        True
        """
        anxiety_rates = []
        occupational_list = []

        self.gather_anxiety_occupational(anxiety_rates, occupational_list, region)

        return self._compute_correlation(anxiety_rates, occupational_list)

    def gather_anxiety_occupational(self, anx_list: List[float], occ_list: List[float], region: Optional[str]) -> None:
        """Helper method to recursively collect anxiety rates and occupational
        therapist availability for either the entire tree or a specified region."""
        if self.is_empty():
            return

        if isinstance(self._root, str) and (
                self._root == region or region is None or self._root == 'World Mental Health Data'):
            for subtree in self._subtrees:
                subtree.gather_anxiety_occupational(anx_list, occ_list, region)

        elif isinstance(self._root, tuple):
            if region is None:
                anx_list.append(self._root[1]['anxiety_rate_per_100k'])
                occ_list.append(self._root[1]['occupational_therapists_per_100k'])
            else:
                anx_list.append(self._root[1]['anxiety_rate_per_100k'])
                occ_list.append(self._root[1]['occupational_therapists_per_100k'])

    def corr_admissions_suicide(self, region: Optional[str] = None) -> float:
        """Return the correlation coefficient between mental_hospital_admissions_per_100k
        and suicide_rate_per_100k for the specified region.
        If region is None, compute for ALL countries across the world

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> corr = tree.corr_admissions_suicide()
        >>> isinstance(corr, float)
        True
        >>> -1.0 <= corr <= 1.0
        True
        """
        admissions = []
        suicides = []

        self.gather_adm_and_suicide(admissions, suicides, region)

        return self._compute_correlation(admissions, suicides)

    def gather_adm_and_suicide(self, adm_list: List[float], sui_list: List[float], region: Optional[str]) -> None:
        """Helper method to recursively collect hospital admissions and suicide rates for
        either the entire tree or a specified region."""
        if self.is_empty():
            return

        if isinstance(self._root, str) and (
                self._root == region or region is None or self._root == 'World Mental Health Data'):
            for subtree in self._subtrees:
                subtree.gather_adm_and_suicide(adm_list, sui_list, region)

        elif isinstance(self._root, tuple):
            if region is None:
                adm_list.append(self._root[1]['mental_hospital_admissions_per_100k'])
                sui_list.append(self._root[1]['suicide_rate_per_100k'])
            else:
                adm_list.append(self._root[1]['mental_hospital_admissions_per_100k'])
                sui_list.append(self._root[1]['suicide_rate_per_100k'])

    def day_treatment_facilities(self) -> List[Tuple[str, float, float]]:
        """Return a list of (region_name, total_facilities, average_facilities) where
        facilities represent mental health outpatient facilities (treated as day treatment).
        - total_facilities is the sum across all countries in that region
        - average_facilities is total_facilities / number of countries in the region

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> dist = tree.day_treatment_facilities()
        >>> isinstance(dist, list)
        True
        >>> isinstance(dist[0], tuple) and len(dist[0]) == 3
        True
        """
        if self.is_empty():
            return []

        results = []

        if isinstance(self._root, str) and self._root not in {'World Mental Health Data', None}:
            total_facilities = 0.0
            country_count = 0

            for subtree in self._subtrees:
                if isinstance(subtree._root, tuple):
                    total_facilities += subtree._root[1]['mental_health_outpatient_facilities_per_100k']
                    country_count += 1

            if country_count > 0:
                avg_facilities = total_facilities / country_count
                results.append((self._root, total_facilities, round(avg_facilities, 2)))

        for subtree in self._subtrees:
            results.extend(subtree.day_treatment_facilities())

        return results

    def high_depression_low_hospitals(self, depression_threshold: float, hospital_threshold: float) -> List[str]:
        """Return a list of countries where depression rates exceed depression_threshold but mental hospitals are
        below hospital_threshold.

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> tree.high_depression_low_hospitals(5000, 0.1)
        ['Australia']
        >>> tree.high_depression_low_hospitals(4500, 0.2)
        ['United States', 'Mexico', 'Brazil', 'France', 'South Africa', 'Australia']
        >>> tree.high_depression_low_hospitals(6000, 0.2)
        []
        """
        if self.is_empty():
            return []

        results = []
        if isinstance(self._root, tuple):
            country_name, data = self._root
            if (
                    data['depression_rate_per_100k'] > depression_threshold
                    and data['mental_hospitals_per_100k'] < hospital_threshold
            ):
                results.append(country_name)

        for subtree in self._subtrees:
            results.extend(subtree.high_depression_low_hospitals(depression_threshold, hospital_threshold))

        return results

    def rank_regions_by_hospital_units(self, region: Optional[str] = None) -> List[tuple]:
        """Return a list of countries in a region ranked by mental health units in general hospitals.
        If no region is specified, return the ranking for all countries.

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> tree.rank_regions_by_hospital_units('Asia')
        [('India', 0.03), ('China', 0.18), ('South Korea', 0.39)]
        >>> tree.rank_regions_by_hospital_units('Africa')
        [('Egypt', 0.01), ('South Africa', 0.07)]
        """
        if self.is_empty():
            return []

        country_averages = []

        for subtree in self._subtrees:  # Regions
            if region is None or subtree._root == region:
                for country_tree in subtree._subtrees:  # Countries
                    country_name, data = country_tree._root
                    country_averages.append(
                        (country_name, round(data['mental_health_units_in_general_hospitals_per_100k'], 2)))

        return sorted(country_averages, key=lambda x: x[1])

    def _get_country_need_ratios(self, region_tree: Tree) -> List[Tuple[str, float]]:
        """Helper to compute (country_name, (depression + anxiety) / hospital admissions) within a region."""
        ratios = []
        for country_tree in region_tree._subtrees:
            country_name, data = country_tree._root
            total_need = data['depression_rate_per_100k'] + data['anxiety_rate_per_100k']
            total_admissions = data['mental_hospital_admissions_per_100k']
            if total_admissions > 0:
                ratio = round(total_need / total_admissions, 2)
                ratios.append((country_name, ratio))
        return ratios

    def underserved_by_admissions(self, region: Optional[str] = None) -> List[tuple]:
        """Return a list of countries ranked by the highest ratio of (Depression + Anxiety rates) / Hospital Admissions.
        If a region is specified, return only countries within that region.

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> tree.underserved_by_admissions('Asia')
        [('India', 441.12), ('China', 82.57), ('South Korea', 31.84)]
        >>> tree.underserved_by_admissions('North America')
        [('United States', 225.11), ('Canada', 191.85), ('Mexico', 148.28)]
        """
        if self.is_empty():
            return []

        country_ratios = []

        for region_tree in self._subtrees:
            if region is None or region_tree._root == region:
                country_ratios.extend(self._get_country_need_ratios(region_tree))

        return sorted(country_ratios, key=lambda x: x[1], reverse=True)

    def total_mental_health_workforce(self) -> List[Tuple[str, float]]:
        """
        Return a list of (region_name, total_workforce), where total_workforce is the combined number
        of occupational therapists and psychologists per 100k people in that region.

        The list is sorted in descending order by workforce size.

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> result = tree.total_mental_health_workforce()
        >>> isinstance(result, list)
        True
        """
        if self.is_empty():
            return []

        workforce_counts = []

        for region_tree in self._subtrees:
            region_name = region_tree._root
            total_workforce = 0.0
            country_count = 0

            for country_tree in region_tree._subtrees:
                if isinstance(country_tree._root, tuple):
                    data = country_tree._root[1]
                    total_workforce += (
                        data.get('occupational_therapists_per_100k', 0)
                        + data.get('psychologists_per_100k', 0)
                    )
                    country_count += 1

            if country_count > 0:
                workforce_counts.append((region_name, round(total_workforce, 2)))

        return sorted(workforce_counts, key=lambda x: x[1], reverse=True)

    def avg_general_hosp_admissions(self) -> List[Tuple[str, float]]:
        """
        Return a list of (region_name, average_admissions), where average_admissions is the
        average number of admissions to general hospitals for mental health per 100k people.

        The result is sorted in descending order of average admissions.

        >>> tree = Tree.load_mental_health_data('world_mental_health_data.json')
        >>> result = tree.avg_general_hosp_admissions()
        >>> isinstance(result, list)
        True
        """
        if self.is_empty():
            return []

        region_admissions = []

        for subtree in self._subtrees:  # Regions
            total_admissions = 0.0
            country_count = 0

            for country_tree in subtree._subtrees:  # Countries
                _, data = country_tree._root
                total_admissions += data['mental_health_units_in_general_hospitals_admissions_per_100k']
                country_count += 1

            if country_count > 0:
                average = round(total_admissions / country_count, 2)
                region_admissions.append((subtree._root, average))

        return sorted(region_admissions, key=lambda x: x[1], reverse=True)


if __name__ == '__main__':

    world_tree = Tree.load_mental_health_data('world_mental_health_data.json')
    print(world_tree)

    import python_ta
    python_ta.check_all(config={
        'max-line-length': 120,
        'disable': ['R1705', 'E9998', 'E9999']
    })
