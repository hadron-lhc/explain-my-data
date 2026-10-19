"""Tests de enriquecimiento de mensajes (P5).

- Asociaciones numéricas: R² y nota de no-linealidad.
- Diferencias de grupo: ratio extremos y n por grupo extremo.
- Asimetría: cinco números resumen.
- Asociaciones cat-cat: ranking corto de combos sobrerrepresentados.
- Cross-links: deterministas, con tope, sin quality.
"""

import pandas as pd
import pytest

from src.distributions import NumericDistribution, calculate_numeric_distribution
from src.insights import (
    Insight,
    apply_cross_links,
    generate_distribution_insights,
    generate_relationship_insights,
)
from src.relationships import (
    CategoricalCategoricalRelationship,
    CategoricalNumericRelationship,
    NumericRelationship,
)


class TestNumericAssociationEnrichment:
    def test_r_squared_present(self):
        relationship = NumericRelationship(
            column_x="age",
            column_y="fare",
            pearson=0.70,
            spearman=0.70,
            n_observations=800,
        )

        insights = generate_relationship_insights([relationship], [])

        assert len(insights) == 1
        assert "R²: 0.49" in insights[0].message
        assert insights[0].evidence["r_squared"] == pytest.approx(0.49)

    def test_nonlinear_note_when_spearman_materially_higher(self):
        relationship = NumericRelationship(
            column_x="age",
            column_y="fare",
            pearson=0.55,
            spearman=0.85,
            n_observations=800,
        )

        insights = generate_relationship_insights([relationship], [])

        assert "non-linear" in insights[0].message

    def test_no_nonlinear_note_when_both_match(self):
        relationship = NumericRelationship(
            column_x="age",
            column_y="fare",
            pearson=0.90,
            spearman=0.92,
            n_observations=800,
        )

        insights = generate_relationship_insights([relationship], [])

        assert "non-linear" not in insights[0].message


class TestGroupDifferenceEnrichment:
    def make_relationship(self, **overrides) -> CategoricalNumericRelationship:
        data = {
            "column_categorical": "region",
            "column_numeric": "sales",
            "group_means": {"north": 10.0, "south": 2.0},
            "group_medians": {"north": 9.0, "south": 2.0},
            "group_counts": {"north": 100, "south": 50},
            "n_observations": 150,
            "group_separation": 1.8,
            "n_groups": 2,
            "group_outliers": {},
            "coverage_pct": 100.0,
        }
        data.update(overrides)
        return CategoricalNumericRelationship(**data)

    def test_ratio_and_group_sizes_present(self):
        insights = generate_relationship_insights([], [self.make_relationship()])

        assert len(insights) == 1
        message = insights[0].message
        assert "5.0x the bottom group" in message
        assert "(n: north 100, south 50)" in message
        assert insights[0].evidence["top_bottom_ratio"] == pytest.approx(5.0)

    def test_no_ratio_phrase_when_groups_close(self):
        relationship = self.make_relationship(
            group_means={"north": 10.0, "south": 8.0},
        )

        insights = generate_relationship_insights([], [relationship])

        assert "the bottom group" not in insights[0].message
        # El n por grupo extremo sí se emite siempre.
        assert "(n: north 100, south 50)" in insights[0].message


class TestSkewEnrichment:
    def test_five_number_summary_present(self):
        distribution = NumericDistribution(
            column_name="income",
            n_observations=1000,
            skewness=2.4,
            kurtosis=8.0,
            minimum=0.0,
            q1=10.0,
            median=20.0,
            q3=40.0,
            maximum=500.0,
        )

        insights = generate_distribution_insights([distribution])

        assert len(insights) == 1
        assert "Range: 0.00 to 500.00" in insights[0].message
        assert "Q1 10.00" in insights[0].message

    def test_summary_computed_from_series(self):
        series = pd.Series([0.0, 1.0, 2.0, 3.0, 100.0] * 4)
        distribution = calculate_numeric_distribution(series, "income")

        assert distribution is not None
        assert distribution.minimum == 0.0
        assert distribution.median == 2.0
        assert distribution.maximum == 100.0


class TestCategoricalCategoricalEnrichment:
    def make_relationship(self) -> CategoricalCategoricalRelationship:
        return CategoricalCategoricalRelationship(
            column_x="embarked",
            column_y="survived",
            n_observations=891,
            n_groups_x=3,
            n_groups_y=2,
            chi2=100.0,
            p_value=1e-10,
            cramers_v=0.5,
            top_associations=[
                ("S", "0", 4.2),
                ("C", "1", -2.6),
                ("Q", "1", 1.1),
            ],
        )

    def test_top_combos_listed_up_to_three(self):
        insights = generate_relationship_insights(
            [], [], [self.make_relationship()]
        )

        message = insights[0].message
        assert "Standout combinations" in message
        assert "embarked=S with survived=0" in message
        assert "embarked=C with survived=1" in message
        # Residual 1.1 < 2.0: no entra en el ranking.
        assert "embarked=Q" not in message

    def test_no_combos_phrase_when_residuals_weak(self):
        relationship = self.make_relationship()
        relationship.top_associations = [("S", "0", 1.2)]

        insights = generate_relationship_insights([], [], [relationship])

        assert "Standout combinations" not in insights[0].message


class TestCrossLinks:
    def make_insight(self, title: str, columns: list[str], score: float) -> Insight:
        return Insight(
            title=title,
            message=f"{title}.",
            severity="info",
            category="relationship",
            columns=columns,
            score=score,
        )

    def test_weaker_card_links_to_stronger(self):
        strong = self.make_insight(
            "Strong association between age and fare", ["age", "fare"], 0.9
        )
        weak = self.make_insight(
            "age differs across embark_town groups", ["age", "embarked"], 0.6
        )

        apply_cross_links([strong, weak])

        assert any(
            caveat.startswith("Related: Strong association between age and fare")
            for caveat in weak.caveats
        )
        assert not any(
            caveat.startswith("Related:") for caveat in strong.caveats
        )

    def test_max_two_links_per_card(self):
        shared_column = "age"
        cards = [
            self.make_insight(f"Finding {index}", [shared_column], score)
            for index, score in enumerate([0.95, 0.9, 0.8, 0.7, 0.6])
        ]

        apply_cross_links(cards)

        for card in cards:
            related = [
                caveat for caveat in card.caveats if caveat.startswith("Related:")
            ]
            assert len(related) <= 2

    def test_quality_cards_never_link(self):
        warning = Insight(
            title="High missing percentage in age",
            message="age has 20% missing.",
            severity="warning",
            category="quality",
            columns=["age"],
            score=0.99,
        )
        relationship = self.make_insight(
            "Strong association between age and fare", ["age", "fare"], 0.8
        )

        apply_cross_links([warning, relationship])

        assert not any(
            caveat.startswith("Related:") for caveat in warning.caveats
        )
        assert not any(
            caveat.startswith("Related:") for caveat in relationship.caveats
        )

    def test_comparative_cards_never_link(self):
        # Las comparativas comparten la métrica con casi todo
        # ("X leads in Y"); linkarlas convierte los caveats en spam.
        ranking = self.make_insight(
            "Southampton accounts for the highest total sibsp",
            ["embarked", "sibsp"],
            0.95,
        )
        ranking.category = "comparative"
        relationship = self.make_insight(
            "sibsp differs across alone groups", ["alone", "sibsp"], 0.7
        )

        apply_cross_links([ranking, relationship])

        assert not any(
            caveat.startswith("Related:") for caveat in ranking.caveats
        )
        assert not any(
            caveat.startswith("Related:") for caveat in relationship.caveats
        )

    def test_low_score_cards_never_link(self):
        high = self.make_insight("Big finding", ["age"], 0.9)
        low = self.make_insight("Tiny finding", ["age"], 0.3)

        apply_cross_links([high, low])

        assert not any(
            caveat.startswith("Related:") for caveat in high.caveats + low.caveats
        )

    def test_deterministic(self):
        def build():
            return [
                self.make_insight(
                    "Strong association between age and fare", ["age", "fare"], 0.9
                ),
                self.make_insight(
                    "age differs across class groups", ["age", "class"], 0.7
                ),
                self.make_insight(
                    "age is right-skewed", ["age"], 0.8
                ),
            ]

        first = build()
        second = build()

        apply_cross_links(first)
        apply_cross_links(second)

        assert [card.caveats for card in first] == [card.caveats for card in second]
