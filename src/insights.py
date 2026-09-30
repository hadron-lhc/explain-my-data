from dataclasses import dataclass

from src.quality import ColumnQuality, QualityReport

from src.relationships import (
    CategoricalNumericRelationship,
    NumericRelationship,
)


@dataclass
class Insight:
    title: str
    message: str
    severity: str


def _missing_values_insight(
    column: ColumnQuality,
) -> Insight | None:
    if column.missing_count == 0:
        return None

    return Insight(
        title=f"Missing values in {column.column_name}",
        message=(
            f"{column.column_name} contains {column.missing_count} "
            f"missing values ({column.missing_percentage:.1f}% of the column)."
        ),
        severity="warning",
    )


def _type_mismatch_insight(
    column: ColumnQuality,
) -> Insight | None:
    if not column.is_type_mismatch:
        return None

    return Insight(
        title=f"Possible type mismatch in {column.column_name}",
        message=(
            f"{column.column_name} is currently stored as text, but "
            f"{column.conversion_rate * 100:.1f}% of its values can be "
            f"interpreted as numeric values."
        ),
        severity="warning",
    )


def _outlier_insight(
    column: ColumnQuality,
) -> Insight | None:
    if column.n_outliers == 0:
        return None

    return Insight(
        title=f"Potential unusual values in {column.column_name}",
        message=(
            f"{column.column_name} contains {column.n_outliers} values "
            f"outside the overall IQR range "
            f"({column.outlier_percentage:.1f}% of the column). "
            f"These values may require review."
        ),
        severity="warning",
    )


def _duplicates_insight(
    report: QualityReport,
) -> Insight | None:
    if report.n_duplicates == 0:
        return None

    return Insight(
        title="Duplicate rows detected",
        message=(
            f"The dataset contains {report.n_duplicates} duplicate rows "
            f"({report.duplicate_percentage:.1f}% of all rows)."
        ),
        severity="warning",
    )


def _numeric_relationship_insight(
    relationship: NumericRelationship,
) -> Insight | None:
    strength = max(
        abs(relationship.pearson),
        abs(relationship.spearman),
    )

    if strength < 0.5:
        return None

    if relationship.pearson >= 0.5:
        direction = "positive"
    elif relationship.pearson <= -0.5:
        direction = "negative"
    else:
        direction = "monotonic"

    return Insight(
        title=(
            f"Strong association between "
            f"{relationship.column_x} and {relationship.column_y}"
        ),
        message=(
            f"{relationship.column_x} and {relationship.column_y} "
            f"show a {direction} association "
            f"(Pearson: {relationship.pearson:.2f}, "
            f"Spearman: {relationship.spearman:.2f})."
        ),
        severity="info",
    )


def _categorical_numeric_relationship_insight(
    relationship: CategoricalNumericRelationship,
) -> Insight | None:
    if relationship.group_separation < 0.5:
        return None

    return Insight(
        title=(
            f"{relationship.column_numeric} differs across "
            f"{relationship.column_categorical} groups"
        ),
        message=(
            f"Average {relationship.column_numeric} varies substantially "
            f"between {relationship.column_categorical} groups "
            f"(group separation: {relationship.group_separation:.2f})."
        ),
        severity="info",
    )


def generate_relationship_insights(
    numeric_relationships: list[NumericRelationship],
    categorical_numeric_relationships: list[CategoricalNumericRelationship],
) -> list[Insight]:
    insights: list[Insight] = []

    for relationship in numeric_relationships:
        insight = _numeric_relationship_insight(relationship)
        if insight is not None:
            insights.append(insight)

    for relationship in categorical_numeric_relationships:
        insight = _categorical_numeric_relationship_insight(relationship)
        if insight is not None:
            insights.append(insight)

    return insights


def generate_quality_insights(
    report: QualityReport,
) -> list[Insight]:
    insights: list[Insight] = []

    for column in report.columns:
        missing_insight = _missing_values_insight(column)
        if missing_insight is not None:
            insights.append(missing_insight)

        type_mismatch_insight = _type_mismatch_insight(column)
        if type_mismatch_insight is not None:
            insights.append(type_mismatch_insight)

        outlier_insight = _outlier_insight(column)
        if outlier_insight is not None:
            insights.append(outlier_insight)

    duplicate_insight = _duplicates_insight(report)
    if duplicate_insight is not None:
        insights.append(duplicate_insight)

    return insights
