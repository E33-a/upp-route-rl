"""Modelo OLS para estimar el tiempo necesario para llenar una combi."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


X_COLUMN = "alumnos_esperando"
Y_SOURCE_COLUMN = "tiempo_espera_acum"
Y_MODEL_NAME = "minutos_para_llenar"


@dataclass(frozen=True)
class OLSResult:
    intercept: float
    slope: float
    r_squared: float
    p_value: float
    observations: int


@dataclass(frozen=True)
class OLSDiagnostics:
    regression_ss: float
    residual_ss: float
    total_ss: float
    regression_df: int
    residual_df: int
    regression_ms: float
    residual_ms: float
    f_statistic: float
    anova_p_value: float
    linearity_f_statistic: float
    linearity_p_value: float
    durbin_watson: float
    shapiro_wilk: float
    shapiro_p_value: float
    breusch_pagan_lm: float
    breusch_pagan_p_value: float


def load_and_filter_data(csv_path: Path) -> pd.DataFrame:
    """Lee el CSV y conserva despachos llenos o con alumnos en espera."""
    df = pd.read_csv(csv_path)
    required = {X_COLUMN, Y_SOURCE_COLUMN, "tipo_salida"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Faltan columnas requeridas en el dataset: " + ", ".join(sorted(missing))
        )

    numeric_columns = [X_COLUMN, Y_SOURCE_COLUMN]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    # El dataset registra los dos estados operativos solicitados: salida con
    # combi llena o salida parcial después de permanecer en espera máxima.
    relevant = df["tipo_salida"].isin(["Llena", "Parcial"])
    filtered = df.loc[relevant].dropna(subset=numeric_columns).copy()
    filtered[Y_MODEL_NAME] = filtered[Y_SOURCE_COLUMN]

    if len(filtered) < 3 or filtered[X_COLUMN].nunique() < 2:
        raise ValueError("No hay suficientes datos variables para ajustar el modelo OLS.")
    return filtered


def fit_ols(df: pd.DataFrame) -> OLSResult:
    """Ajusta Y = a + bX y calcula R cuadrada y p-valor bilateral de b."""
    x = df[X_COLUMN].to_numpy(dtype=float)
    y = df[Y_MODEL_NAME].to_numpy(dtype=float)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    sxx = float(np.dot(x_centered, x_centered))

    slope = float(np.dot(x_centered, y_centered) / sxx)
    intercept = float(y.mean() - slope * x.mean())
    predicted = intercept + slope * x
    residuals = y - predicted
    ss_res = float(np.dot(residuals, residuals))
    ss_total = float(np.dot(y_centered, y_centered))
    r_squared = 1.0 - (ss_res / ss_total)

    degrees_freedom = len(x) - 2
    residual_variance = ss_res / degrees_freedom
    slope_standard_error = np.sqrt(residual_variance / sxx)
    t_statistic = slope / slope_standard_error
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), degrees_freedom))

    return OLSResult(
        intercept=intercept,
        slope=slope,
        r_squared=r_squared,
        p_value=p_value,
        observations=len(df),
    )


def calculate_diagnostics(df: pd.DataFrame, result: OLSResult) -> OLSDiagnostics:
    """Calcula ANOVA y pruebas de los supuestos del modelo lineal simple."""
    x = df[X_COLUMN].to_numpy(dtype=float)
    y = df[Y_MODEL_NAME].to_numpy(dtype=float)
    fitted = result.intercept + result.slope * x
    residuals = y - fitted
    total_ss = float(np.sum((y - y.mean()) ** 2))
    residual_ss = float(np.sum(residuals**2))
    regression_ss = total_ss - residual_ss
    regression_df = 1
    residual_df = len(y) - 2
    regression_ms = regression_ss / regression_df
    residual_ms = residual_ss / residual_df
    f_statistic = regression_ms / residual_ms
    anova_p_value = float(stats.f.sf(f_statistic, regression_df, residual_df))

    # Linealidad: prueba de término cuadrático adicional H0: beta_2 = 0.
    quadratic_design = np.column_stack((np.ones(len(x)), x, x**2))
    quadratic_coefficients, *_ = np.linalg.lstsq(quadratic_design, y, rcond=None)
    quadratic_residuals = y - quadratic_design @ quadratic_coefficients
    quadratic_ss = float(np.sum(quadratic_residuals**2))
    linearity_f = ((residual_ss - quadratic_ss) / 1) / (quadratic_ss / (len(y) - 3))
    linearity_p = float(stats.f.sf(linearity_f, 1, len(y) - 3))

    residual_difference = np.diff(residuals)
    durbin_watson = float(np.sum(residual_difference**2) / residual_ss)
    shapiro_wilk, shapiro_p = stats.shapiro(residuals)

    # Breusch-Pagan: n*R2 de la regresión auxiliar de e_i^2 sobre [1, X].
    squared_residuals = residuals**2
    auxiliary_design = np.column_stack((np.ones(len(x)), x))
    auxiliary_coefficients, *_ = np.linalg.lstsq(
        auxiliary_design, squared_residuals, rcond=None
    )
    auxiliary_fitted = auxiliary_design @ auxiliary_coefficients
    auxiliary_total_ss = float(np.sum((squared_residuals - squared_residuals.mean()) ** 2))
    auxiliary_residual_ss = float(np.sum((squared_residuals - auxiliary_fitted) ** 2))
    auxiliary_r_squared = 1.0 - auxiliary_residual_ss / auxiliary_total_ss
    breusch_pagan_lm = len(x) * auxiliary_r_squared
    breusch_pagan_p = float(stats.chi2.sf(breusch_pagan_lm, 1))

    return OLSDiagnostics(
        regression_ss=regression_ss,
        residual_ss=residual_ss,
        total_ss=total_ss,
        regression_df=regression_df,
        residual_df=residual_df,
        regression_ms=regression_ms,
        residual_ms=residual_ms,
        f_statistic=f_statistic,
        anova_p_value=anova_p_value,
        linearity_f_statistic=float(linearity_f),
        linearity_p_value=linearity_p,
        durbin_watson=durbin_watson,
        shapiro_wilk=float(shapiro_wilk),
        shapiro_p_value=float(shapiro_p),
        breusch_pagan_lm=float(breusch_pagan_lm),
        breusch_pagan_p_value=breusch_pagan_p,
    )


def save_regression_plot(df: pd.DataFrame, result: OLSResult, output_path: Path) -> None:
    """Guarda la dispersión observada y la recta OLS ajustada."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = df[X_COLUMN].to_numpy(dtype=float)
    y = df[Y_MODEL_NAME].to_numpy(dtype=float)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = result.intercept + result.slope * x_line

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x, y, alpha=0.95, s=42, color="#95B84F", edgecolors="none")
    ax.plot(x_line, y_line, color="#95B84F", linewidth=2.2, linestyle=":")
    ax.set_title(
        "Tiempo en llenar la combi vs alumnos esperando",
        fontsize=15,
        fontweight="bold",
        color="#555555",
    )
    ax.set_xlabel("Alumnos al esperar la combi (X)", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "Minutos para llenar / tiempo de espera\nacumulado (Y)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(-20, 120)
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(-20, 121, 20))
    ax.text(
        0.98,
        0.98,
        f"y = {result.slope:.4f}x + {result.intercept:.4f}\n"
        f"R² = {result.r_squared:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
    )
    ax.grid(color="#D0D0D0", linewidth=1.0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_homoscedasticity_plot(
    df: pd.DataFrame, result: OLSResult, diagnostics: OLSDiagnostics, output_path: Path
) -> None:
    """Grafica los valores predichos contra los residuos del modelo OLS."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = df[X_COLUMN].to_numpy(dtype=float)
    y = df[Y_MODEL_NAME].to_numpy(dtype=float)
    fitted = result.intercept + result.slope * x
    residuals = y - fitted

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(fitted, residuals, alpha=0.95, s=48, color="#95B84F", edgecolors="none")
    ax.set_title("Homocedasticidad", fontsize=18, color="#555555", pad=18)
    ax.set_xlim(-15, 20)
    ax.set_ylim(-20, 100)
    ax.set_xticks(np.arange(-15, 21, 5))
    ax.set_yticks(np.arange(-20, 101, 20))
    ax.grid(color="#D0D0D0", linewidth=1.1)
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_color("#AFAFAF")
    ax.spines["bottom"].set_color("#AFAFAF")
    ax.spines["top"].set_color("#D0D0D0")
    ax.spines["right"].set_color("#D0D0D0")
    ax.tick_params(colors="#555555", labelsize=11)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
