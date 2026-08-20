"""Punto de entrada para entrenar y visualizar la regresión lineal OLS."""

from src.config import DATA_DIR, PLOTS_DIR
from src.regression_model import (
    calculate_diagnostics,
    fit_ols,
    load_and_filter_data,
    save_homoscedasticity_plot,
    save_regression_plot,
)


def main() -> None:
    dataset_path = DATA_DIR / "demand_dataset.csv"
    plot_path = PLOTS_DIR / "fig1_regresion.png"
    homoscedasticity_plot_path = PLOTS_DIR / "fig2_homocedasticidad.png"

    data = load_and_filter_data(dataset_path)
    result = fit_ols(data)
    diagnostics = calculate_diagnostics(data, result)
    save_regression_plot(data, result, plot_path)
    save_homoscedasticity_plot(
        data, result, diagnostics, homoscedasticity_plot_path
    )

    sign = "+" if result.slope >= 0 else "-"
    print(f"Registros usados: {result.observations}")
    print(
        f"Ecuación OLS: Y = {result.intercept:.6f} {sign} "
        f"{abs(result.slope):.6f} X"
    )
    print(f"R²: {result.r_squared:.6f}")
    print(f"p-valor de la pendiente: {result.p_value:.6e}")
    print("\nPruebas de supuestos (alpha = 0.05)")
    print(
        "Linealidad (término X²): "
        f"F={diagnostics.linearity_f_statistic:.6f}, "
        f"p={diagnostics.linearity_p_value:.6e}"
    )
    print(f"Independencia (Durbin-Watson): {diagnostics.durbin_watson:.6f}")
    print("Normalidad: revisar el supuesto mediante la prueba de Shapiro-Wilk.")
    print(
        "Homocedasticidad (Breusch-Pagan): "
        f"LM={diagnostics.breusch_pagan_lm:.6f}, "
        f"p={diagnostics.breusch_pagan_p_value:.6e}"
    )
    print(f"Gráfica guardada en: {plot_path}")
    print(f"Gráfica de homocedasticidad guardada en: {homoscedasticity_plot_path}")


if __name__ == "__main__":
    main()
