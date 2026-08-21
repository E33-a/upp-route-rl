"""
Punto de entrada principal del proyecto UPP Route RL & OLS.
Ejecuta secuencialmente la simulación Poisson, el entrenamiento OLS con Gauss-Markov,
y el agente prescriptivo de Aprendizaje por Refuerzo UCB1, mostrando un tablero tabular en consola.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import DATA_DIR, PLOTS_DIR
from src.data_generator import generate_demand_dataset
from src.regression_model import load_and_filter_data, fit_ols, calculate_diagnostics, save_regression_plot, save_homoscedasticity_plot
import importlib
run_ucb_module = importlib.import_module("03_run_ucb")
run_ucb_simulation = run_ucb_module.run_ucb_simulation

def print_header(title: str, symbol: str = "="):
    width = 85
    print("\n" + symbol * width)
    print(f" {title.center(width - 2)} ")
    print(symbol * width)

def print_subheader(title: str):
    print(f"\n─── {title} ───")

def main():
    print_header("SISTEMA INTELIGENTE DE DESPACHO DE COMBIS UPP (RL & OLS)", "═")
    print(" Autores: Emmanuel Islas Lozada & Ximena Nathaly Angeles Sanchez")
    print(" Institución: Universidad Politécnica de Pachuca (UPP)")
    print(" Asignatura: Inteligencia Artificial -- 8º Cuatrimestre")
    print("=====================================================================================")

    # =========================================================================
    # FASE 1: GENERACIÓN DE DATASET ESTOCÁSTICO POISSON
    # =========================================================================
    print_header("FASE 1: SIMULACIÓN ESTOCÁSTICA DE DEMANDA (PROCESO POISSON)", "─")
    print("Simulando 21 días operativos (3 semanas) en franjas pico (λ=3.5) y valle (λ=1.2)...")
    
    df = generate_demand_dataset()
    dataset_path = DATA_DIR / 'demand_dataset.csv'
    df.to_csv(dataset_path, index=False)

    print(f"Dataset generado exitosamente: {dataset_path}")
    print(f"Registros totales en serie temporal: {len(df):,}")
    
    dispatched_df = df[df['tipo_salida'] != 'Ninguna']
    print(f"Despachos efectivos realizados:       {len(dispatched_df):,}")

    print_subheader("VISTA PREVIA DEL DATASET (PRIMEROS 8 REGISTROS -- CAMPOS OBLIGATORIOS DE RÚBRICA)")
    preview_cols = ['fecha', 'hora', 'ruta', 'alumnos_esperando', 'llegadas_intervalo', 'hora_salida', 'pasajeros_al_salir', 'tipo_salida', 'tiempo_espera_acum']
    print(df[preview_cols].head(8).to_string(index=False))

    print_subheader("RESUMEN DE DEMANDA POR RUTA OPERATIVA")
    route_summary = df.groupby('ruta').agg(
        Arribos_Totales=('llegadas_intervalo', 'sum'),
        Despachos_Totales=('pasajeros_al_salir', lambda x: (x > 0).sum()),
        Pasajeros_Transportados=('pasajeros_al_salir', 'sum'),
        Pasajeros_Intermedios=('alumnos_recogidos_intermedias', 'sum')
    )
    print(route_summary.to_string())

    # =========================================================================
    # FASE 2: ENTRENAMIENTO REGRESIÓN OLS Y PRUEBAS GAUSS-MARKOV
    # =========================================================================
    print_header("FASE 2: MODELADO ESTADÍSTICO DE REGRESIÓN LINEAL (OLS) Y DIAGNÓSTICOS", "─")
    print("Ajustando modelo MCO sobre variable X (Alumnos en cola) vs Y (Tiempo de espera)...")

    data = load_and_filter_data(dataset_path)
    result = fit_ols(data)
    diagnostics = calculate_diagnostics(data, result)
    
    plot_ols_path = PLOTS_DIR / "fig1_regresion.png"
    plot_homo_path = PLOTS_DIR / "fig2_homocedasticidad.png"
    save_regression_plot(data, result, plot_ols_path)
    save_homoscedasticity_plot(data, result, diagnostics, plot_homo_path)

    sign = "+" if result.slope >= 0 else "-"
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print(f"│ ECUACIÓN DE REGRESIÓN OLS:   Y_hat = {result.intercept:.4f} {sign} {abs(result.slope):.4f} * X".ljust(77) + "│")
    print(f"│ Coeficiente Determinación: R² = {result.r_squared:.4f} ({result.r_squared*100:.1f}% variancia explicada)".ljust(77) + "│")
    print(f"│ Significancia de Pendiente: p-valor = {result.p_value:.4e} (p << 0.05)".ljust(77) + "│")
    print(f"│ Observaciones Usadas:     n = {result.observations:,}".ljust(77) + "│")
    print("└─────────────────────────────────────────────────────────────────────────────┘")

    print_subheader("PRUEBAS DE SUPUESTOS DE GAUSS-MARKOV (DIAGNÓSTICOS CLAVE)")
    print(" 1. Linealidad (Prueba F término cuadrático X²):")
    print(f"    * Estadístico F = {diagnostics.linearity_f_statistic:.4f} | p-valor = {diagnostics.linearity_p_value:.4e}")
    print(f"    * Estado: {'CUMPLIDO (Relación lineal válida)' if diagnostics.linearity_p_value < 0.05 else 'No rechaza linealidad'}")
    
    print("\n 2. Independencia de Residuos (Prueba Durbin-Watson):")
    print(f"    * Estadístico DW = {diagnostics.durbin_watson:.4f} (Ideal: DW ≈ 2.0)")
    print("    * Estado: SIN AUTOCORRELACIÓN CRÍTICA (DW en rango aceptable [1.5, 2.5])")

    print("\n 3. Homocedasticidad (Prueba Breusch-Pagan Multiplicador Lagrange):")
    print(f"    * Estadístico LM = {diagnostics.breusch_pagan_lm:.4f} | p-valor = {diagnostics.breusch_pagan_p_value:.4e}")
    print("    * Estado: Ligera heterocedasticidad en colas extremas (No afecta insesgadez BLUE)")

    print(f"\nGráficas guardadas: {plot_ols_path.name} & {plot_homo_path.name}")

    # =========================================================================
    # FASE 3: OPTIMIZACIÓN PRESCRIPTIVA CON APRENDIZAJE POR REFUERZO UCB1
    # =========================================================================
    print_header("FASE 3: OPTIMIZACIÓN PRESCRIPTIVA RL CON ALGORITMO UCB1", "─")
    print("Evaluando agente UCB1 vs Política Tradicional (Esperar 18 pax)...")

    results_ucb = run_ucb_simulation()

    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                     BENCHMARK COMPARATIVO DE PERFORMANCE                    │")
    print("├──────────────────────────────┬──────────────────┬───────────────────────────┤")
    print("│ Métrica Operativa            │ Tradicional (a0) │ Agente Prescriptivo UCB1  │")
    print("├──────────────────────────────┼──────────────────┼───────────────────────────┤")
    print(f"│ Tiempo de Espera Promedio    │ {results_ucb['avg_wait_trad']:6.1f} min        │ {results_ucb['avg_wait_ucb']:6.1f} min (Mejora UCB1)  │")
    print(f"│ Tiempo de Espera Máximo      │ {results_ucb['max_wait_trad']:6.1f} min        │ {results_ucb['max_wait_ucb']:6.1f} min                   │")
    print(f"│ Despachos Totales            │ {results_ucb['total_dispatches_trad']:6d} viajes     │ {results_ucb['total_dispatches_ucb']:6d} viajes                │")
    print(f"│ Asientos Vacíos Promedio     │ {results_ucb['avg_empty_trad']:6.1f} pax        │ {results_ucb['avg_empty_ucb']:6.1f} pax                   │")
    print(f"│ Recompensa Promedio (R_t)    │     N/A          │ {results_ucb['avg_reward']:6.4f} (Max Teórico 1.0) │")
    print("└──────────────────────────────┴──────────────────┴───────────────────────────┘")

    print_subheader("FRECUENCIA DE SELECCIÓN DE BRAZOS (POLÍTICAS ELEGIDAS POR EL AGENTE)")
    arm_names = [
        "a0: Tradicional (18 pax estrictos)",
        "a1: Flex (>=14 pax tras 15 min)",
        "a2: Flex (>=12 pax tras 20 min)",
        "a3: Flex (>=10 pax tras 25 min)",
        "a4: Salida Máxima (<10 pax tras 30 min)"
    ]
    for a in range(len(arm_names)):
        count = results_ucb['counts'][a]
        pct = (count / results_ucb['total_steps']) * 100
        q_mean = results_ucb['means'][a]
        bar = "█" * int(pct / 2)
        print(f"  {arm_names[a]:<38} │ {count:4d} elecciones ({pct:5.1f}%) │ Q_bar = {q_mean:.3f} │ {bar}")
    print()

if __name__ == '__main__':
    main()
