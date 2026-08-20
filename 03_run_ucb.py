import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import DATA_DIR, PLOTS_DIR
from src.ucb_bandit import UCB1Agent, evaluate_policy_dispatch, calculate_reward

def run_ucb_simulation():
    print("🤖 Iniciando simulación del Algoritmo UCB1 (Multi-Armed Bandit)...")
    
    csv_path = DATA_DIR / 'demand_dataset.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró {csv_path}. Ejecuta primero 01_generate_dataset.py")
        
    df = pd.read_csv(csv_path)
    
    agent = UCB1Agent(n_arms=5, c=1.414)
    
    # Listas para almacenar métricas a lo largo del tiempo
    history_rewards = []
    history_actions = []
    history_wait_times_ucb = []
    history_wait_times_trad = []
    history_empty_seats_ucb = []
    history_empty_seats_trad = []
    
    total_dispatches_ucb = 0
    total_dispatches_trad = 0
    
    # Recorrer cada intervalo del dataset (2,352 registros)
    for idx, row in df.iterrows():
        students_waiting = int(row['alumnos_esperando'])
        wait_time_accum = float(row['tiempo_espera_acum'])
        
        # 1. El agente UCB1 selecciona una política (brazo a_0 ... a_4)
        action = agent.select_action()
        history_actions.append(action)
        
        # 2. Evaluar desempeño bajo UCB1
        dispatched_ucb, pax_ucb, empty_ucb = evaluate_policy_dispatch(action, students_waiting, wait_time_accum)
        
        # 3. Evaluar desempeño bajo Política Tradicional (a_0: salir solo si hay 18 pax)
        dispatched_trad, pax_trad, empty_trad = evaluate_policy_dispatch(0, students_waiting, wait_time_accum)
        
        leftover_ucb = max(0, students_waiting - pax_ucb)
        leftover_trad = max(0, students_waiting - pax_trad)
        
        # 4. Calcular recompensa R en [0, 1] para la decisión elegida por UCB1
        reward = calculate_reward(
            wait_time_min=wait_time_accum,
            leftover_students=leftover_ucb,
            empty_seats=empty_ucb
        )
        
        agent.update(action, reward)
        history_rewards.append(reward)
        
        if dispatched_ucb:
            total_dispatches_ucb += 1
            history_wait_times_ucb.append(wait_time_accum)
            history_empty_seats_ucb.append(empty_ucb)
            
        if dispatched_trad:
            total_dispatches_trad += 1
            history_wait_times_trad.append(wait_time_accum)
            history_empty_seats_trad.append(empty_trad)

    # Convertir a numpy arrays para estadísticas
    avg_reward = float(np.mean(history_rewards))
    cum_reward = float(np.sum(history_rewards))
    
    avg_wait_ucb = float(np.mean(history_wait_times_ucb)) if history_wait_times_ucb else 0.0
    avg_wait_trad = float(np.mean(history_wait_times_trad)) if history_wait_times_trad else 0.0
    
    max_wait_ucb = float(np.max(history_wait_times_ucb)) if history_wait_times_ucb else 0.0
    max_wait_trad = float(np.max(history_wait_times_trad)) if history_wait_times_trad else 0.0
    
    avg_empty_ucb = float(np.mean(history_empty_seats_ucb)) if history_empty_seats_ucb else 0.0
    avg_empty_trad = float(np.mean(history_empty_seats_trad)) if history_empty_seats_trad else 0.0

    print("\n--- 🏆 RESULTADOS COMPARATIVOS: POLÍTICA TRADICIONAL VS ALGORITMO UCB1 ---")
    print(f"🔹 Recompensa Promedio Acumulada UCB1: {avg_reward:.4f} (Máximo teórico 1.0)")
    print(f"🔹 Recompensa Total Acumulada UCB1:   {cum_reward:.1f} pts")
    print("\n📊 Métricas clave de rendimiento:")
    print(f"  • Tiempo Promedio de Espera:  Tradicional = {avg_wait_trad:.1f} min  |  UCB1 = {avg_wait_ucb:.1f} min  (Reducción: {((avg_wait_trad - avg_wait_ucb)/avg_wait_trad)*100:.1f}%)")
    print(f"  • Tiempo Máximo de Espera:    Tradicional = {max_wait_trad:.1f} min  |  UCB1 = {max_wait_ucb:.1f} min")
    print(f"  • Despachos Efectivos:        Tradicional = {total_dispatches_trad} viajes |  UCB1 = {total_dispatches_ucb} viajes")
    print(f"  • Asientos Vacíos Promedio:   Tradicional = {avg_empty_trad:.1f} pax  |  UCB1 = {avg_empty_ucb:.1f} pax")
    
    print("\n🎯 Elección de Brazos (Políticas Seleccionadas por UCB1):")
    for a in range(agent.n_arms):
        pct = (agent.counts[a] / agent.total_steps) * 100
        print(f"  • {agent.arm_names[a]}: {agent.counts[a]} elecciones ({pct:.1f}%) | Q_bar = {agent.means[a]:.3f}")

    # Estilo de gráficas profesional
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    palette = ['#4A154B', '#2A9D8F', '#E76F51', '#F4A261', '#E9C46A']

    # 📈 FIGURA 2: Recompensa Acumulada y Selección de Acciones
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    cum_rewards_series = np.cumsum(history_rewards)
    ax1.plot(cum_rewards_series, color='#4A154B', linewidth=2, label='Recompensa Acumulada UCB1')
    ax1.set_ylabel('Recompensa Acumulada', fontsize=12, fontweight='bold')
    ax1.set_title('Evolución del Aprendizaje del Agente UCB1 (21 Días de Simulación)', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', frameon=True)

    ax2.plot(history_actions, color='#E76F51', alpha=0.6, marker='o', linestyle='none', markersize=2)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(['a0 (18pax)', 'a1 (14pax/15m)', 'a2 (12pax/25m)', 'a3 (10pax/35m)', 'a4 (<10pax/45m)'])
    ax2.set_xlabel('Pasos de Simulación (Intervalos de 15 min)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Política Elegida (Brazo)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig2_path = PLOTS_DIR / 'fig2_ucb_recompensa.png'
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"\n✅ Gráfica 2 guardada en: {fig2_path}")

    # 📈 FIGURA 3: Comparativa Cuantitativa Tradicional vs UCB1
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

    metrics_df = pd.DataFrame({
        'Política': ['Tradicional (18 pax)', 'Algoritmo UCB1'],
        'Tiempo Espera Prom (min)': [avg_wait_trad, avg_wait_ucb],
        'Tiempo Espera Máx (min)': [max_wait_trad, max_wait_ucb]
    })

    sns.barplot(data=metrics_df, x='Política', y='Tiempo Espera Prom (min)', ax=ax3, palette=['#E76F51', '#2A9D8F'])
    ax3.set_title('Tiempo Promedio de Espera por Alumno', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Minutos de Espera', fontsize=11)
    for p in ax3.patches:
        ax3.annotate(f"{p.get_height():.1f} min", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    sns.barplot(data=metrics_df, x='Política', y='Tiempo Espera Máx (min)', ax=ax4, palette=['#E76F51', '#2A9D8F'])
    ax4.set_title('Tiempo Máximo de Espera Registrado', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Minutos Máximos', fontsize=11)
    for p in ax4.patches:
        ax4.annotate(f"{p.get_height():.1f} min", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    plt.suptitle('Comparación de Impacto: Política Tradicional vs Algoritmo UCB1', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig3_path = PLOTS_DIR / 'fig3_comparativa.png'
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"✅ Gráfica 3 guardada en: {fig3_path}")

if __name__ == '__main__':
    run_ucb_simulation()
