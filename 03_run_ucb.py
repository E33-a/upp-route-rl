import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import DATA_DIR, PLOTS_DIR
from src.ucb_bandit import UCB1Agent, evaluate_policy_dispatch, calculate_reward

def run_ucb_simulation():
    """
    Executes the 21-day UCB1 Multi-Armed Bandit simulation, compares policy performance
    against the traditional baseline (18 pax full dispatch), and exports analytical plots.
    """
    print("Starting UCB1 Multi-Armed Bandit Simulation...")
    
    csv_path = DATA_DIR / 'demand_dataset.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please execute 01_generate_dataset.py first.")
        
    df = pd.read_csv(csv_path)
    
    agent = UCB1Agent(n_arms=5, c=1.414)
    
    # Tracking metrics across simulation intervals
    history_rewards = []
    history_actions = []
    history_wait_times_ucb = []
    history_wait_times_trad = []
    history_empty_seats_ucb = []
    history_empty_seats_trad = []
    
    total_dispatches_ucb = 0
    total_dispatches_trad = 0
    
    # Iterate through each 15-minute simulation interval (2,352 records)
    for idx, row in df.iterrows():
        students_waiting = int(row['pasajeros_al_salir'])
        wait_time_accum = float(row['tiempo_espera_acum'])
        
        # 1. Agent selects policy arm (a0 ... a4)
        action = agent.select_action()
        history_actions.append(action)
        
        # 2. Evaluate performance under UCB1 policy
        dispatched_ucb, pax_ucb, empty_ucb = evaluate_policy_dispatch(action, students_waiting, wait_time_accum)
        
        # 3. Evaluate baseline traditional policy (a0: 18 pax full dispatch)
        dispatched_trad, pax_trad, empty_trad = evaluate_policy_dispatch(0, students_waiting, wait_time_accum)
        
        leftover_ucb = max(0, students_waiting - pax_ucb)
        leftover_trad = max(0, students_waiting - pax_trad)
        
        # 4. Compute normalized reward R in [0, 1] for UCB1 action
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

    # Compute summary statistics
    avg_reward = float(np.mean(history_rewards))
    cum_reward = float(np.sum(history_rewards))
    
    avg_wait_ucb = float(np.mean(history_wait_times_ucb)) if history_wait_times_ucb else 0.0
    avg_wait_trad = float(np.mean(history_wait_times_trad)) if history_wait_times_trad else 0.0
    
    max_wait_ucb = float(np.max(history_wait_times_ucb)) if history_wait_times_ucb else 0.0
    max_wait_trad = float(np.max(history_wait_times_trad)) if history_wait_times_trad else 0.0
    
    avg_empty_ucb = float(np.mean(history_empty_seats_ucb)) if history_empty_seats_ucb else 0.0
    avg_empty_trad = float(np.mean(history_empty_seats_trad)) if history_empty_seats_trad else 0.0

    print("\n--- COMPARATIVE RESULTS: TRADITIONAL POLICY VS UCB1 AGENT ---")
    print(f"Average Reward UCB1:            {avg_reward:.4f} (Theoretical max 1.0)")
    print(f"Total Cumulative Reward UCB1:   {cum_reward:.1f} pts")
    print("\nKey Performance Metrics:")
    print(f"  * Average Waiting Time:  Traditional = {avg_wait_trad:.1f} min  |  UCB1 = {avg_wait_ucb:.1f} min")
    print(f"  * Maximum Waiting Time:  Traditional = {max_wait_trad:.1f} min  |  UCB1 = {max_wait_ucb:.1f} min")
    print(f"  * Total Dispatches:      Traditional = {total_dispatches_trad} trips |  UCB1 = {total_dispatches_ucb} trips")
    print(f"  * Average Empty Seats:   Traditional = {avg_empty_trad:.1f} pax  |  UCB1 = {avg_empty_ucb:.1f} pax")
    
    print("\nArm Selection Frequency (Policies Chosen by UCB1):")
    for a in range(agent.n_arms):
        pct = (agent.counts[a] / agent.total_steps) * 100
        print(f"  * {agent.arm_names[a]}: {agent.counts[a]} choices ({pct:.1f}%) | Q_bar = {agent.means[a]:.3f}")

    # FIGURE 2: Cumulative Reward and Action Selection History (Dark Theme)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    fig.patch.set_facecolor("#1C2541")
    ax1.set_facecolor("#0B132B")
    ax2.set_facecolor("#0B132B")

    cum_rewards_series = np.cumsum(history_rewards)
    ax1.plot(cum_rewards_series, color='#38BDF8', linewidth=2.5, label='UCB1 Recompensa Acumulada')
    ax1.set_ylabel('Recompensa Acumulada', fontsize=11, fontweight='bold', color='#38BDF8')
    ax1.set_title('Evolución de Aprendizaje del Agente UCB1 (21 Días Operativos)', fontsize=13, fontweight='bold', color='#FFFFFF', pad=12)
    ax1.tick_params(colors="#FFFFFF", labelsize=10)
    for spine in ax1.spines.values():
        spine.set_color("#3A506B")
    ax1.grid(color="#3A506B", linestyle=":", linewidth=0.8, alpha=0.7)
    ax1.legend(facecolor="#1C2541", edgecolor="#38BDF8", labelcolor="#FFFFFF", loc="upper left", fontsize=10)

    ax2.plot(history_actions, color='#4ADE80', alpha=0.75, marker='o', linestyle='none', markersize=3.0)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(['a0 (18pax)', 'a1 (14p/15m)', 'a2 (12p/20m)', 'a3 (10p/25m)', 'a4 (8p/30m)'])
    ax2.set_xlabel('Eventos de Despacho Evaluados', fontsize=11, fontweight='bold', color='#4ADE80')
    ax2.set_ylabel('Política Elegida', fontsize=11, fontweight='bold', color='#4ADE80')
    ax2.tick_params(colors="#FFFFFF", labelsize=10)
    for spine in ax2.spines.values():
        spine.set_color("#3A506B")
    ax2.grid(color="#3A506B", linestyle=":", linewidth=0.8, alpha=0.7)

    plt.tight_layout()
    fig2_path = PLOTS_DIR / 'fig2_ucb_recompensa.png'
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Figure 2 saved at: {fig2_path}")

    # FIGURE 3: Quantitative Comparison (Traditional vs UCB1 - Dark Theme)
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.patch.set_facecolor("#1C2541")
    ax3.set_facecolor("#0B132B")
    ax4.set_facecolor("#0B132B")

    metrics_df = pd.DataFrame({
        'Policy': ['Tradicional (a0)', 'Agente UCB1'],
        'Average Dispatch Wait (min)': [avg_wait_trad, avg_wait_ucb],
        'Maximum Dispatch Wait (min)': [max_wait_trad, max_wait_ucb]
    })

    ax3.bar(metrics_df['Policy'], metrics_df['Average Dispatch Wait (min)'], color=['#38BDF8', '#4ADE80'], width=0.55)
    ax3.set_title('Tiempo de Espera Promedio en Despacho', fontsize=12, fontweight='bold', color='#FFFFFF', pad=10)
    ax3.set_ylabel('Espera Promedio (Minutos)', fontsize=11, color='#FFFFFF')
    ax3.tick_params(colors="#FFFFFF", labelsize=10)
    for spine in ax3.spines.values():
        spine.set_color("#3A506B")
    ax3.grid(color="#3A506B", linestyle=":", linewidth=0.8, alpha=0.7, axis='y')
    for p in ax3.patches:
        ax3.annotate(f"{p.get_height():.1f} min", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', color='#0B132B', fontweight='bold', fontsize=11)

    ax4.bar(metrics_df['Policy'], metrics_df['Maximum Dispatch Wait (min)'], color=['#38BDF8', '#4ADE80'], width=0.55)
    ax4.set_title('Tiempo de Espera Máximo en Despacho', fontsize=12, fontweight='bold', color='#FFFFFF', pad=10)
    ax4.set_ylabel('Espera Máxima (Minutos)', fontsize=11, color='#FFFFFF')
    ax4.tick_params(colors="#FFFFFF", labelsize=10)
    for spine in ax4.spines.values():
        spine.set_color("#3A506B")
    ax4.grid(color="#3A506B", linestyle=":", linewidth=0.8, alpha=0.7, axis='y')
    for p in ax4.patches:
        ax4.annotate(f"{p.get_height():.1f} min", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', color='#0B132B', fontweight='bold', fontsize=11)

    fig.suptitle('Benchmark de Desempeño: Política Tradicional vs Agente UCB1', fontsize=13, fontweight='bold', color='#FDE047', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig3_path = PLOTS_DIR / 'fig3_comparativa.png'
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight', pad_inches=0.12, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Figure 3 saved at: {fig3_path}")

    return {
        'avg_reward': avg_reward,
        'cum_reward': cum_reward,
        'avg_wait_ucb': avg_wait_ucb,
        'avg_wait_trad': avg_wait_trad,
        'max_wait_ucb': max_wait_ucb,
        'max_wait_trad': max_wait_trad,
        'total_dispatches_ucb': total_dispatches_ucb,
        'total_dispatches_trad': total_dispatches_trad,
        'avg_empty_ucb': avg_empty_ucb,
        'avg_empty_trad': avg_empty_trad,
        'counts': agent.counts,
        'means': agent.means,
        'total_steps': agent.total_steps
    }

if __name__ == '__main__':
    run_ucb_simulation()
