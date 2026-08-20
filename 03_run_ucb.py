import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import DATA_DIR, PLOTS_DIR
from src.ucb_bandit import UCB1Agent, evaluate_policy_dispatch, calculate_reward

def run_ucb_simulation():
    """
    Executes the 21-day UCB1 Multi-Armed Bandit simulation, compares policy performance
    against the traditional baseline (18 pax full dispatch), and exports analytical plots.
    """
    print("🤖 Starting UCB1 Multi-Armed Bandit Simulation...")
    
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

    print("\n--- 🏆 COMPARATIVE RESULTS: TRADITIONAL POLICY VS UCB1 AGENT ---")
    print(f"🔹 Average Cumulative Reward UCB1: {avg_reward:.4f} (Theoretical max 1.0)")
    print(f"🔹 Total Cumulative Reward UCB1:   {cum_reward:.1f} pts")
    print("\n📊 Key Performance Metrics:")
    print(f"  • Average Waiting Time:  Traditional = {avg_wait_trad:.1f} min  |  UCB1 = {avg_wait_ucb:.1f} min")
    print(f"  • Maximum Waiting Time:  Traditional = {max_wait_trad:.1f} min  |  UCB1 = {max_wait_ucb:.1f} min")
    print(f"  • Total Dispatches:      Traditional = {total_dispatches_trad} trips |  UCB1 = {total_dispatches_ucb} trips")
    print(f"  • Average Empty Seats:   Traditional = {avg_empty_trad:.1f} pax  |  UCB1 = {avg_empty_ucb:.1f} pax")
    
    print("\n🎯 Arm Selection Frequency (Policies Chosen by UCB1):")
    for a in range(agent.n_arms):
        pct = (agent.counts[a] / agent.total_steps) * 100
        print(f"  • {agent.arm_names[a]}: {agent.counts[a]} choices ({pct:.1f}%) | Q_bar = {agent.means[a]:.3f}")

    # Plot styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 📈 FIGURE 2: Cumulative Reward and Action Selection History
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    cum_rewards_series = np.cumsum(history_rewards)
    ax1.plot(cum_rewards_series, color='#4A154B', linewidth=2, label='UCB1 Cumulative Reward')
    ax1.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax1.set_title('UCB1 Agent Learning Curve (21 Operational Days)', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', frameon=True)

    ax2.plot(history_actions, color='#E76F51', alpha=0.6, marker='o', linestyle='none', markersize=2)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(['a0 (18pax)', 'a1 (14pax/15m)', 'a2 (12pax/25m)', 'a3 (10pax/35m)', 'a4 (<10pax/45m)'])
    ax2.set_xlabel('Simulation Steps (15-min Intervals)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Selected Policy Arm', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig2_path = PLOTS_DIR / 'fig2_ucb_recompensa.png'
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"\n✅ Figure 2 saved at: {fig2_path}")

    # 📈 FIGURE 3: Quantitative Comparison (Traditional vs UCB1)
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

    metrics_df = pd.DataFrame({
        'Policy': ['Traditional (18 pax)', 'UCB1 Algorithm'],
        'Average Wait Time (min)': [avg_wait_trad, avg_wait_ucb],
        'Maximum Wait Time (min)': [max_wait_trad, max_wait_ucb]
    })

    sns.barplot(data=metrics_df, x='Policy', y='Average Wait Time (min)', hue='Policy', ax=ax3, palette=['#E76F51', '#2A9D8F'], legend=False)
    ax3.set_title('Average Passenger Waiting Time', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Wait Time (Minutes)', fontsize=11)
    for p in ax3.patches:
        ax3.annotate(f"{p.get_height():.1f} min", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    sns.barplot(data=metrics_df, x='Policy', y='Maximum Wait Time (min)', hue='Policy', ax=ax4, palette=['#E76F51', '#2A9D8F'], legend=False)
    ax4.set_title('Maximum Passenger Waiting Time', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Max Wait Time (Minutes)', fontsize=11)
    for p in ax4.patches:
        ax4.annotate(f"{p.get_height():.1f} min", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    plt.suptitle('Performance Benchmark: Traditional Baseline vs UCB1 Agent', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig3_path = PLOTS_DIR / 'fig3_comparativa.png'
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"✅ Figure 3 saved at: {fig3_path}")

if __name__ == '__main__':
    run_ucb_simulation()
