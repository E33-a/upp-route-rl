import pandas as pd
from src.config import DATA_DIR
from src.data_generator import generate_demand_dataset

def main():
    print("🚀 Generando dataset estocástico basado en Eventos de Despacho (Proceso Poisson)...")
    
    df = generate_demand_dataset()
    
    output_path = DATA_DIR / 'demand_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset generado exitosamente en: {output_path}")
    print(f"📊 Total de viajes de combi registrados: {len(df)}")
    print("\n--- Vista Previa del Dataset (Primeros 10 viajes) ---")
    print(df.head(10).to_string(index=False))
    
    print("\n--- Resumen por Ruta ---")
    summary = df.groupby('ruta').agg(
        total_viajes=('id_viaje', 'count'),
        total_pasajeros=('pasajeros', 'sum'),
        espera_max_promedio=('tiempo_espera_max_min', 'mean'),
        espera_prom_promedio=('tiempo_espera_prom_min', 'mean'),
        duracion_viaje_promedio=('tiempo_recorrido_min', 'mean')
    ).round(1)
    print(summary.to_string())

if __name__ == '__main__':
    main()
