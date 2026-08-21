import pandas as pd
from src.config import DATA_DIR
from src.data_generator import generate_demand_dataset

def main():
    print("Generando dataset oficial de demanda y despachos (Proceso Poisson)...")
    
    df = generate_demand_dataset()
    
    output_path = DATA_DIR / 'demand_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generado exitosamente en: {output_path}")
    print(f"Registros totales en la serie de tiempo: {len(df)}")
    
    dispatched_df = df[df['tipo_salida'] != 'Ninguna']
    print(f"Combis despachadas efectivamente: {len(dispatched_df)}")
    
    print("\n--- Vista Previa de los Primeros 10 Registros (13 Campos Requeridos) ---")
    print(df.head(10).to_string(index=False))
    
    print("\n--- Resumen por Ruta ---")
    summary = df.groupby('ruta').agg(
        llegadas_totales=('llegadas_intervalo', 'sum'),
        total_despachos=('pasajeros_al_salir', lambda x: (x > 0).sum()),
        pasajeros_totales=('pasajeros_al_salir', 'sum'),
        recogidos_intermedias=('alumnos_recogidos_intermedias', 'sum')
    )
    print(summary.to_string())

if __name__ == '__main__':
    main()
