import geobr
import os

# Criar diretório para os dados
os.makedirs("./geodata", exist_ok=True)

# 1. Baixar e salvar o GeoJSON dos estados
print("Baixando dados dos estados...")
gdf_states = geobr.read_state()
gdf_states.to_file("./geodata/brazil_states.json", driver="GeoJSON")
print("✓ Salvo em ./geodata/brazil_states.json")

# 2. Baixar e salvar o GeoJSON de todos os municípios
print("\nBaixando dados dos municípios (isso pode levar alguns minutos)...")
gdf_munis = geobr.read_municipality()
gdf_munis.to_file("./geodata/brazil_municipalities.json", driver="GeoJSON")
print("✓ Salvo em ./geodata/brazil_municipalities.json")

print("\nDownload concluído!")
