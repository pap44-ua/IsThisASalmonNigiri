
import pkg_resources
import subprocess
import sys

# Lista de todos los paquetes instalados
packages = [dist.project_name for dist in pkg_resources.working_set]

# Actualizar cada paquete
for package in packages:
    print(f"Actualizando {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

print("Actualización completa.")
