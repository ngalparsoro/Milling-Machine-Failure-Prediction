"""
Copia los ficheros _hfdata.csv que faltan en kit_extracted/
desde la descarga completa del dataset KIT.

Uso:
    python recover_kit_missing.py --source <ruta_al_directorio_descomprimido>

Ejemplo:
    python recover_kit_missing.py --source ./10.35097-hvvwn1kfwf7qt48z/data/dataset/Data/Dataset
"""
import os, shutil, argparse

MISSING = [
    # Normales
    'IM-01F', 'IM-01R',
    'IMP-01', 'IMP-03', 'IMP-04', 'IMP-06', 'IMP-09', 'IMP-10', 'IMP-12',
    'TF-02',
    # Anomalías
    'IM-01R-A01', 'IM-01R-A03', 'IM-01R-A04',
    'IMP-01-A02',
    'TF-03-A02',
]

KIT_RECOVERED = os.path.join(os.path.dirname(__file__), 'kit_extracted')

def find_and_copy(source_root):
    found, not_found = [], []

    for trial in MISSING:
        target = os.path.join(KIT_RECOVERED, f'{trial}_hfdata.csv')
        if os.path.exists(target):
            print(f'  ya existe: {trial}_hfdata.csv')
            found.append(trial)
            continue

        # Buscar recursivamente en el directorio fuente
        hit = None
        for root, _, files in os.walk(source_root):
            fname = f'{trial}_hfdata.csv'
            if fname in files:
                hit = os.path.join(root, fname)
                break

        if hit:
            shutil.copy2(hit, target)
            size_mb = os.path.getsize(target) / 1e6
            print(f'  copiado: {trial}_hfdata.csv ({size_mb:.1f} MB)')
            found.append(trial)
        else:
            print(f'  NO ENCONTRADO: {trial}_hfdata.csv')
            not_found.append(trial)

    print(f'\nResumen: {len(found)} copiados | {len(not_found)} no encontrados')

    if not not_found:
        total = len([f for f in os.listdir(KIT_RECOVERED) if f.endswith('_hfdata.csv')])
        print(f'kit_extracted ahora completo: {total} experimentos')
    else:
        print('Ficheros no encontrados:', not_found)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True,
                        help='Ruta al directorio Dataset descomprimido')
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f'Error: directorio no encontrado: {args.source}')
        exit(1)

    print(f'Buscando en: {args.source}')
    print(f'Destino: {KIT_RECOVERED}')
    print()
    find_and_copy(args.source)
