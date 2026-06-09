import re
import pandas as pd


def extract_gb(memory_str, storage_type):
    pattern = rf'(\d+(?:\.\d+)?)(GB|TB)\s+{re.escape(storage_type)}'
    match = re.search(pattern, str(memory_str), re.IGNORECASE)
    if match:
        value = float(match.group(1))
        return value * 1024 if match.group(2).upper() == 'TB' else value
    return 0


def build_encoding_maps(train_df):
    """Aprende los mappings ordinales a partir del train."""
    maps = {}

    company = train_df.groupby('Company')['Price_euros'].mean().sort_values().reset_index()
    company['company_encod'] = range(len(company))
    maps['company'] = dict(zip(company['Company'], company['company_encod']))

    type_ = train_df.groupby('TypeName')['Price_euros'].mean().sort_values().reset_index()
    type_['type_encod'] = range(len(type_))
    maps['type'] = dict(zip(type_['TypeName'], type_['type_encod']))

    screen = train_df.groupby('_screen_clean')['Price_euros'].mean().sort_values().reset_index()
    screen['screen_encod'] = range(len(screen))
    maps['screen'] = dict(zip(screen['_screen_clean'], screen['screen_encod']))

    cpu = train_df.groupby('_tipo_cpu')['Price_euros'].mean().sort_values().reset_index()
    cpu['cpu_encod'] = range(len(cpu))
    maps['cpu'] = dict(zip(cpu['_tipo_cpu'], cpu['cpu_encod']))

    ghz = train_df.groupby('_ghz_cpu')['Price_euros'].mean().sort_values().reset_index()
    ghz['ghz_encod'] = range(len(ghz))
    maps['ghz'] = dict(zip(ghz['_ghz_cpu'], ghz['ghz_encod']))

    gpu = train_df.groupby('Gpu')['Price_euros'].mean().sort_values().reset_index()
    gpu['gpu_encod'] = range(len(gpu))
    maps['gpu'] = dict(zip(gpu['Gpu'], gpu['gpu_encod']))

    opsys = train_df.groupby('OpSys')['Price_euros'].mean().sort_values().reset_index()
    opsys['opsys_encod'] = range(len(opsys))
    maps['opsys'] = dict(zip(opsys['OpSys'], opsys['opsys_encod']))

    return maps


def preprocess_screen(df):
    """Extrae Touchscreen, IPS_Panel y limpia ScreenResolution."""
    df = df.copy()
    df['ScreenResolution'] = df['ScreenResolution'].str.replace(r'\d+x\d+', '', regex=True).str.strip()
    df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen').astype(int)
    df['ScreenResolution'] = (
        df['ScreenResolution']
        .str.replace('/ Touchscreen', '', regex=False)
        .str.replace('Touchscreen', '', regex=False)
        .str.strip()
    )
    df['IPS_Panel'] = df['ScreenResolution'].str.contains('IPS Panel').astype(int)
    df['ScreenResolution'] = (
        df['ScreenResolution']
        .str.replace('IPS Panel', '', regex=False)
        .str.strip()
        .str.replace(r'\s*/\s*', ' ', regex=True)
        .str.strip()
    )
    df['_screen_clean'] = df['ScreenResolution']
    return df


def preprocess_cpu(df):
    """Extrae ghz_cpu y familia de CPU."""
    df = df.copy()
    df['_ghz_cpu'] = df['Cpu'].str.extract(r'(\d+\.?\d*)GHz', expand=False).astype(float)
    df['_tipo_cpu'] = df['Cpu'].str.extract(
        r'^(Intel Core i\d+|Intel Pentium|Intel Celeron|Intel Atom|Intel Xeon|'
        r'AMD [A-Z]\d+-Series|AMD E-Series|AMD FX|AMD Ryzen \d+|AMD A-Series)',
        expand=False
    )
    return df


def transform(df, maps, has_price=True):
    df = df.copy()

    # Columnas que no aportan
    df = df.drop(columns=['laptop_ID', 'Product'])

    # Company
    df['company_encod'] = df['Company'].map(maps['company']).fillna(0).astype(int)
    df = df.drop(columns='Company')

    # TypeName
    df['type_encod'] = df['TypeName'].map(maps['type']).fillna(0).astype(int)
    df = df.drop(columns='TypeName')

    # ScreenResolution
    df = preprocess_screen(df)
    df['screen_encod'] = df['_screen_clean'].map(maps['screen']).fillna(0).astype(int)
    df = df.drop(columns=['ScreenResolution', '_screen_clean'])

    # CPU
    df = preprocess_cpu(df)
    df['ghz_cpu'] = df['_ghz_cpu']
    df['cpu_encod'] = df['_tipo_cpu'].map(maps['cpu']).fillna(0).astype(int)
    df['ghz_encod'] = df['_ghz_cpu'].map(maps['ghz']).fillna(0).astype(int)
    df = df.drop(columns=['Cpu', '_ghz_cpu', '_tipo_cpu'])

    # RAM
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)

    # Memory
    df['memory_ssd']   = df['Memory'].apply(lambda x: extract_gb(x, 'SSD'))
    df['memory_hdd']   = df['Memory'].apply(lambda x: extract_gb(x, 'HDD'))
    df['memory_flash'] = df['Memory'].apply(lambda x: extract_gb(x, 'Flash Storage'))
    df = df.drop(columns='Memory')

    # GPU
    df['gpu_encod'] = df['Gpu'].map(maps['gpu']).fillna(0).astype(int)
    df = df.drop(columns='Gpu')

    # OpSys
    df['opsys_encod'] = df['OpSys'].map(maps['opsys']).fillna(0).astype(int)
    df = df.drop(columns='OpSys')

    # Weight
    df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

    # Reordenar columnas igual que train_trans
    cols = ['id', 'Inches', 'Ram', 'Weight']
    if has_price:
        cols.append('Price_euros')
    cols += [
        'company_encod', 'type_encod', 'Touchscreen', 'IPS_Panel',
        'screen_encod', 'ghz_cpu', 'cpu_encod', 'ghz_encod',
        'memory_ssd', 'memory_hdd', 'memory_flash', 'gpu_encod', 'opsys_encod'
    ]
    return df[cols]


if __name__ == '__main__':
    train_raw = pd.read_csv('data/train.csv')
    test_raw  = pd.read_csv('data/test.csv')

    # Preprocesar columnas intermedias en train para poder construir los maps
    train_prep = preprocess_screen(train_raw)
    train_prep = preprocess_cpu(train_prep)

    maps = build_encoding_maps(train_prep)

    train_trans = transform(train_raw, maps, has_price=True)
    test_trans  = transform(test_raw,  maps, has_price=False)

    train_trans.to_csv('data/train_trans.csv', index=False)
    test_trans.to_csv('data/test_trans.csv',   index=False)

    print(f"train_trans: {train_trans.shape}")
    print(f"test_trans:  {test_trans.shape}")
