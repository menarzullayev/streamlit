# Rang xaritasi - asosiy vizuallashtirishlar uchun
COLOR_MAP_MAIN = {
    'True Age (GT)': '#808080',  # Kulrang
    'V201': '#FF4B4B',           # Qizil (Asosiy Kamera)
    'HD': '#00BFFF',             # Moviy
    'FHD': '#00CC66'             # Yashil
}

# Jami kameralar ro'yxati
CAMERAS = ['v201', 'hd', 'fhd']

# Yosh guruhlari (Tavsiya #8)
AGE_BINS = [0, 18, 40, 100]
AGE_LABELS = ['0-18 (Youth)', '19-40 (Adult)', '40+ (Senior)']

# Haqiqiy va tozalangan ustun nomlari
TRUE_FILENAME = 'filename'
TRUE_AGE = 'age'
TRUE_GENDER = 'gender'
TRUE_WORKER = 'worker'

V201_AGE_CLEAN = 'v201_age'
HD_AGE_CLEAN = 'hd_age'
FHD_AGE_CLEAN = 'fhd_age'