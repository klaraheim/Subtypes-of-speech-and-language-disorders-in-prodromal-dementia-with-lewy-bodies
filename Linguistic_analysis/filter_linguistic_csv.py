# Filtrování CSV souboru jen pro MCI-LB a HC skupiny

import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join("output_linguistic_residualised", "linguistic_residualized.csv")

OUTPUT_CSV = os.path.join(SCRIPT_DIR, "linguistic_residualized_filtered.csv")

# seznam povolených subjektů
ALLOWED_SUBJECTS = [
    # COBEN (HC)
    "COBEN_CZCOBEN002", "COBEN_CZCOBEN004", "COBEN_CZCOBEN005", "COBEN_CZCOBEN007",
    "COBEN_CZCOBEN008", "COBEN_CZCOBEN009", "COBEN_CZCOBEN010", "COBEN_CZCOBEN011",
    "COBEN_CZCOBEN012", "COBEN_CZCOBEN013", "COBEN_CZCOBEN014", "COBEN_CZCOBEN015",
    "COBEN_CZCOBEN016", "COBEN_CZCOBEN017", "COBEN_CZCOBEN018", "COBEN_CZCOBEN019",
    "COBEN_CZCOBEN021", "COBEN_CZCOBEN022", "COBEN_CZCOBEN023", "COBEN_CZCOBEN024",
    "COBEN_CZCOBEN025", "COBEN_CZCOBEN026", "COBEN_CZCOBEN027", "COBEN_CZCOBEN029",
    "COBEN_CZCOBEN030", "COBEN_CZCOBEN032", "COBEN_CZCOBEN062", "COBEN_CZCOBEN063",
    "COBEN_CZCOBEN064", "COBEN_CZCOBEN065",

    # Kardiovize (HC)
    "Kardiovize_1091", "Kardiovize_111", "Kardiovize_1141", "Kardiovize_1160",
    "Kardiovize_1175", "Kardiovize_1202", "Kardiovize_1210", "Kardiovize_1234",
    "Kardiovize_1254", "Kardiovize_1268", "Kardiovize_127", "Kardiovize_1294",
    "Kardiovize_1337", "Kardiovize_1390", "Kardiovize_1396", "Kardiovize_1404",
    "Kardiovize_1421", "Kardiovize_1453", "Kardiovize_1483", "Kardiovize_1537",
    "Kardiovize_1562", "Kardiovize_1602", "Kardiovize_1609", "Kardiovize_1628",
    "Kardiovize_1671", "Kardiovize_1705", "Kardiovize_1710", "Kardiovize_1715",
    "Kardiovize_1719", "Kardiovize_173", "Kardiovize_174", "Kardiovize_1797",
    "Kardiovize_1802", "Kardiovize_1828", "Kardiovize_1843", "Kardiovize_185",
    "Kardiovize_1873", "Kardiovize_1874", "Kardiovize_188", "Kardiovize_1903",
    "Kardiovize_1936", "Kardiovize_1937", "Kardiovize_2096", "Kardiovize_2109",
    "Kardiovize_211", "Kardiovize_2147", "Kardiovize_2161", "Kardiovize_2234",
    "Kardiovize_2249", "Kardiovize_2293", "Kardiovize_2329", "Kardiovize_2343",
    "Kardiovize_2401", "Kardiovize_2441", "Kardiovize_246", "Kardiovize_2463",
    "Kardiovize_2466", "Kardiovize_2472", "Kardiovize_2491", "Kardiovize_2492",
    "Kardiovize_2495", "Kardiovize_2537", "Kardiovize_2542", "Kardiovize_255",
    "Kardiovize_264", "Kardiovize_274", "Kardiovize_277", "Kardiovize_3",
    "Kardiovize_303", "Kardiovize_312", "Kardiovize_336", "Kardiovize_339",
    "Kardiovize_349", "Kardiovize_40", "Kardiovize_401", "Kardiovize_442",
    "Kardiovize_45", "Kardiovize_46", "Kardiovize_50", "Kardiovize_507",
    "Kardiovize_53", "Kardiovize_548", "Kardiovize_584", "Kardiovize_618",
    "Kardiovize_630", "Kardiovize_650", "Kardiovize_680", "Kardiovize_716",
    "Kardiovize_726", "Kardiovize_733", "Kardiovize_78", "Kardiovize_803",
    "Kardiovize_806", "Kardiovize_821", "Kardiovize_822", "Kardiovize_827",
    "Kardiovize_850", "Kardiovize_879", "Kardiovize_96", "Kardiovize_969",
    "Kardiovize_978",

    # Lang-LBD_pilot (HC)
    "Lang-LBD_pilot_HC1", "Lang-LBD_pilot_HC13", "Lang-LBD_pilot_HC14", "Lang-LBD_pilot_HC18",
    "Lang-LBD_pilot_HC19", "Lang-LBD_pilot_HC2", "Lang-LBD_pilot_HC20", "Lang-LBD_pilot_HC21",
    "Lang-LBD_pilot_HC22", "Lang-LBD_pilot_HC23", "Lang-LBD_pilot_HC24", "Lang-LBD_pilot_HC28",
    "Lang-LBD_pilot_HC29", "Lang-LBD_pilot_HC3", "Lang-LBD_pilot_HC30", "Lang-LBD_pilot_HC31",
    "Lang-LBD_pilot_HC32", "Lang-LBD_pilot_HC33", "Lang-LBD_pilot_HC4", "Lang-LBD_pilot_HC5",
    "Lang-LBD_pilot_HC6", "Lang-LBD_pilot_HC7", "Lang-LBD_pilot_HC8", "Lang-LBD_pilot_HC9",
    "Lang-LBD_pilot_P19",

    # preDLB (HC)
    "preDLB_HC-1#1", "preDLB_HC-13#1", "preDLB_HC-17#1", "preDLB_HC-21#1",
    "preDLB_HC-23#1", "preDLB_HC-24#1", "preDLB_HC-25#1", "preDLB_HC-28#1",
    "preDLB_HC-34#1", "preDLB_HC-6#1", "preDLB_HC-8#1",
    "preDLB_pre-LBD-33#1", "preDLB_pre-LBD-53#1", "preDLB_pre-LBD-61#1",
    "preDLB_pre-LBD-62#1", "preDLB_pre-LBD-70#1", "preDLB_pre-LBD-84#1",
    "preDLB_pre-LBD-88#1", "preDLB_pre-LBD-9#1",

    # Kardiovize (MCI-LB)
    "Kardiovize_1096", "Kardiovize_1148", "Kardiovize_1186", "Kardiovize_1222",
    "Kardiovize_1289", "Kardiovize_1336", "Kardiovize_1361", "Kardiovize_1548",
    "Kardiovize_1730", "Kardiovize_1751", "Kardiovize_1858", "Kardiovize_1872",
    "Kardiovize_1889", "Kardiovize_2027", "Kardiovize_2129", "Kardiovize_2241",
    "Kardiovize_2263", "Kardiovize_2294", "Kardiovize_2299", "Kardiovize_2330",
    "Kardiovize_2344", "Kardiovize_2372", "Kardiovize_2488", "Kardiovize_360",
    "Kardiovize_445", "Kardiovize_506", "Kardiovize_81", "Kardiovize_854",
    "Kardiovize_93", "Kardiovize_955",

    # Lang-LBD_pilot (MCI-LB)
    "Lang-LBD_pilot_P10", "Lang-LBD_pilot_P14", "Lang-LBD_pilot_P15", "Lang-LBD_pilot_P5",
    "Lang-LBD_pilot_P7",

    # preDLB (MCI-LB)
    "preDLB_HC-19#1", "preDLB_HC-9#1", "preDLB_pre-LBD-102#1", "preDLB_pre-LBD-11#1",
    "preDLB_pre-LBD-110#1", "preDLB_pre-LBD-112#1", "preDLB_pre-LBD-114#1",
    "preDLB_pre-LBD-120#1", "preDLB_pre-LBD-13#1", "preDLB_pre-LBD-15#1",
    "preDLB_pre-LBD-16#1", "preDLB_pre-LBD-17#1", "preDLB_pre-LBD-18#1",
    "preDLB_pre-LBD-2#1", "preDLB_pre-LBD-24#1", "preDLB_pre-LBD-28#1",
    "preDLB_pre-LBD-29#1", "preDLB_pre-LBD-30#1", "preDLB_pre-LBD-31#1",
    "preDLB_pre-LBD-32#1", "preDLB_pre-LBD-34#1", "preDLB_pre-LBD-37#1",
    "preDLB_pre-LBD-41#1", "preDLB_pre-LBD-44#1", "preDLB_pre-LBD-45#1",
    "preDLB_pre-LBD-49#1", "preDLB_pre-LBD-5#1", "preDLB_pre-LBD-51#1",
    "preDLB_pre-LBD-52#1", "preDLB_pre-LBD-57#1", "preDLB_pre-LBD-58#1",
    "preDLB_pre-LBD-59#1", "preDLB_pre-LBD-65#1", "preDLB_pre-LBD-69#1",
    "preDLB_pre-LBD-71#1", "preDLB_pre-LBD-72#1", "preDLB_pre-LBD-8#1",
    "preDLB_pre-LBD-83#1", "preDLB_pre-LBD-87#1", "preDLB_pre-LBD-89#1",
    "preDLB_pre-LBD-91#1", "preDLB_pre-LBD-92#1", "preDLB_pre-LBD-98#1",
    "preDLB_pre-LBD-99#1",
]


def main():

    print(f"\n Načítám: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"   Původní počet řádků: {len(df)}")

    print(f"\n Filtruji...")
    df_filtered = df[df['Subject'].isin(ALLOWED_SUBJECTS)]

    print(f"   Filtrovaný počet řádků: {len(df_filtered)}")
    print(f"   Odstraněno: {len(df) - len(df_filtered)} řádků")

    df_filtered.to_csv(OUTPUT_CSV, index=False)

    print("HOTOVO!")
    print(f"\nVýstupní soubor: {OUTPUT_CSV}")
    print(f"Počet subjektů: {len(df_filtered)}")

if __name__ == "__main__":
    main()
