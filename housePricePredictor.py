import pandas as pd
import numpy as np

df = pd.read_csv("home_price.csv")

# Çok eksik olduğu için bazı sütunları siliyoruz
df.drop(["Eşya_Durumu", "Tapu_Durumu", "Yatırıma_Uygunluk", "Kullanım_Durumu", "Takas"], axis=1, inplace=True)

# Sayısal eksik değer doldurma
df["Oda_Sayısı"] = df["Oda_Sayısı"].fillna(df["Oda_Sayısı"].median())
# Banyo sayısı girilmediyse 1 dir
df["Banyo_Sayısı"] = df["Banyo_Sayısı"].fillna(1)

# Şehir grubu
def sehir_grubu_ata(sehir):
    sehir = sehir.lower()
    metropol = ["istanbul", "ankara", "izmir"]
    buyuk = ["antalya", "bursa",  "konya", "adana", "mersin"]
    orta = ["gaziantep", "kayseri",  "samsun", "trabzon"]
    if sehir in buyuk:
        return "büyükşehir"
    elif sehir in metropol:
        return "metropol"
    elif sehir in orta:
        return "orta"
    else:
        return "küçük"

df["Şehir_Grubu"] = df["Şehir"].apply(sehir_grubu_ata)
df.drop(columns=["Şehir"], inplace=True)
df = pd.get_dummies(df, columns=["Şehir_Grubu"], drop_first=False)

# Isıtma tipi
df = pd.get_dummies(df, columns=["Isıtma_Tipi"], drop_first=False)

# Kat bilgisi kategorik dönüşüm
df["Bulunduğu_Kat"] = df["Bulunduğu_Kat"].fillna(df["Bulunduğu_Kat"].mode()[0])

def temizle_bina_yasi(x):
    if isinstance(x, str):
        if x == "0 (Yeni)":
            return 0
        elif '-' in x:
            alt, ust = x.split('-')
            return (int(alt) + int(ust)) / 2
        elif 'Ve Üzeri' in x:
            return 25
        else:
            return int(x)
    return x

df["Binanın_Yaşı"] = df["Binanın_Yaşı"].apply(temizle_bina_yasi)

# Yeni özellikler
df["Net_Brut_Orani"] = df["Net_Metrekare"] / df["Brüt_Metrekare"]
df["Metrekare_Farkı"] = df["Brüt_Metrekare"] - df["Net_Metrekare"]
df = df[df["Net_Brut_Orani"] > 0.3]