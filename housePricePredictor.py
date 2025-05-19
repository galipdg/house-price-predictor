import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import cross_val_score

df = pd.read_csv("home_price.csv")

# Çok eksik olduğu için sütunları siliyoruz
df.drop(["Eşya_Durumu", "Tapu_Durumu","Yatırıma_Uygunluk", "Kullanım_Durumu", "Takas"], axis=1, inplace=True)

# Sayısal verideki eksikler için medyan
df["Oda_Sayısı"] = df["Oda_Sayısı"].fillna(df["Oda_Sayısı"].median())

# Banyo sayısı girilmediyse 1 tane vardır
df["Banyo_Sayısı"] = df["Banyo_Sayısı"].fillna(1)


# Şehirleri kategorilere ayır
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

# Orijinal şehir ismini yeniden alabilmek için df'yi tekrar oku
df["Şehir_Grubu"] = df["Şehir"].apply(sehir_grubu_ata)
df.drop(columns=["Şehir"], inplace=True)

# Şehir grubu için one-hot encoding
df = pd.get_dummies(df, columns=["Şehir_Grubu"], drop_first=False)

# Isıtma tipi için one-hot encoding
df = pd.get_dummies(df, columns=["Isıtma_Tipi"], drop_first=False)

# Kategorik olanlar için mode
df["Bulunduğu_Kat"] = df["Bulunduğu_Kat"].fillna(df["Bulunduğu_Kat"].mode()[0])

df["Net_Brut_Orani"] = df["Net_Metrekare"] / df["Brüt_Metrekare"]
df = df[df["Net_Brut_Orani"] > 0.3]  # Oranı %30'dan büyük olanları tut

def temizle_bina_yasi(x): # bina yaşı object tipindeydi floata çeviriyoruz
    if isinstance(x, str):
        if x == "0 (Yeni)": # Yeni binaların yaşı 0
            return 0
        elif '-' in x: # - geçen bi aralık girildiyse araalığın ortalamasını gir
            alt, ust = x.split('-')
            return (int(alt) + int(ust)) / 2
        elif 'Ve Üzeri' in x: # ve üzeri geçen 21 ve üzeri oluyor onlara 25 atadık
            return 25
        else:
            return int(x)
    else:
        return x

df["Binanın_Yaşı"] = df["Binanın_Yaşı"].apply(temizle_bina_yasi)

def temizle_bulundugu_kat(x):
    if isinstance(x, str):
        if "Müstakil" in x or "Villa Tipi" in x:
            return "müstakil"
        elif "Kot" in x:
            return "zemin_altı"
        elif x == "Bodrum Kat":
            return "bodrum"
        elif x == "Bahçe Katı" or x == "Bahçe Dublex":
            return "bahçe"
        elif x == "Yüksek Giriş" or x == "Düz Giriş (Zemin)":
            return "zemin"
        elif x == "Çatı Katı" or x == "Çatı Dubleks":
            return "çatı"
        elif "Kat" in x and (x[0].isdigit() or x.startswith("40+")):
            try:
                kat_num = int(x.split(".")[0].replace("+", ""))
                if kat_num >= 10:
                    return "lüks_daire"
                else:
                    return "normal"
            except:
                return "belirsiz"
        else:
            return "belirsiz"
    return "belirsiz"
df["Bulunduğu_Kat"] = df["Bulunduğu_Kat"].apply(temizle_bulundugu_kat)


# Bulunduğu_Kat sütununu one-hot encoding'e çevir
df = pd.get_dummies(df, columns=["Bulunduğu_Kat"], drop_first=True)

columns_to_check = [
    "Net_Metrekare",
    "Brüt_Metrekare",
    "Fiyat",
    "Binanın_Yaşı",
    "Oda_Sayısı",
    "Banyo_Sayısı",
    "Binanın_Kat_Sayısı"
]

# Yeni özellikler (Feature Engineering)
df["Metrekare_Farkı"] = df["Brüt_Metrekare"] - df["Net_Metrekare"]
#df["Birim_Fiyat"] = df["Fiyat"] / df["Net_Metrekare"]
df["Çok_Katlı_Bina"] = (df["Binanın_Kat_Sayısı"] > 10).astype(int)

# Boxplot ile outlier analiz görselleştirmesi
for col in columns_to_check:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"{col} Boxplot")
    plt.tight_layout()
    plt.savefig(f"{col}_boxplot.png")
    plt.close()

# --- Şehir Bazlı Clip ---
buyuk_df = df[df["Şehir_Grubu_büyükşehir"] == 1].copy()
diger_df = df[df["Şehir_Grubu_büyükşehir"] != 1].copy()

# Büyükşehirlerde yüksek değerler olabilir
buyuk_df["Fiyat"] = buyuk_df["Fiyat"].clip(upper=25000000)
buyuk_df["Net_Metrekare"] = buyuk_df["Net_Metrekare"].clip(upper=600)
buyuk_df["Brüt_Metrekare"] = buyuk_df["Brüt_Metrekare"].clip(upper=800)

# Diğer şehirlerde daha sıkı sınırlar
diger_df["Fiyat"] = diger_df["Fiyat"].clip(upper=10000000)
diger_df["Net_Metrekare"] = diger_df["Net_Metrekare"].clip(upper=400)
diger_df["Brüt_Metrekare"] = diger_df["Brüt_Metrekare"].clip(upper=600)

# Tekrar birleştir
df = pd.concat([buyuk_df, diger_df], ignore_index=True)

# Diğer sütunlar için genel clip
df["Oda_Sayısı"] = df["Oda_Sayısı"].clip(upper=8)
df["Banyo_Sayısı"] = df["Banyo_Sayısı"].clip(upper=5)
df["Binanın_Kat_Sayısı"] = df["Binanın_Kat_Sayısı"].clip(upper=25)


def evaluate_model(model, df, target="Fiyat"):
    X = df.drop(columns=[target])
    if "Log_Fiyat" in X.columns:
        X = X.drop(columns=["Log_Fiyat"])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    print(f"{model.__class__.__name__} R2: {r2:.4f}")
    return r2

# FEATURE IMPORTANCE GÖRSELLEŞTİRME FONKSİYONU
def plot_feature_importance(df, target="Fiyat"):
    X = df.drop(columns=[target])
    if "Log_Fiyat" in X.columns:
        X = X.drop(columns=["Log_Fiyat"])
    y = df[target]
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values(ascending=True).plot(kind="barh", figsize=(10, 8), title="Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

print("\n--- Model Performans Karşılaştırması ---")
evaluate_model(LinearRegression(), df)
evaluate_model(DecisionTreeRegressor(max_depth=10, min_samples_split=10, random_state=42), df)
evaluate_model(RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    max_features="sqrt",
    random_state=42
), df)

print("\n--- Öznitelik Önem Değerleri (Feature Importances) ---")
plot_feature_importance(df)

# Log dönüşümlü fiyat boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=np.log1p(df["Fiyat"]))
plt.title("Boxplot: Log(Fiyat)")
plt.tight_layout()
plt.savefig("outlier_plots/Log_Fiyat_boxplot.png")
plt.close()
