# ESG_thesis
20262820

# ESG 子因子與財務績效之關聯性分析
### An Analysis of the Association between ESG Sub-factors and Financial Performance
結合可解釋機器學習與雙向固定效果模型之台灣上市公司實證
*Explainable Machine Learning and Two-Way Fixed Effects Models — Evidence from Taiwan-Listed Companies*

國立中興大學資訊管理學系碩士論文原始碼 / Master's thesis code, Department of Management Information Systems, National Chung Hsing University

**作者 / Author**：劉囿廷 Yu-Ting Liu
**指導教授 / Advisors**：林冠成 Kuan-Cheng Lin、吳君怡 Jun-Yi Wu

---

## 專案簡介

既有 ESG 與財務績效研究多以第三方評級機構之綜合評分為衡量指標，惟不同機構評分相關性偏低，且綜合評分易掩蓋子指標間之方向差異、難以捕捉非線性關係；純機器學習方法則缺乏嚴謹推論基礎。

本研究以台灣 ESG 數位平台之原始申報數值取代第三方合成評分，建立**結合可解釋機器學習與雙向固定效果模型（TWFE）之兩階段分析框架**，檢視台灣上市公司 ESG 子因子與財務績效間之**組內時間序列關聯**（within-firm time-series association）。

- **樣本**：401 家非金融業台灣上市公司、1,111 筆公司年觀察值（2022–2024）
- **第一階段（特徵篩選）**：XGBoost、Random Forest 為預測模型，透過 SHAP 值與 Bootstrap 穩定性驗證，自 137 個候選 ESG 子因子中篩選出候選子因子，再經 VIF／相關矩陣共線性診斷，得到 11 個最終子因子（E=1、S=4、G=6）
- **第二階段（統計推論）**：雙向固定效果模型（公司固定效果 + 年度固定效果）估計最終 ESG 子因子與財務績效（ROE、Tobin's Q；ROA、EPS 於前期版本中曾納入，EPS 因過度配適已剔除）之關聯，並以 Benjamini–Hochberg FDR 進行多重檢定修正、以事後選擇推論（post-selection inference）驗證篩選穩健性

本研究之核心貢獻為**方法論**：建立一套可驗證、可重製的 ESG 子因子—財務績效篩選框架，而非宣稱特定 ESG 子因子「驅動」財務績效。多項穩健性檢驗誠實呈現：多數機器學習篩選出之候選子因子，並未轉化為穩定之組內統計關聯，且既有邊際顯著結果具樣本特定性與產業侷限性（詳見下方〈主要研究發現〉）。

---

## 研究架構

```
原始資料（ESG 數位平台 + TEJ）
        │
        ▼
資料前處理與長格式追蹤資料（panel data）建置
        │
        ▼
┌───────────────────────────────┐
│  第一階段：可解釋機器學習特徵篩選    │
│  XGBoost + Random Forest        │
│  → SHAP 值 + Bootstrap 穩定性驗證 │
│  → VIF / 相關矩陣共線性診斷        │
│  → 11 個最終 ESG 子因子           │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  第二階段：雙向固定效果模型（TWFE） │
│  公司 FE + 年度 FE                │
│  依變數：ROE、Tobin's Q          │
│  控制變數：含產業年度平均績效      │
│           （leave-one-out, ind_avg）│
│  → Benjamini–Hochberg FDR 修正   │
│  → 多項穩健性 / 事後推論驗證       │
└───────────────────────────────┘
        │
        ▼
    研究結果與穩健性分析
```

---

## Notebook 說明

程式碼以 Google Colab 撰寫並依序執行，四份 notebook 共用同一套資料前處理流程（Cell 1–29），差異在於後段的建模設定與穩健性檢驗重點：

| Notebook | 對應內容 |
|---|---|
| `ESG_Data_Processing_ver2.ipynb` | **主要分析流程**：資料前處理 → ML_SET 建置 → 描述性統計 → 公司層級切割訓練（XGBoost / RF / 線性迴歸）→ Bootstrap SHAP 穩定性分析 → 相關矩陣／VIF 診斷 → 四層固定效果模型比較（Model 1–4）→ 產業固定效果補充模型 → 落後一期／排除金融業／Winsorization 穩健性檢驗 → 產業重大性分析 → FDR 多重檢定修正 |
| `ESG_Data_Processing_ver2_補充實驗.ipynb` | **補充實驗與論文第 4.5–4.7 節重寫版本**：在主模型（`panel_fe_common`）基礎上，追加公司層級 50/50 雙向切割驗證、同期／落後一期／反向 Placebo 時間方向比較、靜態 vs 動態 TWFE 比較、產業交互作用檢定（含 FDR 修正）、標準化 Beta 係數、組內／組間變異分解（Within/Between Variation） |
| `ESG_Data_Processing_ver_nobalance.ipynb` | **不平衡追蹤資料（unbalanced panel）穩健性版本**：放寬主模型之 ESG 揭露完整性要求，檢驗核心結果是否僅存在於「ESG 揭露完整」樣本，而非具一般性 |
| `ESG_Data_Processing_ver_博士.ipynb` | **員工具博士學歷比例延伸穩健性分析**：排除台積電、排除規模前 1% 極端值、逐一排除法（Leave-One-Out by industry）、自變數縮尾、與生技製藥業之產業交互作用檢定，檢驗「博士% → Tobin's Q」邊際顯著結果是否受單一公司或極端值驅動 |

> 四份 notebook 之 Cell 1–29（資料讀取、多檔案合併、面板資料建置）內容相同，僅後續分析段落不同；後續整理時建議保留 `ver2_補充實驗.ipynb` 作為主線（涵蓋 4.5–4.7 節最終版本），其餘三份可整理為 `robustness/` 子目錄下之獨立穩健性檢驗腳本，以降低重複程式碼。

---

## 主要研究發現

- 11 個最終 ESG 子因子模型，相較於單一公司治理評鑑分數，在組內解釋力上呈現較高表現，惟 AIC／BIC 等資訊準則之比較結果並不完全一致，模型優勢須審慎解讀
- 主模型下，ROE 與 Tobin's Q 之 ESG 子因子關聯，經 FDR 修正後均未通過顯著門檻
- 員工具博士學歷比例（博士%）對 Tobin's Q 於 p<0.1 邊際顯著，惟於不平衡追蹤資料穩健性檢驗中消失（p=0.902），顯示該結果可能僅適用於 ESG 揭露完整之公司子樣本
- 三組理論驅動之產業交互作用檢定中，僅「博士% × 生技醫療業」達顯著，並為唯一通過 FDR 修正之結果

---

## 環境需求

程式碼於 **Google Colab** 開發，並掛載 **Google Drive** 讀寫資料與結果，本機執行前需調整路徑。

主要套件：

```
pandas, numpy, scipy
scikit-learn（RandomForestRegressor, Lasso, LassoCV, ElasticNetCV, LinearRegression, StandardScaler, LabelEncoder）
xgboost
shap
statsmodels（VIF、multipletests / FDR）
linearmodels（PanelOLS, PooledOLS）
matplotlib, seaborn
tqdm
```

安裝範例：

```bash
pip install pandas numpy scipy scikit-learn xgboost shap statsmodels linearmodels matplotlib seaborn tqdm
```

---

## 資料說明

本研究資料來源為：

- **台灣 ESG 數位平台**（ESG 子因子原始申報數值）
- **TEJ 台灣經濟新報資料庫**（財務績效與控制變數）
