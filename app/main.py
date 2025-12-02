from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from io import BytesIO
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import json
import logging
import math

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forecast_api")

# ---------------------------
# App + CORS
# ---------------------------
app = FastAPI(title="Forecast API - semanal/mensal/quinzenal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Helpers: parsing / sanitização
# ---------------------------

def sheet_name_to_year(name: str) -> Optional[int]:
    """Tenta extrair um ano inteiro de um nome de aba."""
    try:
        y = int(name)
        if 1900 <= y <= 2100:
            return y
    except Exception:
        pass
    return None

def sanitize_number(x: Any) -> Optional[float]:
    """Converte para float válido ou retorna None se NaN/Inf/None/inválido."""
    if x is None:
        return None
    try:
        # pandas.isna cobre NaN e None
        if pd.isna(x):
            return None
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None

def sanitize_list(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitiza lista de objetos {'date':..., 'value': ...} removendo valores inválidos."""
    cleaned = []
    for item in lst:
        try:
            val = sanitize_number(item.get("value"))
            if val is None:
                continue
            cleaned.append({"date": item.get("date"), "value": float(val)})
        except Exception:
            continue
    return cleaned

def safe_series_to_list(series: pd.Series) -> List[Dict[str, Any]]:
    """Converte uma Series para lista de dicts sanitizada."""
    res = []
    for idx, val in series.items():
        sval = sanitize_number(val)
        if sval is None:
            continue
        # idx pode ser Timestamp ou Period
        date_str = None
        try:
            date_str = str(pd.to_datetime(idx).date())
        except Exception:
            date_str = str(idx)
        res.append({"date": date_str, "value": float(round(sval, 2))})
    return res

def ensure_forecast_values(raw_forecast, forecast_index):
    """
    Recebe raw_forecast (iterável) e índice (DatetimeIndex),
    sanitiza e devolve pd.Series com valores float e sem NaN.
    Se valor inválido, usa fallback (0.0) e loga o evento.
    """
    cleaned = []
    for i, val in enumerate(raw_forecast):
        v = sanitize_number(val)
        if v is None:
            logger.warning(f"Forecast gerou valor inválido na posição {i}. Substituindo por 0.0")
            v = 0.0
        cleaned.append(v)
    try:
        s = pd.Series(cleaned, index=forecast_index).round(2)
    except Exception:
        # fallback sem índice
        s = pd.Series(cleaned).round(2)
    return s

def auto_adjust_seasonality(length: int, requested: int) -> Optional[int]:
    """
    Ajusta sazonalidade automaticamente:
    - se length < requested * 2 -> provavelmente não há dados suficientes para seasonal
    - retorna None para desativar sazonalidade, ou um valor menor adequado
    """
    if requested is None:
        return None
    if length < max(6, requested * 2):
        # tenta reduzir para metade até caber, ou desativa
        alt = requested // 2
        while alt >= 2:
            if length >= alt * 2:
                return alt
            alt = alt // 2
        return None
    return requested

# ---------------------------
# Modelo Pydantic (documentação)
# ---------------------------
class ForecastRequest(BaseModel):
    base_sheet: Optional[str] = None
    base_sheets: Optional[List[str]] = None
    target_year: int
    date_col: str = "Data"
    value_col: str = "Valor"
    seasonality: Optional[int] = None


def parse_excel_bytes(contents: bytes) -> Dict[str, pd.DataFrame]:
    excel = pd.ExcelFile(BytesIO(contents))
    dfs = {}
    for s in excel.sheet_names:
        try:
            dfs[s] = excel.parse(s)
        except Exception as e:
            logger.debug(f"Ignorando aba {s} (erro ao parsear): {e}")
    return dfs

# ---------------------------
# ENDPOINT: forecast_periodo_mensal
# ---------------------------
@app.post("/forecast_periodo_mensal")
async def forecast_periodo_mensal(
    file: UploadFile = File(...),
    json_data: str = Form(...)
):
    params = json.loads(json_data)
    # compatibilidade legado / novo
    base_sheets = params.get("base_sheets")
    base_sheet = params.get("base_sheet")
    if base_sheets is None and base_sheet:
        base_sheets = [base_sheet]
    if base_sheets is None:
        return {"erro": "Informe base_sheets (lista) ou base_sheet (única)."}

    target_year = int(params.get("target_year"))
    date_col = params.get("date_col", "Data")
    value_col = params.get("value_col", "Valor")
    requested_seasonality = int(params.get("seasonality", 12))

    contents = await file.read()
    dfs_by_sheet = parse_excel_bytes(contents)

    missing = [s for s in base_sheets if s not in dfs_by_sheet]
    if missing:
        return {"erro": f"Abas não encontradas: {missing}. Abas disponíveis: {list(dfs_by_sheet.keys())}"}

    # combinar abas selecionadas
    df_list = []
    parsed_years = []
    for s in base_sheets:
        df_s = dfs_by_sheet[s].copy()
        df_s["_sheet"] = s
        df_list.append(df_s)
        y = sheet_name_to_year(s)
        if y is not None:
            parsed_years.append(y)

    combined = pd.concat(df_list, ignore_index=True)
    combined[date_col] = pd.to_datetime(combined[date_col], dayfirst=True, errors="coerce")
    combined = combined.dropna(subset=[date_col])
    if combined.empty:
        return {"erro": "As abas selecionadas não contêm datas válidas."}
    combined = combined.sort_values(date_col).set_index(date_col)

    # agregação mensal
    series_base_monthly = combined[value_col].resample("M").sum()

    # valida tamanho minimo
    if len(series_base_monthly.dropna()) < 3:
        return {"erro": "Poucos dados mensais nas abas base selecionadas."}

    max_base_year = max(parsed_years) if parsed_years else None
    if max_base_year and target_year <= max_base_year:
        return {"erro": "Ano alvo deve ser maior que o maior ano base selecionado."}

    if max_base_year:
        forecast_start = pd.to_datetime(f"{max_base_year+1}-01-01")
    else:
        last = series_base_monthly.index[-1]
        forecast_start = (last + pd.offsets.MonthEnd()).normalize()

    forecast_end = pd.to_datetime(f"{target_year}-12-31")
    forecast_index = pd.date_range(start=forecast_start, end=forecast_end, freq="M")
    future_periods = len(forecast_index)
    if future_periods <= 0:
        return {"erro": "Período de previsão inválido (verifique anos base/target)."}

    # ajustar sazonalidade automaticamente
    usable_len = len(series_base_monthly.dropna())
    seasonality = auto_adjust_seasonality(usable_len, requested_seasonality)
    logger.info(f"[mensal] usable_len={usable_len}, requested={requested_seasonality}, usando seasonality={seasonality}")

    # treinar modelo com tentativa de sazonalidade quando possível
    try:
        if seasonality and seasonality >= 2:
            model = ExponentialSmoothing(series_base_monthly, trend="add", seasonal="add", seasonal_periods=seasonality).fit()
        else:
            model = ExponentialSmoothing(series_base_monthly, trend="add").fit()
    except Exception as e:
        logger.warning(f"[mensal] Erro treinando Holt-Winters com sazonalidade: {e}. Tentando sem sazonalidade.")
        model = ExponentialSmoothing(series_base_monthly, trend="add").fit()

    raw_forecast = None
    try:
        raw_forecast = model.forecast(future_periods)
    except Exception as e:
        logger.error(f"[mensal] Erro ao gerar forecast: {e}")
        raw_forecast = [0.0] * future_periods

    forecast_values = ensure_forecast_values(raw_forecast, forecast_index)

    # montar retorno histórico e reais por ano (se houver abas por ano)
    historico_base_list = safe_series_to_list(series_base_monthly)

    # reais por ano: tenta mapear abas nomeadas por ano entre max_base_year+1 e target_year
    reais_por_ano = {}
    sheet_year_map = {}
    for s in dfs_by_sheet:
        y = sheet_name_to_year(s)
        if y is not None:
            sheet_year_map[y] = s

    if max_base_year:
        for y in range(max_base_year + 1, target_year + 1):
            if y in sheet_year_map:
                df_y = dfs_by_sheet[sheet_year_map[y]].copy()
                df_y[date_col] = pd.to_datetime(df_y[date_col], dayfirst=True, errors="coerce")
                df_y = df_y.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
                series_y = df_y[value_col].resample("M").sum()
                series_y = series_y[series_y.index.year == y]
                reais_por_ano[y] = safe_series_to_list(series_y)

    # sanitização final das listas para evitar NaN/Inf no JSON
    historico_base_list = sanitize_list(historico_base_list)
    reais_por_ano = {ano: sanitize_list(lista) for ano, lista in reais_por_ano.items()}
    previsao_list = safe_series_to_list(forecast_values)

    # resposta
    return {
        "base_sheets": base_sheets,
        "base_years_guess": parsed_years,
        "target_year": target_year,
        "historico_base": historico_base_list,
        "reais_por_ano": reais_por_ano,
        "previsao": previsao_list,
        "forecast_start": str(forecast_index[0].date()),
        "forecast_end": str(forecast_index[-1].date()),
        "future_months": future_periods,
        "seasonality_used": seasonality
    }

@app.post("/forecast_periodo")
async def forecast_periodo(
    file: UploadFile = File(...),
    json_data: str = Form(...)
):
    params = json.loads(json_data)
    base_sheets = params.get("base_sheets")
    base_sheet = params.get("base_sheet")
    if base_sheets is None and base_sheet:
        base_sheets = [base_sheet]
    if base_sheets is None:
        return {"erro": "Informe base_sheets (lista) ou base_sheet (única)."}

    target_year = int(params.get("target_year"))
    date_col = params.get("date_col", "Data")
    value_col = params.get("value_col", "Valor")
    requested_seasonality = int(params.get("seasonality", 24)) # ===== ALTERAR SAZONALIDADE 

    contents = await file.read()
    dfs_by_sheet = parse_excel_bytes(contents)

    missing = [s for s in base_sheets if s not in dfs_by_sheet]
    if missing:
        return {"erro": f"Abas não encontradas: {missing}. Abas disponíveis: {list(dfs_by_sheet.keys())}"}

    # combinar abas selecionadas
    df_list = []
    parsed_years = []
    for s in base_sheets:
        df_s = dfs_by_sheet[s].copy()
        df_s["_sheet"] = s
        df_list.append(df_s)
        y = sheet_name_to_year(s)
        if y is not None:
            parsed_years.append(y)

    combined = pd.concat(df_list, ignore_index=True)
    combined[date_col] = pd.to_datetime(combined[date_col], dayfirst=True, errors="coerce")
    combined = combined.dropna(subset=[date_col])
    if combined.empty:
        return {"erro": "As abas selecionadas não contêm datas válidas."}
    combined = combined.sort_values(date_col).set_index(date_col)

    # agregação semanal (domingo como final de semana)
    series_base_weekly = combined[value_col].resample("W-SUN").sum()

    if len(series_base_weekly.dropna()) < 4:
        return {"erro": "Poucos dados no ano(s) base após agregação semanal. Necessário pelo menos 4 semanas no conjunto combinado."}

    max_base_year = max(parsed_years) if parsed_years else None
    if max_base_year and target_year <= max_base_year:
        return {"erro": "Ano alvo deve ser maior que o maior ano base selecionado."}

    if max_base_year:
        forecast_start = pd.to_datetime(f"{max_base_year + 1}-01-01")
    else:
        forecast_start = series_base_weekly.index[-1] + pd.Timedelta(weeks=1)

    forecast_end = pd.to_datetime(f"{target_year}-12-31")
    forecast_index = pd.date_range(start=forecast_start, end=forecast_end, freq="W-SUN")
    if len(forecast_index) == 0:
        return {"erro": "Período de previsão inválido (verifique anos base/target)."}

    future_periods = len(forecast_index)

    usable_len = len(series_base_weekly.dropna())
    #seasonality = auto_adjust_seasonality(usable_len, requested_seasonality)
    seasonality = 24 # ===== ALTERAR SAZONALIDADE 
    logger.info(f"[semanal] usable_len={usable_len}, requested={requested_seasonality}, usando seasonality={seasonality}")

    try:
        if seasonality and seasonality >= 2:
            model = ExponentialSmoothing(series_base_weekly, trend="add", seasonal="add", seasonal_periods=seasonality).fit()
        else:
            model = ExponentialSmoothing(series_base_weekly, trend="add").fit()
    except Exception as e:
        logger.warning(f"[semanal] Erro treinando modelo: {e}. Tentando sem sazonalidade.")
        model = ExponentialSmoothing(series_base_weekly, trend="add").fit()

    try:
        raw_forecast = model.forecast(future_periods)
    except Exception as e:
        logger.error(f"[semanal] Erro ao gerar forecast: {e}")
        raw_forecast = [0.0] * future_periods

    forecast_values = ensure_forecast_values(raw_forecast, forecast_index)

    historico_base_list = safe_series_to_list(series_base_weekly)

    reais_por_ano = {}
    sheet_year_map = {}
    for s in dfs_by_sheet:
        y = sheet_name_to_year(s)
        if y is not None:
            sheet_year_map[y] = s

    if max_base_year:
        for y in range(max_base_year + 1, target_year + 1):
            if y in sheet_year_map:
                sheet_name = sheet_year_map[y]
                df_y = dfs_by_sheet[sheet_name].copy()
                df_y[date_col] = pd.to_datetime(df_y[date_col], dayfirst=True, errors='coerce')
                df_y = df_y.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
                series_y_weekly = df_y[value_col].resample("W-SUN").sum()
                series_y_weekly = series_y_weekly[series_y_weekly.index.year == y]
                if len(series_y_weekly) > 0:
                    reais_por_ano[y] = safe_series_to_list(series_y_weekly)

    historico_base_list = sanitize_list(historico_base_list)
    reais_por_ano = {ano: sanitize_list(lista) for ano, lista in reais_por_ano.items()}
    previsao_list = safe_series_to_list(forecast_values)

    return {
        "base_sheets": base_sheets,
        "base_years_guess": parsed_years,
        "target_year": target_year,
        "forecast_start": str(forecast_index[0].date()),
        "forecast_end": str(forecast_index[-1].date()),
        "historico_base": historico_base_list,
        "reais_por_ano": reais_por_ano,
        "previsao": previsao_list,
        "seasonal_periods_used": seasonality,
        "future_weeks": future_periods
    }

# ---------------------------
# ENDPOINT: forecast_periodo_quinzenal
# ---------------------------
@app.post("/forecast_periodo_quinzenal")
async def forecast_periodo_quinzenal(
    file: UploadFile = File(...),
    json_data: str = Form(...)
):
    params = json.loads(json_data)

    base_sheets = params.get("base_sheets")
    base_sheet = params.get("base_sheet")
    if base_sheets is None and base_sheet:
        base_sheets = [base_sheet]
    if base_sheets is None:
        return {"erro": "Informe base_sheets (lista) ou base_sheet (única)."}

    target_year = int(params.get("target_year"))
    date_col = params.get("date_col", "Data")
    value_col = params.get("value_col", "Valor")

    # sazonalidade padrão = 24 períodos (12 meses × 2 quinzenas)
    requested_seasonality = int(params.get("seasonality", 12)) # ===== ALTERAR SAZONALIDADE 

    contents = await file.read()
    dfs_by_sheet = parse_excel_bytes(contents)

    # validar abas
    missing = [s for s in base_sheets if s not in dfs_by_sheet]
    if missing:
        return {
            "erro": f"Abas não encontradas: {missing}. Abas disponíveis: {list(dfs_by_sheet.keys())}"
        }

    # Função auxiliar: converte uma data para o começo da quinzena (1 ou 16)
    def to_quinzena(dt):
        # dt deve ser Timestamp-like
        try:
            if dt.day <= 15:
                return pd.Timestamp(dt.year, dt.month, 1)
            else:
                return pd.Timestamp(dt.year, dt.month, 16)
        except Exception:
            # fallback simples: tenta converter para Timestamp e reexecutar
            dt2 = pd.to_datetime(dt, errors="coerce")
            if pd.isna(dt2):
                return None
            if dt2.day <= 15:
                return pd.Timestamp(dt2.year, dt2.month, 1)
            return pd.Timestamp(dt2.year, dt2.month, 16)

    # Função auxiliar: próxima quinzena após dt
    def next_quinzena(dt):
        if dt.day == 1:
            return pd.Timestamp(dt.year, dt.month, 16)
        else:
            # 16 -> 1 do mês seguinte
            return (dt + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1)

    # Gera lista de datas de quinzenas alinhadas (1 ou 16)
    def gerar_datas_quinzenas(start, periods):
        datas = []
        dt = pd.to_datetime(start)
        # garantir que start seja 1 ou 16; se não for, alinhamos para a próxima quinzena correta
        if dt.day <= 15:
            dt = pd.Timestamp(dt.year, dt.month, 1)
        else:
            dt = pd.Timestamp(dt.year, dt.month, 16)

        for _ in range(periods):
            datas.append(pd.Timestamp(dt.year, dt.month, dt.day))
            dt = next_quinzena(dt)
        return pd.DatetimeIndex(datas)

    # carregar e combinar todas as abas base
    df_list = []
    parsed_years = []
    for s in base_sheets:
        df_s = dfs_by_sheet[s].copy()
        df_s["_sheet"] = s
        df_list.append(df_s)

        y = sheet_name_to_year(s)
        if y is not None:
            parsed_years.append(y)

    combined = pd.concat(df_list, ignore_index=True)
    combined[date_col] = pd.to_datetime(combined[date_col], dayfirst=True, errors="coerce")
    combined = combined.dropna(subset=[date_col])

    if combined.empty:
        return {"erro": "As abas selecionadas não contêm datas válidas."}

    combined = combined.sort_values(date_col).set_index(date_col)

    # -----------------------------
    # AGREGAÇÃO QUINZENAL CORRETA (1–15 / 16–fim)
    # -----------------------------
    # adiciona coluna 'quinzena' com timestamp 1 ou 16 do mês
    combined["quinzena"] = combined.index.map(to_quinzena)
    # remove possíveis None
    combined = combined.dropna(subset=["quinzena"])
    # agrupa por quinzena e soma valores
    series_base_quinzenal = combined.groupby("quinzena")[value_col].sum()
    # garantir índice como DatetimeIndex e ordenado
    series_base_quinzenal.index = pd.to_datetime(series_base_quinzenal.index)
    series_base_quinzenal = series_base_quinzenal.sort_index()

    if len(series_base_quinzenal.dropna()) < 4:
        return {"erro": "Poucos dados para previsão quinzenal (mínimo 4 períodos agregados)."}

    # detectar ano máximo base
    max_base_year = max(parsed_years) if parsed_years else None

    if max_base_year and target_year <= max_base_year:
        return {"erro": "Ano alvo deve ser maior que o maior ano base."}

    # -------------------------------------------
    # Início da previsão: próxima quinzena após a última existente
    # -------------------------------------------
    last_date = series_base_quinzenal.index[-1]
    forecast_start = next_quinzena(last_date)

    # -------------------------------------------
    # Quantidade de quinzenas até o target_year
    # -------------------------------------------
    if max_base_year:
        anos_a_prever = target_year - max_base_year
    else:
        anos_a_prever = target_year - last_date.year

    if anos_a_prever < 1:
        anos_a_prever = 1

    # 24 quinzenas por ano (fixo)
    future_periods = anos_a_prever * 24

    forecast_index = gerar_datas_quinzenas(forecast_start, future_periods)

    # -------------------------------------------
    # AUTO-SAZONALIDADE
    # -------------------------------------------
    usable_len = len(series_base_quinzenal.dropna())
    #seasonality = auto_adjust_seasonality(usable_len, requested_seasonality)
    seasonality = 12 # ===== ALTERAR SAZONALIDADE 
    logger.info(
        f"[quinzenal] usable_len={usable_len}, requested={requested_seasonality}, usando seasonality={seasonality}"
    )

    # -------------------------------------------
    # TREINAMENTO DO MODELO
    # -------------------------------------------
    try:
        if seasonality and seasonality >= 2:
            model = ExponentialSmoothing(
                series_base_quinzenal,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonality
            ).fit()
        else:
            model = ExponentialSmoothing(
                series_base_quinzenal,
                trend="add"
            ).fit()

    except Exception as e:
        logger.warning(f"[quinzenal] Erro com sazonalidade: {e}. Tentando sem sazonalidade.")
        model = ExponentialSmoothing(series_base_quinzenal, trend="add").fit()

    # -------------------------------------------
    # GERAR FORECAST
    # -------------------------------------------
    try:
        raw_forecast = model.forecast(future_periods)
    except Exception as e:
        logger.error(f"[quinzenal] erro ao gerar forecast: {e}")
        raw_forecast = [0.0] * future_periods

    # garantir série com índice alinhado às datas geradas
    forecast_values = ensure_forecast_values(raw_forecast, forecast_index)

    # -------------------------------------------
    # HISTÓRICO BASE
    # -------------------------------------------
    historico_base = safe_series_to_list(series_base_quinzenal)
    historico_base = sanitize_list(historico_base)

    # -------------------------------------------
    # REAIS POR ANO (se existirem abas)
    # -------------------------------------------
    reais_por_ano = {}
    sheet_year_map = {}

    for s in dfs_by_sheet:
        y = sheet_name_to_year(s)
        if y is not None:
            sheet_year_map[y] = s

    if max_base_year:
        for y in range(max_base_year + 1, target_year + 1):
            if y in sheet_year_map:
                dfy = dfs_by_sheet[sheet_year_map[y]].copy()
                dfy[date_col] = pd.to_datetime(dfy[date_col], dayfirst=True, errors="coerce")
                dfy = dfy.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

                # agrega por quinzena usando a mesma função to_quinzena
                dfy["quinzena"] = dfy.index.map(to_quinzena)
                dfy = dfy.dropna(subset=["quinzena"])
                qy = dfy.groupby("quinzena")[value_col].sum()
                qy.index = pd.to_datetime(qy.index)
                qy = qy[qy.index.year == y]

                if len(qy) > 0:
                    reais_por_ano[y] = sanitize_list(safe_series_to_list(qy))

    previsao_list = safe_series_to_list(forecast_values)
    previsao_list = sanitize_list(previsao_list)

    # -------------------------------------------
    # RETORNO FINAL
    # -------------------------------------------
    return {
        "base_sheets": base_sheets,
        "base_years_guess": parsed_years,
        "target_year": target_year,

        "historico_base": historico_base,
        "reais_por_ano": reais_por_ano,
        "previsao": previsao_list,

        "forecast_start": str(forecast_index[0].date()),
        "forecast_end": str(forecast_index[-1].date()),

        "future_periods": future_periods,
        "seasonality_used": seasonality,
    }

# ---------------------------
# ENDPOINT: forecast_periodo_trimestral
# ---------------------------
@app.post("/forecast_periodo_trimestral")
async def forecast_periodo_trimestral(
    file: UploadFile = File(...),
    json_data: str = Form(...)
):
    params = json.loads(json_data)

    base_sheets = params.get("base_sheets")
    base_sheet = params.get("base_sheet")
    if base_sheets is None and base_sheet:
        base_sheets = [base_sheet]
    if base_sheets is None:
        return {"erro": "Informe base_sheets (lista) ou base_sheet (única)."}

    target_year = int(params.get("target_year"))
    date_col = params.get("date_col", "Data")
    value_col = params.get("value_col", "Valor")

    requested_seasonality = int(params.get("seasonality", 4))

    contents = await file.read()
    dfs_by_sheet = parse_excel_bytes(contents)

    # validar abas
    missing = [s for s in base_sheets if s not in dfs_by_sheet]
    if missing:
        return {"erro": f"Abas não encontradas: {missing}. Abas disponíveis: {list(dfs_by_sheet.keys())}"}

    # carregar dataframes
    df_list = []
    parsed_years = []
    for s in base_sheets:
        df_s = dfs_by_sheet[s].copy()
        df_list.append(df_s)

        y = sheet_name_to_year(s)
        if y is not None:
            parsed_years.append(y)

    combined = pd.concat(df_list, ignore_index=True)
    combined[date_col] = pd.to_datetime(combined[date_col], dayfirst=True, errors="coerce")
    combined = combined.dropna(subset=[date_col])
    combined = combined.sort_values(date_col).set_index(date_col)

    if combined.empty:
        return {"erro": "As abas selecionadas não contêm datas válidas."}

    # -------------------------------------
    # AGREGAÇÃO TRIMESTRAL (1º dia do trimestre)
    # -------------------------------------
    series_base_tri = combined[value_col].resample("Q").sum()
    series_base_tri.index = series_base_tri.index.to_period("Q").to_timestamp("Q")
    series_base_tri = series_base_tri.sort_index()

    if len(series_base_tri.dropna()) < 4:
        return {"erro": "Poucos dados para previsão trimestral (mínimo 4 trimestres)."}

    max_base_year = max(parsed_years)

    if target_year <= max_base_year:
        return {"erro": "Ano alvo deve ser maior que o maior ano base selecionado."}

    # -------------------------------------
    # PREVISÃO — datas trimestrais (CORRIGIDO)
    # -------------------------------------

    last_date = series_base_tri.index[-1]

    # começa exatamente no próximo trimestre correto
    forecast_start = last_date + pd.offsets.QuarterBegin()

    # fim do último trimestre do ano alvo
    forecast_end = pd.Timestamp(target_year, 12, 31)

    # gera trimestres perfeitamente alinhados
    forecast_index = pd.date_range(
        start=forecast_start,
        end=forecast_end,
        freq="Q"
    )

    future_periods = len(forecast_index)

    if future_periods <= 0:
        return {"erro": "Não há trimestres para prever. Verifique ano alvo e dados base."}

    # -------------------------------------
    # AUTO-SAZONALIDADE
    # -------------------------------------
    usable_len = len(series_base_tri.dropna())
    seasonality = auto_adjust_seasonality(usable_len, requested_seasonality)

    # -------------------------------------
    # TREINAR MODELO
    # -------------------------------------
    try:
        if seasonality and seasonality >= 2:
            model = ExponentialSmoothing(
                series_base_tri,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonality
            ).fit()
        else:
            model = ExponentialSmoothing(series_base_tri, trend="add").fit()
    except Exception:
        model = ExponentialSmoothing(series_base_tri, trend="add").fit()

    # gerar forecast
    try:
        raw_forecast = model.forecast(future_periods)
    except Exception:
        raw_forecast = [0.0] * future_periods

    forecast_values = ensure_forecast_values(raw_forecast, forecast_index)

    # -------------------------------------
    # HISTÓRICO BASE
    # -------------------------------------
    historico_base = sanitize_list(safe_series_to_list(series_base_tri))

    # -------------------------------------
    # REAIS POR ANO
    # -------------------------------------
    reais_por_ano = {}
    sheet_year_map = {}

    for s in dfs_by_sheet:
        y = sheet_name_to_year(s)
        if y is not None:
            sheet_year_map[y] = s

    for y in range(max_base_year + 1, target_year + 1):
        if y in sheet_year_map:
            dfy = dfs_by_sheet[sheet_year_map[y]].copy()
            dfy[date_col] = pd.to_datetime(dfy[date_col], dayfirst=True, errors="coerce")
            dfy = dfy.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

            qy = dfy[value_col].resample("Q").sum()
            qy.index = qy.index.to_period("Q").to_timestamp("Q")

            reais_por_ano[y] = sanitize_list(safe_series_to_list(qy))

    previsao_list = sanitize_list(safe_series_to_list(forecast_values))

    # -------------------------------------
    # RETORNO FINAL
    # -------------------------------------
    return {
        "base_sheets": base_sheets,
        "base_years_guess": parsed_years,
        "target_year": target_year,

        "historico_base": historico_base,
        "reais_por_ano": reais_por_ano,
        "previsao": previsao_list,

        "forecast_start": str(forecast_index[0].date()),
        "forecast_end": str(forecast_index[-1].date()),

        "future_periods": future_periods,
        "seasonality_used": seasonality,
    }
