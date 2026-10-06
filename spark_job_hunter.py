import os
import json
import time
import requests
import pandas as pd
from jobspy import scrape_jobs
from google import genai
from google.genai import types

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, when, lit, desc, udf
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# 👇 PEGA AQUÍ EL LINK /exec DEL GOOGLE SHEETS DE ALMA:
SHEETS_WEBHOOK_URL = "PEGA_AQUI_EL_LINK_DE_APPS_SCRIPT_DE_ALMA"

CANDIDATE_PROFILE = """
- Formación y Perfil: Profesional en áreas económico-administrativas / comerciales.
- Experiencia previa: Marketing Intern en Nestlé, Sales Intern en Dell, Sales Analyst en Grupo Alen.
- Objetivo profesional: Posiciones de Marketing, Trade Marketing o Brand Specialist.
- Expectativa salarial: Sueldo actual $25,000 MXN brutos. Busca entre $25,000 y $30,000 MXN brutos en CDMX / Área Metropolitana.
- Nivel buscado: Specialist, Analyst o Coordinator (No Gerente/Manager, No Becario/Intern).
"""

FEEDBACK_HISTORY = ""

gemini_schema = StructType([
    StructField("match_pct", IntegerType(), True),
    StructField("estimated_salary_mxn", StringType(), True),
    StructField("ai_verdict", StringType(), True)
])

def cargar_historial_y_retroalimentacion(output_csv: str):
    """Lee las notas del Google Sheets de Alma para sincronizar y re-alimentar a Gemini."""
    df_hist = pd.DataFrame()
    if SHEETS_WEBHOOK_URL.startswith("https://script.google.com"):
        try:
            resp = requests.get(SHEETS_WEBHOOK_URL, timeout=20)
            if resp.status_code == 200:
                datos = resp.json()
                if datos:
                    df_hist = pd.DataFrame(datos)
                    print(f"Sincronizadas {len(df_hist)} vacantes desde el Google Sheets de Alma.")
        except Exception as e:
            print(f"Aviso al leer Google Sheets: {e}")

    if df_hist.empty and os.path.exists(output_csv):
        try:
            df_hist = pd.read_csv(output_csv).fillna("")
        except Exception:
            df_hist = pd.DataFrame()

    feedback_lines = []
    if not df_hist.empty and "me_interesa" in df_hist.columns:
        for _, r in df_hist.iterrows():
            interes = str(r.get("me_interesa", "")).strip()
            razon = str(r.get("razon_o_notas", "")).strip()
            estatus = str(r.get("estatus_aplicacion", "")).strip()
            if interes.lower() in ["sí", "si", "no"] or razon != "" or estatus.lower() == "ya apliqué":
                feedback_lines.append(
                    f"- Puesto: {r.get('title', '')} en {r.get('company', '')} -> ¿Interesó?: {interes} | Estatus: {estatus} | Notas: {razon}"
                )

    feedback_str = ""
    if feedback_lines:
        ultimas_notas = "\n".join(feedback_lines[:15])
        feedback_str = f"\nHISTORIAL DE PREFERENCIAS Y NOTAS DE ALMA:\n{ultimas_notas}\n"

    return df_hist, feedback_str

def evaluate_job_with_gemini(title: str, company: str, location: str, description: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"match_pct": 0, "estimated_salary_mxn": "Sin API Key", "ai_verdict": "Falta configurar GEMINI_API_KEY"}

    client = genai.Client(api_key=api_key)
    prompt = f"""
    Evalúa esta vacante en México contra el perfil comercial/marketing y las preferencias de Alma.
    PERFIL: {CANDIDATE_PROFILE}
    {FEEDBACK_HISTORY}
    VACANTE: Puesto: {title} | Empresa: {company} | Ubicación: {location} | Descripción: {description[:2500]}

    Responde SOLO un JSON válido con estas 3 llaves:
    {{"match_pct": 85, "estimated_salary_mxn": "$25,000 - $30,000 MXN", "ai_verdict": "Explicación breve en español"}}
    """

    # Pausa de 6 segundos optimizada con .cache() para terminar en ~2-3 minutos
    time.sleep(6)
    modelos = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]

    for modelo_actual in modelos:
        try:
            response = client.models.generate_content(
                model=modelo_actual,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            data = json.loads(response.text)
            return {
                "match_pct": int(data.get("match_pct", 50)),
                "estimated_salary_mxn": str(data.get("estimated_salary_mxn", "Por confirmar")),
                "ai_verdict": str(data.get("ai_verdict", "Analizado"))
            }
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "503" in error_msg or "404" in error_msg:
                time.sleep(12)
            else:
                return {
                    "match_pct": 50,
                    "estimated_salary_mxn": "Revisar en portal",
                    "ai_verdict": f"Aviso API: {error_msg[:80]}"
                }

    return {
        "match_pct": 50,
        "estimated_salary_mxn": "Revisar en portal",
        "ai_verdict": "Servidor ocupado tras reintentos"
    }

def run_spark_pipeline():
    global FEEDBACK_HISTORY
    output_file = "vacantes_evaluadas_gemini_spark.csv"

    old_pdf, FEEDBACK_HISTORY = cargar_historial_y_retroalimentacion(output_file)
    urls_existentes = set(old_pdf["job_url"].astype(str).tolist()) if (not old_pdf.empty and "job_url" in old_pdf.columns) else set()

    spark = SparkSession.builder \
        .appName("Automated_Marketing_Gemini_Hunter") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    gemini_eval_udf = udf(evaluate_job_with_gemini, gemini_schema)

        # 🎯 Búsquedas 100% enfocadas en Marketing, Trade Marketing y Brand Analyst en CDMX
    queries = {
        "Marketing_Brand": "Marketing Brand Specialist Analyst",
        "Trade_Marketing": "Trade Marketing Analyst Specialist",
        "Comercial_Commercial": "Commercial Analyst Marketing Mexico"
    }


    all_jobs = []
    for category, q in queries.items():
        try:
            df = scrape_jobs(
                site_name=["linkedin", "indeed"],
                search_term=q,
                location="Mexico City, Mexico",
                results_wanted=40,
                hours_old=168,
                country_indeed="Mexico"
            )
            if not df.empty:
                df["vertical"] = category
                all_jobs.append(df)
        except Exception as e:
            print(f"Error extrayendo {category}: {e}")

    if not all_jobs:
        print("No se extrajeron vacantes.")
        spark.stop()
        return

    raw_pdf = pd.concat(all_jobs, ignore_index=True)
    cols_to_keep = ["title", "company", "location", "date_posted", "job_url", "description", "vertical"]
    for c in cols_to_keep:
        if c not in raw_pdf.columns:
            raw_pdf[c] = ""
    raw_pdf = raw_pdf[cols_to_keep].fillna("").astype(str)

    # Evitar reevaluar vacantes que ya están en su Google Sheets
    raw_pdf = raw_pdf[~raw_pdf["job_url"].isin(urls_existentes)]

    orden_columnas = [
        "me_interesa", "estatus_aplicacion", "razon_o_notas",
        "match_pct", "vertical", "company_tier", "title", "company",
        "estimated_salary_mxn", "ai_verdict", "location", "date_posted", "job_url"
    ]

    if raw_pdf.empty:
        print("No hay vacantes nuevas distintas a las del Google Sheets de Alma.")
        if not old_pdf.empty:
            for c in orden_columnas:
                if c not in old_pdf.columns:
                    old_pdf[c] = ""
            old_pdf = old_pdf[orden_columnas]
            old_pdf.to_csv(output_file, index=False, encoding="utf-8-sig")
        spark.stop()
        return

    sdf = spark.createDataFrame(raw_pdf)

    exclude_titles_regex = r"(director|head|vp|becario|intern|practicante|trainee|gerente de marca senior)"
    tier1_companies_regex = (
        r"(pepsico|procter|p&g|unilever|loreal|l'oréal|henkel|mondelez|nestle|nestlé|"
        r"kimberly|beiersdorf|coty|natura|avon|reckitt|sc johnson|clorox|"
        r"bayer|pfizer|sanofi|roche|novartis|astrazeneca|gsk|haleon|johnson|"
        r"ey|ernst|deloitte|kpmg|pwc|accenture|santander|bbva|scotiabank|hsbc|amex|mercadolibre|amazon|3m)"
    )

    pre_filtered_sdf = (
        sdf.dropDuplicates(["job_url"])
        .dropDuplicates(["title", "company"])
        .filter(~lower(col("title")).rlike(exclude_titles_regex))
        .withColumn(
            "company_tier",
            when(lower(col("company")).rlike(tier1_companies_regex), lit("Tier 1 - Transnacional"))
            .otherwise(lit("General"))
        )
        .withColumn(
            "priority_score",
            when(col("company_tier").startswith("Tier 1"), lit(100)).otherwise(lit(75))
        )
        .orderBy(desc("priority_score"), desc("date_posted"))
        .limit(15)
    )

    # .cache() para que Gemini evalúe cada vacante una sola vez de forma eficiente
    evaluated_sdf = (
        pre_filtered_sdf.coalesce(1)
        .withColumn(
            "gemini_eval",
            gemini_eval_udf(col("title"), col("company"), col("location"), col("description"))
        )
        .cache()
    )

    scored_sdf = (
        evaluated_sdf.select(
            col("gemini_eval.match_pct").alias("match_pct"),
            "vertical",
            "company_tier",
            "title",
            "company",
            col("gemini_eval.estimated_salary_mxn").alias("estimated_salary_mxn"),
            col("gemini_eval.ai_verdict").alias("ai_verdict"),
            "location",
            "date_posted",
            "job_url"
        )
        .orderBy(desc("match_pct"), desc("date_posted"))
    )

    new_pdf = scored_sdf.toPandas()
    new_pdf["me_interesa"] = "Por revisar"
    new_pdf["estatus_aplicacion"] = "Pendiente"
    new_pdf["razon_o_notas"] = ""
    new_pdf = new_pdf[orden_columnas].fillna("")

    # Enviar vacantes nuevas al Google Sheets de Alma
    if SHEETS_WEBHOOK_URL.startswith("https://script.google.com") and not new_pdf.empty:
        try:
            payload = {
                "headers": orden_columnas,
                "rows": new_pdf.values.tolist()
            }
            requests.post(SHEETS_WEBHOOK_URL, json=payload, timeout=20)
            print(f"Se agregaron {len(new_pdf)} vacantes nuevas al Google Sheets de Alma.")
        except Exception as e:
            print(f"Aviso al enviar a Google Sheets: {e}")

    if not old_pdf.empty:
        for c in orden_columnas:
            if c not in old_pdf.columns:
                old_pdf[c] = ""
        old_pdf = old_pdf[orden_columnas]
        final_pdf = pd.concat([old_pdf, new_pdf], ignore_index=True)
    else:
        final_pdf = new_pdf

    final_pdf.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Respaldo actualizado con {len(final_pdf)} vacantes totales.")

    spark.stop()

if __name__ == "__main__":
    run_spark_pipeline()
