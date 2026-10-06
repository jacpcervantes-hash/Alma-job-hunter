import os
import json
import time
import pandas as pd
from jobspy import scrape_jobs
from google import genai
from google.genai import types

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, when, lit, desc, udf
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

CANDIDATE_PROFILE = """
- Formación: Ingeniería Industrial con Especialidad en Gestión de Proyectos (Project Management).
- Experiencia: Marketing Intern en Nestlé, Sales Intern en Dell, Sales Analyst en Grupo Alen. Busca posiciones marketing o trade marketing
- Sueldo actual: $25,000 MXN brutos. Busco entre $25,000 y $30,000 MXN brutos en CDMX/Área Metropolitana en nivel Specialist, Analyst o Coordinator (No Gerente/Manager, No Becario/Intern).
"""

gemini_schema = StructType([
    StructField("match_pct", IntegerType(), True),
    StructField("estimated_salary_mxn", StringType(), True),
    StructField("ai_verdict", StringType(), True)
])

def evaluate_job_with_gemini(title: str, company: str, location: str, description: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"match_pct": 0, "estimated_salary_mxn": "Sin API Key", "ai_verdict": "Falta configurar GEMINI_API_KEY"}

    client = genai.Client(api_key=api_key)
    prompt = f"""
    Evalúa esta vacante en México contra el perfil del candidato.
    PERFIL: {CANDIDATE_PROFILE}
    VACANTE: Puesto: {title} | Empresa: {company} | Ubicación: {location} | Descripción: {description[:2500]}

    Responde SOLO un JSON válido con estas 3 llaves:
    {{"match_pct": 85, "estimated_salary_mxn": "$32,000 - $38,000 MXN", "ai_verdict": "Explicación breve en español"}}
    """

    # Pausa base de 12 segundos para cuidar el límite gratuito por minuto
    time.sleep(12)

    # Intento 1 con 3.8-flash; si está saturado (503/429), pasa a 3.5-flash-lite de respaldo
    modelos = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash-lite"]

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
                time.sleep(15)
            else:
                return {
                    "match_pct": 50,
                    "estimated_salary_mxn": "Revisar en portal",
                    "ai_verdict": f"Aviso API: {error_msg[:80]}"
                }

    return {
        "match_pct": 50,
        "estimated_salary_mxn": "Revisar en portal",
        "ai_verdict": "Servidor ocupado tras 3 intentos"
    }

def run_spark_pipeline():
    spark = SparkSession.builder \
        .appName("Automated_RD_PMO_Gemini_Hunter") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    gemini_eval_udf = udf(evaluate_job_with_gemini, gemini_schema)

    # Búsquedas directas y limpias (últimos 7 días = 168 horas)
    queries = {
        "PMO_Proyectos": "PMO Project Analyst Coordinator",
        "RD_Desarrollo": "R&D Research Development Formulacion"
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

    sdf = spark.createDataFrame(raw_pdf)

    # Excluir niveles gerenciales altos y becarios
    exclude_titles_regex = r"(director|head|vp|becario|intern|practicante|trainee)"
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

    # Evaluar las 15 mejores con Gemini y ordenar de mayor a menor match
    scored_sdf = (
        pre_filtered_sdf.coalesce(1)
        .withColumn(
            "gemini_eval",
            gemini_eval_udf(col("title"), col("company"), col("location"), col("description"))
        )
        .select(
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

    final_pdf = scored_sdf.toPandas()
    output_file = "vacantes_evaluadas_gemini_spark.csv"
    final_pdf.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Archivo guardado con {len(final_pdf)} vacantes.")

    spark.stop()

if __name__ == "__main__":
    run_spark_pipeline()
