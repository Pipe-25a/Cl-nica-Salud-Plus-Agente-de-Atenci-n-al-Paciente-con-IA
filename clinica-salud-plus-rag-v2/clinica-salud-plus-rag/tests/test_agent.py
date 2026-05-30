"""
tests/test_agent.py
Suite de pruebas del agente de Clínica Salud Plus.
Cubre los 4 tipos de consulta: informativas, urgencias, agendamiento y multietapa.
Evalúa fidelidad, relevancia y tasa de derivación correcta.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.agent_core import crear_agente, consultar


# ── Casos de prueba ───────────────────────────────────────────────────────────
CASOS_PRUEBA = [
    # ── Consultas informativas ──────────────────────────────────────────────
    {
        "id": "T01",
        "tipo": "INFORMATIVA",
        "pregunta": "¿Cuáles son los horarios de atención de la clínica?",
        "keywords_esperadas": ["lunes", "viernes", "8:00", "20:00"],
    },
    {
        "id": "T02",
        "tipo": "INFORMATIVA",
        "pregunta": "¿Qué especialidades médicas tienen disponibles?",
        "keywords_esperadas": ["cardiologia", "pediatria", "ginecologia"],
    },
    {
        "id": "T03",
        "tipo": "INFORMATIVA",
        "pregunta": "¿Cómo me preparo para un examen de sangre?",
        "keywords_esperadas": ["ayuno", "8 horas", "medicamentos"],
    },
    {
        "id": "T04",
        "tipo": "INFORMATIVA",
        "pregunta": "¿Cuánto demoran los resultados de los exámenes?",
        "keywords_esperadas": ["24", "48", "horas"],
    },
    {
        "id": "T05",
        "tipo": "INFORMATIVA",
        "pregunta": "¿Aceptan Fonasa?",
        "keywords_esperadas": ["fonasa"],
    },
    {
        "id": "T06",
        "tipo": "INFORMATIVA",
        "pregunta": "¿Cuáles son los horarios del laboratorio clínico?",
        "keywords_esperadas": ["7:30", "laboratorio"],
    },
    {
        "id": "T07",
        "tipo": "INFORMATIVA",
        "pregunta": "¿Cómo me preparo para una ecografía abdominal?",
        "keywords_esperadas": ["ayuno", "agua", "6 horas"],
    },
    {
        "id": "T08",
        "tipo": "INFORMATIVA",
        "pregunta": "¿Hay estacionamiento disponible?",
        "keywords_esperadas": ["estacionamiento", "gratuito"],
    },
    {
        "id": "T09",
        "tipo": "INFORMATIVA",
        "pregunta": "¿Qué médicos tienen disponibles y cuáles son sus horarios?",
        "keywords_esperadas": ["martinez", "lopez", "rojas"],
    },
    {
        "id": "T10",
        "tipo": "INFORMATIVA",
        "pregunta": "¿Cuál es el número de teléfono para agendar horas?",
        "keywords_esperadas": ["600 123 4567"],
    },
    # ── Urgencias ───────────────────────────────────────────────────────────
    {
        "id": "T11",
        "tipo": "URGENCIA",
        "pregunta": "Tengo un dolor fuerte en el pecho y no puedo respirar bien.",
        "keywords_esperadas": ["131", "urgencias"],
    },
    {
        "id": "T12",
        "tipo": "URGENCIA",
        "pregunta": "Mi hijo tuvo una convulsión, ¿qué hago?",
        "keywords_esperadas": ["131", "urgencias"],
    },
    {
        "id": "T13",
        "tipo": "URGENCIA",
        "pregunta": "Mi familiar se desmayó y no reacciona.",
        "keywords_esperadas": ["131", "urgencias"],
    },
    # ── Agendamiento ────────────────────────────────────────────────────────
    {
        "id": "T14",
        "tipo": "AGENDAMIENTO",
        "pregunta": "Quiero agendar una hora con la Dra. Ana López.",
        "keywords_esperadas": ["600 123 4567"],
    },
    {
        "id": "T15",
        "tipo": "AGENDAMIENTO",
        "pregunta": "¿Cómo puedo reservar una consulta de cardiología?",
        "keywords_esperadas": ["600 123 4567"],
    },
    # ── Consultas ambiguas / sin información ────────────────────────────────
    {
        "id": "T16",
        "tipo": "INFORMATIVA",
        "pregunta": "¿Tienen servicio de urgencias?",
        "keywords_esperadas": ["urgencias", "24 horas"],
    },
    {
        "id": "T17",
        "tipo": "INFORMATIVA",
        "pregunta": "¿Qué isapres aceptan?",
        "keywords_esperadas": ["banmedica", "colmena"],
    },
    # ── Consultas multietapa ─────────────────────────────────────────────────
    {
        "id": "T18",
        "tipo": "MULTIETAPA_1",
        "pregunta": "¿Cómo me preparo para una mamografía?",
        "keywords_esperadas": ["desodorante", "cremas"],
    },
    {
        "id": "T19",
        "tipo": "MULTIETAPA_2",  # Referencia implícita a T18
        "pregunta": "¿Y cuánto tiempo antes del examen debo llegar?",
        "keywords_esperadas": [],  # Respuesta abierta
    },
    {
        "id": "T20",
        "tipo": "INFORMATIVA",
        "pregunta": "¿Cuáles son los datos para hacer una transferencia?",
        "keywords_esperadas": ["banco estado", "76.123.456"],
    },
]


def evaluar_respuesta(respuesta: str, keywords: list[str]) -> dict:
    """Evalúa si la respuesta contiene las palabras clave esperadas."""
    respuesta_lower = respuesta.lower()
    encontradas = [kw for kw in keywords if kw.lower() in respuesta_lower]
    total        = len(keywords) if keywords else 1
    score        = len(encontradas) / total if total else 1.0
    return {
        "score":       score,
        "encontradas": encontradas,
        "faltantes":   [kw for kw in keywords if kw.lower() not in respuesta_lower],
    }


def ejecutar_pruebas():
    """Ejecuta la suite completa de pruebas y muestra resultados."""
    print("=" * 65)
    print("  SUITE DE PRUEBAS — Agente Clínica Salud Plus")
    print("=" * 65)

    ejecutor, memoria_lp = crear_agente(session_id="test_suite")

    resultados   = []
    score_total  = 0.0
    urgencias_ok = 0
    urgencias_total = 0

    for caso in CASOS_PRUEBA:
        print(f"\n[{caso['id']}] ({caso['tipo']}) {caso['pregunta']}")
        print("─" * 55)

        inicio   = time.time()
        respuesta = consultar(ejecutor, memoria_lp, caso["pregunta"], session_id="test_suite")
        duracion  = time.time() - inicio

        eval_res = evaluar_respuesta(respuesta, caso["keywords_esperadas"])
        score_total += eval_res["score"]

        if caso["tipo"] == "URGENCIA":
            urgencias_total += 1
            if eval_res["score"] >= 0.5:
                urgencias_ok += 1

        estado = "✅" if eval_res["score"] >= 0.5 else "⚠️"
        print(f"RESPUESTA: {respuesta[:250]}{'...' if len(respuesta) > 250 else ''}")
        print(f"{estado} Score: {eval_res['score']:.0%} | Tiempo: {duracion:.1f}s")
        if eval_res["faltantes"]:
            print(f"   Keywords faltantes: {eval_res['faltantes']}")

        resultados.append({**caso, "respuesta": respuesta, **eval_res, "duracion": duracion})

    # ── Resumen ────────────────────────────────────────────────────────────────
    n             = len(CASOS_PRUEBA)
    fidelidad_avg = score_total / n
    derivacion_rate = (urgencias_ok / urgencias_total * 100) if urgencias_total else 0

    print("\n" + "=" * 65)
    print("  RESUMEN DE EVALUACIÓN")
    print("=" * 65)
    print(f"  Total de pruebas:          {n}")
    print(f"  Fidelidad promedio:        {fidelidad_avg:.0%}")
    print(f"  Derivaciones correctas:    {urgencias_ok}/{urgencias_total} ({derivacion_rate:.0f}%)")
    aprobadas = sum(1 for r in resultados if r["score"] >= 0.5)
    print(f"  Casos aprobados (≥50%):   {aprobadas}/{n}")
    print("=" * 65)

    return resultados


if __name__ == "__main__":
    ejecutar_pruebas()
