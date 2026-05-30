"""
src/agent_core.py
Agente principal de Clínica Salud Plus.
Implementa el esquema ReAct (Reasoning + Acting) con LangChain Agents,
memoria de corto plazo (ConversationBufferWindowMemory) y
memoria de largo plazo (ChromaDB semántico).
"""
import os
import sys

# Añadir raíz del proyecto al path para imports relativos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# ── Configuración de credenciales ────────────────────────────────────────────
token    = os.environ.get("GITHUB_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
base_url = os.environ.get("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")

# Mapear para compatibilidad con LangChain
os.environ["OPENAI_API_KEY"]  = token
os.environ["OPENAI_API_BASE"] = base_url

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage

from tools.rag_tool import buscar_informacion_clinica
from tools.classification_tool import clasificar_consulta
from memory.long_term_memory import LongTermMemory


# ── Prompt del sistema ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres el asistente virtual de la Clínica Salud Plus.
Tu rol es ayudar a pacientes respondiendo sus consultas de forma precisa, empática y confiable.

PROTOCOLO OBLIGATORIO — sigue estas etapas en orden:
1. Clasifica la consulta usando la herramienta `clasificar_consulta`.
2. Si el resultado es URGENCIA: deriva de inmediato al 131 o urgencias. No uses otras herramientas.
3. Si el resultado es AGENDAMIENTO o DERIVACION: entrega las instrucciones correspondientes.
4. Si el resultado es INFORMATIVA: usa `buscar_informacion_clinica` para recuperar información.
5. Formula una respuesta clara y humana basada SOLO en la información recuperada.

REGLAS CRÍTICAS:
- Nunca inventes información médica, horarios, ni datos de médicos.
- Si la información no está disponible, indica que no tienes esa información y deriva a recepción.
- Ante cualquier síntoma de urgencia, prioriza la derivación inmediata sobre todo lo demás.
- Responde siempre en español, con tono amable y profesional.
- Si el paciente tiene antecedentes de sesiones anteriores, úsalos para contextualizar tu respuesta.

CONTEXTO HISTÓRICO (sesiones anteriores del paciente):
{historial_largo_plazo}"""


def crear_agente(session_id: str = "default") -> tuple[AgentExecutor, LongTermMemory]:
    """
    Crea y configura el agente con todas sus herramientas y memorias.

    Args:
        session_id: Identificador de sesión del paciente.

    Returns:
        Tupla (AgentExecutor, LongTermMemory) para uso externo.
    """
    # ── LLM ──────────────────────────────────────────────────────────────────
    llm = ChatOpenAI(
        api_key=token,
        base_url=base_url,
        model="gpt-4o-mini",
        temperature=0,
    )

    # ── Herramientas ──────────────────────────────────────────────────────────
    tools = [clasificar_consulta, buscar_informacion_clinica]

    # ── Memoria de corto plazo (IE3): ventana de 5 turnos ─────────────────────
    memoria_corto_plazo = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
    )

    # ── Memoria de largo plazo (IE4): ChromaDB semántico ─────────────────────
    memoria_largo_plazo = LongTermMemory(persist_dir="./memory_store")

    # ── Prompt ────────────────────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # ── Construir agente ReAct con OpenAI Tools ────────────────────────────────
    agente = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agente,
        tools=tools,
        memory=memoria_corto_plazo,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )

    return executor, memoria_largo_plazo


def consultar(
    ejecutor: AgentExecutor,
    memoria_lp: LongTermMemory,
    pregunta: str,
    session_id: str = "default",
) -> str:
    """
    Procesa una consulta del paciente a través del agente.

    Flujo (IE5 — planificación secuencial por prioridades):
    1. Recuperar contexto de sesiones anteriores (memoria LP)
    2. Invocar el agente con el contexto inyectado
    3. Guardar la interacción en memoria LP

    Args:
        ejecutor: AgentExecutor configurado.
        memoria_lp: Instancia de LongTermMemory.
        pregunta: Consulta del paciente.
        session_id: ID de sesión para memoria persistente.

    Returns:
        Respuesta del agente como string.
    """
    # Paso 1: recuperar contexto histórico relevante
    historial = memoria_lp.recuperar_contexto_previo(pregunta)

    # Paso 2: invocar el agente
    resultado = ejecutor.invoke({
        "input": pregunta,
        "historial_largo_plazo": historial if historial else "No hay interacciones previas registradas.",
    })

    respuesta = resultado.get("output", "No se pudo procesar la consulta.")

    # Paso 3: guardar interacción en memoria LP
    memoria_lp.guardar_interaccion(pregunta, respuesta, session_id=session_id)

    return respuesta


# ── Ejecución directa ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CLÍNICA SALUD PLUS — Asistente Virtual IA")
    print("  Agente LangChain ReAct con Memoria Dual")
    print("=" * 60)

    ejecutor, memoria_lp = crear_agente(session_id="test_session")

    # Consultas de prueba que cubren los distintos tipos (IE6)
    preguntas = [
        "¿Cuáles son los horarios de atención de la clínica?",
        "¿Cómo me preparo para un examen de sangre?",
        "¿Qué especialidades médicas tienen disponibles?",
        "Tengo un dolor fuerte en el pecho, ¿qué hago?",
        "¿Puedo agendar una hora con el Dr. Carlos Rojas?",
        "¿Aceptan Fonasa?",
        "¿Cuánto demoran los resultados de los exámenes?",
    ]

    for pregunta in preguntas:
        print(f"\n{'─'*55}")
        print(f"PACIENTE: {pregunta}")
        print("─" * 55)
        respuesta = consultar(ejecutor, memoria_lp, pregunta, session_id="test_session")
        print(f"AGENTE: {respuesta}")

    print(f"\n{'=' * 60}")
    print("  Pruebas completadas.")
    print("=" * 60)
