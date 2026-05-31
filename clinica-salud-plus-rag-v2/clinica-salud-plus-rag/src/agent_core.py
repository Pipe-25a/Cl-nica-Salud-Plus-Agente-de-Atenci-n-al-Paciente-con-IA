"""
src/agent_core.py
Agente principal de Clinica Salud Plus.
Implementa el esquema ReAct usando LangGraph (compatible con LangChain >= 1.0)
con memoria de corto plazo y largo plazo (ChromaDB semantico).
"""
import os
import sys
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from dotenv import load_dotenv
load_dotenv()
 
token    = os.environ.get("GITHUB_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
base_url = os.environ.get("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")
 
os.environ["OPENAI_API_KEY"]  = token
os.environ["OPENAI_API_BASE"] = base_url
 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
 
from tools.rag_tool import buscar_informacion_clinica
from tools.classification_tool import clasificar_consulta
from memory.long_term_memory import LongTermMemory
 
 
SYSTEM_PROMPT = """Eres el asistente virtual de la Clinica Salud Plus.
Tu rol es ayudar a pacientes respondiendo sus consultas de forma precisa, empatica y confiable.
 
PROTOCOLO OBLIGATORIO — sigue estas etapas en orden:
1. Clasifica la consulta usando la herramienta clasificar_consulta.
2. Si el resultado es URGENCIA: deriva de inmediato al 131 o urgencias. No uses otras herramientas.
3. Si el resultado es AGENDAMIENTO o DERIVACION: entrega las instrucciones correspondientes.
4. Si el resultado es INFORMATIVA: usa buscar_informacion_clinica para recuperar informacion.
5. Formula una respuesta clara y humana basada SOLO en la informacion recuperada.
 
REGLAS CRITICAS:
- Nunca inventes informacion medica, horarios, ni datos de medicos.
- Si la informacion no esta disponible, indica que no tienes esa informacion y deriva a recepcion (600 123 4567).
- Ante cualquier sintoma de urgencia, prioriza la derivacion inmediata sobre todo lo demas.
- Responde siempre en espanol, con tono amable y profesional."""
 
 
def crear_agente(session_id: str = "default"):
    llm = ChatOpenAI(
        api_key=token,
        base_url=base_url,
        model="gpt-4o-mini",
        temperature=0,
    )
 
    tools = [clasificar_consulta, buscar_informacion_clinica]
 
    # Memoria de corto plazo (IE3): MemorySaver guarda historial por thread_id
    memoria_corto_plazo = MemorySaver()
 
    # Memoria de largo plazo (IE4): ChromaDB semantico
    memoria_largo_plazo = LongTermMemory(persist_dir="./memory_store")
 
    # Agente ReAct con LangGraph
    agente = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memoria_corto_plazo,
        prompt=SYSTEM_PROMPT,
    )
 
    return agente, memoria_largo_plazo
 
 
def consultar(ejecutor, memoria_lp: LongTermMemory, pregunta: str, session_id: str = "default") -> str:
    # Recuperar contexto historico de sesiones anteriores
    historial = memoria_lp.recuperar_contexto_previo(pregunta)
    
    input_text = pregunta
    if historial:
        input_text = f"{pregunta}\n\n[Contexto previo: {historial}]"
 
    config = {"configurable": {"thread_id": session_id}}
 
    resultado = ejecutor.invoke(
        {"messages": [HumanMessage(content=input_text)]},
        config=config,
    )
 
    respuesta = resultado["messages"][-1].content
    memoria_lp.guardar_interaccion(pregunta, respuesta, session_id=session_id)
    return respuesta
 
 
if __name__ == "__main__":
    print("=" * 60)
    print("  CLINICA SALUD PLUS - Asistente Virtual IA")
    print("  Agente LangGraph ReAct con Memoria Dual")
    print("=" * 60)
 
    ejecutor, memoria_lp = crear_agente(session_id="test_session")
 
    preguntas = [
        "Cuales son los horarios de atencion de la clinica?",
        "Como me preparo para un examen de sangre?",
        "Que especialidades medicas tienen disponibles?",
        "Tengo un dolor fuerte en el pecho, que hago?",
        "Puedo agendar una hora con el Dr. Carlos Rojas?",
        "Aceptan Fonasa?",
        "Cuanto demoran los resultados de los examenes?",
    ]
 
    for pregunta in preguntas:
        print(f"\n{'-'*55}")
        print(f"PACIENTE: {pregunta}")
        print("-" * 55)
        respuesta = consultar(ejecutor, memoria_lp, pregunta, session_id="test_session")
        print(f"AGENTE: {respuesta}")
 
    print(f"\n{'=' * 60}")
    print("  Pruebas completadas.")
    print("=" * 60)