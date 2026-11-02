# Proyecto 1: Rover Autónomo

Implementación en Python del proyecto de Inteligencia Artificial para un rover autónomo. El proyecto incluye modelado matemático, algoritmos de búsqueda, planificación clásica y metaheurísticas.

## Estructura del Proyecto

```
rover_autonomo/
├── src/
│   ├── __init__.py           # Inicialización del paquete
│   ├── environment.py        # Entorno y generador de instancias
│   ├── search.py             # Módulo A: Búsqueda en espacios de estados
│   ├── planner.py            # Módulo Simbólico: Planificación clásica
│   ├── metaheuristics.py     # Módulo D: Metaheurísticas
│   └── rules.py              # Módulo E: Sistema basado en reglas (Bono)
├── experiments.py            # Script experimental principal
├── requirements.txt          # Dependencias
└── README.md                 # Este archivo
```

## Descripción de Módulos

### 1. Entorno y Generador de Instancias (`environment.py`)

- **Representación**: Rejilla 2D discreta con costos variables por celda
  - Plano/despejado: costo 1.0
  - Arena/dunas: costo 2.0
  - Rocoso/piedras: costo 4.0
  - Obstáculo/intransitable: Infinito

- **Elementos del entorno**:
  - 1 Base con estación de recarga
  - 1-2 Antenas con cobertura de comunicación
  - K Puntos de Interés Científico (POIs) con valores científicos
  - Rover con batería, memoria y tiempo límite

- **Función principal**: `generate_instance(width, height, num_pois, rock_ratio, seed)`

### 2. Módulo A: Búsqueda en Espacios de Estados (`search.py`)

Algoritmos implementados:
- **Dijkstra**: Búsqueda de Costo Uniforme (f(n) = g(n))
- **A* (Euclídea)**: Heurística de distancia Euclídea escalada
- **A* (Admisible)**: Heurística Manhattan escalada por costo mínimo (admisible y consistente)
- **Weighted A* (w=1.5)**: A* con peso 1.5 (compromiso velocidad-optimalidad)
- **Weighted A* (w=3.0)**: A* con peso 3.0 (más rápido, menos óptimo)

Métricas registradas:
- Costo total del camino
- Nodos expandidos
- Tiempo de CPU
- Longitud del camino

### 3. Módulo Simbólico: Planificación Clásica (`planner.py`)

Modelo STRIPS/PDDL con acciones de alto nivel:
- `mover(?desde, ?hasta)`: Consume batería, usa Módulo A para ruta física
- `tomar_muestra(?m, ?lugar)`: Requiere estar en POI y espacio en memoria
- `analizar_muestra(?m)`: Requiere muestra en memoria
- `transmitir(?m, ?antena)`: Requiere muestra analizada y cobertura
- `recargar(?base)`: Restaura batería

### 4. Módulo D: Metaheurísticas (`metaheuristics.py`)

Algoritmos para optimización del orden de visita de POIs (TSP/Orienteering):
- **Hill Climbing**: Búsqueda local voraz
- **Simulated Annealing**: Recocido simulado con enfriamiento geométrico
- **Algoritmo Genético**: Selección por torneo, cruce OX

Función de fitness: maximiza valor científico, penaliza infactibilidad.

### 5. Módulo E: Sistema Basado en Reglas (`rules.py`) [Bono]

Motor de inferencia con Encadenamiento Hacia Adelante (Forward Chaining) para toma de decisiones reactiva y estratégica:

- **14 reglas de producción** con prioridades y condiciones:
  - R0: Inferir estado vulnerable (batería <28% o tiempo <20%)
  - R1: Recargar en base si batería <95%
  - R2: Emergencia por límite de tiempo
  - R3: Emergencia por batería crítica
  - R4: Política de modo vulnerable
  - R5: Misión completada
  - R6: Transmitir con memoria llena
  - R7: Transmisión oportunista
  - R8: Buscar antena por memoria saturada
  - R9: Recolectar muestra in situ
  - R10: Analizar muestras pendientes
  - R11: Descarte por bajo valor científico
  - R12: Navegar a POI prioritario (ratio valor/costo)
  - R13: Retorno por defecto

- **Decisiones de alto nivel**: RETORNAR_A_BASE_EMERGENCIA, RECARGAR_BATERIA, IR_A_ANTENA_DESCARGAR, TRANSMITIR_DATOS, TOMAR_MUESTRA_POI, ANALIZAR_MUESTRA, EXPLORAR_POI_PRIORITARIO, MISIÓN_COMPLETADA, ESPERA_STANDBY

- **Clase `RoverRuleExpertSystem`**: Extrae hechos del estado del rover y ejecuta el motor de inferencia

## Instalación

```bash
# Crear entorno virtual (opcional pero recomendado)
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

### Ejecutar el protocolo experimental completo:

```bash
python experiments.py
```

Este script:
1. Genera 5 instancias con semillas fijas (42, 101, 202, 303, 404)
2. Ejecuta algoritmos deterministas (Módulo A) 1 vez por instancia
3. Ejecuta metaheurísticas 15 veces por instancia
4. Ejecuta el sistema de reglas (Módulo E) 1 vez por instancia
5. Imprime tablas con Media ± Desviación Estándar
6. Guarda resultados en `results.csv`

### Ejecutar módulos individualmente:

```bash
# Probar generador de instancias
python src/environment.py

# Probar algoritmos de búsqueda
python src/search.py

# Probar planificador
python src/planner.py

# Probar metaheurísticas
python src/metaheuristics.py

# Probar sistema de reglas
python src/rules.py
```

## Uso Programático

```python
from src.environment import generate_instance
from src.search import a_star

# 1. Creamos el mapa
mi_mapa = generate_instance(width=10, height=10, num_pois=2, seed=99)

# 2. Usamos la Base como inicio y el primer POI como meta (garantizados no-obstáculo)
inicio = (mi_mapa.base.x, mi_mapa.base.y)
meta = (mi_mapa.pois[0].x, mi_mapa.pois[0].y)

# 3. Calculamos la ruta óptima
camino, metricas = a_star(mi_mapa, inicio, meta)

# 4. Imprimimos el resultado
print(f"Ruta desde la Base {inicio} hasta POI_0 {meta}:")
print("Costo total:", metricas.total_cost)
print("Nodos expandidos:", metricas.nodes_expanded)
print("Longitud en pasos:", metricas.path_length)
print("Camino:", camino)


```

## Resultados Esperados

El script `experiments.py` genera:
- Tablas comparativas en terminal con formato: `Media ± Desviación`
- Archivo `results.csv` con todos los resultados detallados

## Requisitos del Sistema

- Python 3.8 o superior
- Sistema operativo: Windows, Linux o macOS

## Notas Académicas

Este proyecto se evalúa por:
- **Calidad del modelado matemático**: Representación rigurosa del problema
- **Corrección algorítmica**: Implementación correcta de los algoritmos
- **Rigor experimental**: Protocolo reproducible con estadísticas apropiadas

El proyecto NO incluye componentes visuales ni simuladores gráficos, enfocándose en los aspectos algorítmicos y matemáticos.

## Autor

Proyecto desarrollado para la asignatura de Inteligencia Artificial.
