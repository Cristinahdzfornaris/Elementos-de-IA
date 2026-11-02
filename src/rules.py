"""
Módulo E (Opcional - Bonificable): Sistemas Basados en Reglas
Implementa un motor de inferencia con Encadenamiento Hacia Adelante (Forward Chaining)
para la toma reactiva y estratégica de decisiones del rover autónomo.
"""

from typing import Dict, Any, List, Set, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from .environment import Environment
except ImportError:
    from environment import Environment


class Decision(Enum):
    """Acciones de alto nivel recomendadas por el sistema de reglas."""
    EMERGENCY_RETURN = "RETORNAR_A_BASE_EMERGENCIA"
    RECHARGE = "RECARGAR_BATERIA"
    GO_TO_ANTENNA = "IR_A_ANTENA_DESCARGAR"
    TRANSMIT = "TRANSMITIR_DATOS"
    COLLECT_SAMPLE = "TOMAR_MUESTRA_POI"
    ANALYZE_SAMPLE = "ANALIZAR_MUESTRA"
    EXPLORE_POI = "EXPLORAR_POI_PRIORITARIO"
    MISSION_COMPLETE = "MISIÓN_COMPLETADA"
    STANDBY = "ESPERA_STANDBY"


@dataclass
class Rule:
    """Regla de producción: SI (conditions) ENTONCES (consequent)"""
    name: str
    priority: int
    conditions: List[Callable[[Dict[str, Any]], bool]]
    consequent: Callable[[Dict[str, Any]], Dict[str, Any]]
    description: str = ""

    def evaluate(self, facts: Dict[str, Any]) -> bool:
        return all(cond(facts) for cond in self.conditions)

    def fire(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        return self.consequent(facts)


class ForwardChainingEngine:
    """Motor de inferencia mediante Encadenamiento Hacia Adelante."""

    def __init__(self, rules: List[Rule]):
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)

    def run(self, initial_facts: Dict[str, Any], max_cycles: int = 50) -> Tuple[Dict[str, Any], List[str]]:
        facts = initial_facts.copy()
        fired_rules: List[str] = []
        applied_in_session: Set[str] = set()

        for _ in range(max_cycles):
            rule_fired_in_cycle = False

            for rule in self.rules:
                if rule.name in applied_in_session:
                    continue

                if rule.evaluate(facts):
                    new_facts = rule.fire(facts)
                    facts.update(new_facts)
                    fired_rules.append(rule.name)
                    applied_in_session.add(rule.name)
                    rule_fired_in_cycle = True
                    break

            if not rule_fired_in_cycle:
                break

        return facts, fired_rules


class RoverRuleExpertSystem:
    """
    Sistema experto para el rover autónomo.
    Mapea el estado del entorno a hechos y evalúa las reglas de decisión.
    """

    def __init__(self):
        self.rules = self._build_knowledge_base()
        self.engine = ForwardChainingEngine(self.rules)

    def _build_knowledge_base(self) -> List[Rule]:
        rules = [
            Rule(
                name="R0_Inferir_Estado_Vulnerable",
                priority=105,
                conditions=[
                    lambda f: (f.get("battery_pct", 1.0) < 0.28 or f.get("time_pct", 1.0) < 0.20),
                    lambda f: not f.get("rover_vulnerable", False)
                ],
                consequent=lambda f: {"rover_vulnerable": True, "operational_mode": "CONSERVADOR"},
                description="Deduce estado de vulnerabilidad si los recursos son escasos"
            ),
            Rule(
                name="R1_Critico_Bateria_En_Base",
                priority=100,
                conditions=[
                    lambda f: f.get("at_base", False),
                    lambda f: f.get("battery_pct", 1.0) < 0.95
                ],
                consequent=lambda f: {
                    "decision": Decision.RECHARGE,
                    "reason": "En la base con batería incompleta; recargar"
                },
                description="Recargar en base si la batería no está al 100%"
            ),
            Rule(
                name="R2_Emergencia_Limite_Tiempo",
                priority=98,
                conditions=[
                    lambda f: not f.get("at_base", False),
                    lambda f: f.get("time_remaining", 1000.0) <= f.get("cost_to_base", 0.0) * 1.30
                ],
                consequent=lambda f: {
                    "decision": Decision.EMERGENCY_RETURN,
                    "reason": "Tiempo de misión al límite; regresar inmediatamente a la base"
                },
                description="Abortar por tiempo límite"
            ),
            Rule(
                name="R3_Emergencia_Bateria_Retorno",
                priority=95,
                conditions=[
                    lambda f: not f.get("at_base", False),
                    lambda f: f.get("battery_remaining", 100.0) <= f.get("cost_to_base", 0.0) * 1.25
                ],
                consequent=lambda f: {
                    "decision": Decision.EMERGENCY_RETURN,
                    "reason": "Batería crítica; regresar inmediatamente a la base"
                },
                description="Abortar por batería crítica"
            ),
            Rule(
                name="R4_Politica_Modo_Vulnerable",
                priority=92,
                conditions=[
                    lambda f: f.get("rover_vulnerable", False) is True,
                    lambda f: f.get("cost_to_base", 0.0) > 8.0,
                    lambda f: not f.get("at_base", False)
                ],
                consequent=lambda f: {
                    "decision": Decision.EMERGENCY_RETURN,
                    "reason": "Rover en modo vulnerable y alejado de la base"
                },
                description="Retorno preventivo en modo vulnerable"
            ),
            Rule(
                name="R5_Mision_Completada",
                priority=90,
                conditions=[
                    lambda f: f.get("pending_pois", 1) == 0,
                    lambda f: f.get("memory_used", 0) == 0,
                    lambda f: f.get("at_base", False)
                ],
                consequent=lambda f: {
                    "decision": Decision.MISSION_COMPLETE,
                    "reason": "Todos los POIs explorados y transmitidos; rover en base"
                },
                description="Misión completada"
            ),
            Rule(
                name="R6_Transmitir_Memoria_Llena",
                priority=85,
                conditions=[
                    lambda f: f.get("in_coverage", False),
                    lambda f: f.get("analyzed_samples", 0) > 0,
                    lambda f: f.get("memory_used", 0) >= f.get("max_memory", 3)
                ],
                consequent=lambda f: {
                    "decision": Decision.TRANSMIT,
                    "reason": "Memoria llena y en cobertura; transmitir para liberar espacio"
                },
                description="Transmitir con memoria llena"
            ),
            Rule(
                name="R7_Transmision_Oportunista",
                priority=82,
                conditions=[
                    lambda f: f.get("in_coverage", False),
                    lambda f: f.get("analyzed_samples", 0) > 0,
                    lambda f: f.get("memory_used", 0) < f.get("max_memory", 3)
                ],
                consequent=lambda f: {
                    "decision": Decision.TRANSMIT,
                    "reason": "Aprovechar cobertura de paso para liberar memoria anticipadamente"
                },
                description="Transmisión oportunista"
            ),
            Rule(
                name="R8_Memoria_Saturada_Buscar_Antena",
                priority=80,
                conditions=[
                    lambda f: f.get("memory_used", 0) >= f.get("max_memory", 3),
                    lambda f: not f.get("in_coverage", False)
                ],
                consequent=lambda f: {
                    "decision": Decision.GO_TO_ANTENNA,
                    "reason": "Memoria saturada; navegar hacia la antena más cercana"
                },
                description="Buscar antena por memoria llena"
            ),
            Rule(
                name="R9_Recolectar_Muestra_In_Situ",
                priority=65,
                conditions=[
                    lambda f: f.get("at_uncollected_poi", False),
                    lambda f: f.get("memory_used", 0) < f.get("max_memory", 3),
                    lambda f: f.get("battery_pct", 1.0) > 0.15
                ],
                consequent=lambda f: {
                    "decision": Decision.COLLECT_SAMPLE,
                    "reason": "Sobre un POI inexplorado con memoria disponible; tomar muestra"
                },
                description="Tomar muestra in situ"
            ),
            Rule(
                name="R10_Analizar_Muestra_Pendiente",
                priority=60,
                conditions=[
                    lambda f: f.get("unanalyzed_samples", 0) > 0,
                    lambda f: f.get("battery_pct", 1.0) > 0.15
                ],
                consequent=lambda f: {
                    "decision": Decision.ANALYZE_SAMPLE,
                    "reason": "Analizar muestras almacenadas en memoria"
                },
                description="Analizar muestras a bordo"
            ),
            Rule(
                name="R11_Descarte_Bajo_Valor_Cientifico",
                priority=50,
                conditions=[
                    lambda f: f.get("battery_pct", 1.0) < 0.40,
                    lambda f: f.get("best_target_value", 100.0) < 25.0,
                    lambda f: not f.get("at_base", False)
                ],
                consequent=lambda f: {
                    "decision": Decision.EMERGENCY_RETURN,
                    "reason": "Objetivos restantes tienen bajo valor (<25) con batería <40%; recargar en base"
                },
                description="Descarte por costo/beneficio"
            ),
            Rule(
                name="R12_Navegar_A_POI_Prioritario",
                priority=40,
                conditions=[
                    lambda f: f.get("pending_pois", 0) > 0,
                    lambda f: f.get("memory_used", 0) < f.get("max_memory", 3),
                    lambda f: f.get("battery_pct", 1.0) > 0.25,
                    lambda f: f.get("best_target_poi_id") is not None
                ],
                consequent=lambda f: {
                    "decision": Decision.EXPLORE_POI,
                    "target_poi": f.get("best_target_poi_id"),
                    "reason": f"Navegar hacia el POI {f.get('best_target_poi_id')} con óptimo ratio valor/costo"
                },
                description="Priorización de objetivos"
            ),
            Rule(
                name="R13_Retorno_Por_Defecto",
                priority=20,
                conditions=[
                    lambda f: f.get("pending_pois", 0) == 0,
                    lambda f: not f.get("at_base", False)
                ],
                consequent=lambda f: {
                    "decision": Decision.EMERGENCY_RETURN,
                    "reason": "No quedan objetivos pendientes; regresar a la base"
                },
                description="Retorno por defecto"
            )
        ]
        return rules

    def extract_facts_from_rover(
        self,
        env,
        rover_pos: Tuple[int, int],
        battery: float,
        memory_used: int,
        collected_ids: Set[int],
        analyzed_ids: Set[int],
        transmitted_ids: Set[int],
        time_elapsed: float = 0.0,
        dist_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        x, y = rover_pos
        base_pos = (env.base.x, env.base.y)
        
        if dist_fn:
            cost_to_base = dist_fn(rover_pos, base_pos)
        else:
            cost_to_base = abs(x - base_pos[0]) + abs(y - base_pos[1])

        uncollected = [p for p in env.pois if p.id not in collected_ids]
        current_poi = env.get_poi_at(x, y)

        best_poi_id = None
        best_poi_val = 0.0
        best_ratio = -1.0

        for p in uncollected:
            dist = dist_fn(rover_pos, (p.x, p.y)) if dist_fn else (abs(x - p.x) + abs(y - p.y))
            dist = max(1.0, dist)
            ratio = p.scientific_value / dist
            if ratio > best_ratio:
                best_ratio = ratio
                best_poi_id = p.id
                best_poi_val = p.scientific_value

        mission_limit = getattr(env, "mission_time_limit", 1000.0)
        time_remaining = max(0.0, mission_limit - time_elapsed)

        facts = {
            "rover_pos": rover_pos,
            "battery_remaining": battery,
            "battery_pct": battery / env.max_battery,
            "time_remaining": time_remaining,
            "time_pct": time_remaining / mission_limit,
            "max_memory": env.max_memory,
            "memory_used": memory_used,
            "at_base": env.is_at_base(x, y),
            "in_coverage": env.is_in_coverage(x, y),
            "at_uncollected_poi": (current_poi is not None and current_poi.id not in collected_ids),
            "pending_pois": len(uncollected),
            "analyzed_samples": len(analyzed_ids - transmitted_ids),
            "unanalyzed_samples": len(collected_ids - analyzed_ids),
            "cost_to_base": cost_to_base,
            "best_target_poi_id": best_poi_id,
            "best_target_value": best_poi_val
        }
        return facts

    def decide(self, facts: Dict[str, Any]) -> Tuple[Decision, str, List[str]]:
        inferred_facts, fired_rules = self.engine.run(facts)
        decision = inferred_facts.get("decision", Decision.STANDBY)
        reason = inferred_facts.get("reason", "No se disparó ninguna regla concluyente")
        return decision, reason, fired_rules