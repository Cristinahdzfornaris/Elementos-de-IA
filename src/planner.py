"""
Módulo Simbólico: Planificación Clásica (STRIPS/PDDL)
Modela el problema con acciones de alto nivel y planificación causal integrada con Módulo A.
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq

try:
    from .environment import Environment, POI
    from .search import a_star, dijkstra, SearchMetrics
except ImportError:
    from environment import Environment, POI
    from search import a_star, dijkstra, SearchMetrics


class ActionType(Enum):
    """Tipos de acciones de alto nivel."""
    MOVE = "mover"
    COLLECT = "tomar_muestra"
    ANALYZE = "analizar_muestra"
    TRANSMIT = "transmitir"
    RECHARGE = "recargar"


@dataclass
class Action:
    """Acción de alto nivel en estilo STRIPS."""
    type: ActionType
    params: tuple
    preconditions: Set[str] = field(default_factory=set)
    effects: Set[str] = field(default_factory=set)
    cost: float = 0.0
    
    def __repr__(self):
        return f"{self.type.value}{self.params} (cost={self.cost:.2f})"


@dataclass
class PlanningState:
    """Estado para planificación (representación simbólica)."""
    rover_pos: Tuple[int, int]
    battery: float
    memory: int
    collected_pois: Set[int] = field(default_factory=set)
    analyzed_pois: Set[int] = field(default_factory=set)
    transmitted_pois: Set[int] = field(default_factory=set)
    
    def __hash__(self):
        return hash((
            self.rover_pos,
            round(self.battery, 1),
            self.memory, 
            frozenset(self.collected_pois),
            frozenset(self.analyzed_pois),
            frozenset(self.transmitted_pois)
        ))
    
    def __eq__(self, other):
        return (
            self.rover_pos == other.rover_pos and
            abs(self.battery - other.battery) < 1e-3 and
            self.memory == other.memory and
            self.collected_pois == other.collected_pois and
            self.analyzed_pois == other.analyzed_pois and
            self.transmitted_pois == other.transmitted_pois
        )


class ClassicalPlanner:
    """
    Planificador clásico tipo STRIPS/PDDL.
    Determina el orden causal de acciones de alto nivel (Base, POIs, Antenas)
    e integra la navegación física mediante A* (Módulo A).
    """
    
    def __init__(self, env: Environment):
        self.env = env
        self.max_battery = getattr(env, 'max_battery', 100.0)
        self.max_memory = getattr(env, 'max_memory', 3)
        self._path_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[list, float]] = {}

    def get_path_and_cost(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[list, float]:
        """Obtiene la ruta y costo real entre dos puntos usando A* (con caché)."""
        if start == goal:
            return [start], 0.0
        
        key = (start, goal)
        if key not in self._path_cache:
            path, metrics = a_star(self.env, start, goal)
            if not path:
                path, metrics = dijkstra(self.env, start, goal)
            cost = metrics.total_cost if path else float('inf')
            self._path_cache[key] = (path, cost)
        return self._path_cache[key]

    def get_antenna_positions(self) -> List[Tuple[int, int]]:
        """Identifica las posiciones clave de comunicación."""
        positions = []
        if hasattr(self.env, 'antennas') and self.env.antennas:
            for ant in self.env.antennas:
                if hasattr(ant, 'x') and hasattr(ant, 'y'):
                    positions.append((ant.x, ant.y))
                elif isinstance(ant, (tuple, list)):
                    positions.append((ant[0], ant[1]))
        
        if not positions:
            for y in range(self.env.height):
                for x in range(self.env.width):
                    if self.env.is_in_coverage(x, y):
                        positions.append((x, y))
                        return positions
        return positions

    def get_applicable_actions(self, state: PlanningState) -> List[Action]:
        """
        Retorna acciones aplicables. Aplica poda causal: si una acción local
        está lista (tomar, analizar, transmitir, recargar), se ejecuta directamente
        evitando ramificaciones innecesarias.
        """
        actions = []
        x, y = state.rover_pos
        base_pos = (self.env.base.x, self.env.base.y)
        antennas = self.get_antenna_positions()

        # 1. ACCIONES LOCALES DETERMINISTAS (Prioridad máxima)
        
        # A) Si estamos en Base y la batería no está llena, recargar
        if self.env.is_at_base(x, y) and state.battery < self.max_battery:
            if len(state.transmitted_pois) < len(self.env.pois):
                return [Action(
                    type=ActionType.RECHARGE,
                    params=(x, y),
                    preconditions={"at_base"},
                    effects={"battery_full"},
                    cost=0.0
                )]

        # B) Tomar muestra si estamos en el POI y hay espacio en memoria
        poi = self.env.get_poi_at(x, y)
        if poi is not None and poi.id not in state.collected_pois:
            if state.memory < self.max_memory:
                return [Action(
                    type=ActionType.COLLECT,
                    params=(poi.id, (x, y)),
                    preconditions={"at_poi", "memory_available"},
                    effects={"sample_collected"},
                    cost=1.0
                )]

        # C) Analizar muestra si hay alguna tomada en memoria
        for poi_id in sorted(state.collected_pois):
            if poi_id not in state.analyzed_pois:
                return [Action(
                    type=ActionType.ANALYZE,
                    params=(poi_id,),
                    preconditions={"sample_in_memory"},
                    effects={"sample_analyzed"},
                    cost=2.0
                )]

        # D) Transmitir si estamos en cobertura y tenemos muestras analizadas
        if self.env.is_in_coverage(x, y):
            for poi_id in sorted(state.analyzed_pois):
                if poi_id not in state.transmitted_pois:
                    return [Action(
                        type=ActionType.TRANSMIT,
                        params=(poi_id, (x, y)),
                        preconditions={"sample_analyzed", "in_coverage"},
                        effects={"sample_transmitted", "memory_freed"},
                        cost=1.0
                    )]

        # 2. ACCIONES DE MOVIMIENTO DE ALTO NIVEL (Entre ubicaciones clave)
        
        # Si ya se transmitió todo, la única meta es volver a la base
        if len(state.transmitted_pois) == len(self.env.pois):
            if not self.env.is_at_base(x, y):
                path, move_cost = self.get_path_and_cost((x, y), base_pos)
                if path and state.battery >= move_cost:
                    actions.append(Action(
                        type=ActionType.MOVE,
                        params=((x, y), base_pos),
                        preconditions={"has_battery"},
                        effects={"at_base"},
                        cost=move_cost
                    ))
            return actions

        # Si la memoria está llena, estamos obligados a ir a una antena
        if state.memory >= self.max_memory:
            for ant_pos in antennas:
                if (x, y) != ant_pos:
                    path, move_cost = self.get_path_and_cost((x, y), ant_pos)
                    if path and state.battery >= move_cost:
                        actions.append(Action(
                            type=ActionType.MOVE,
                            params=((x, y), ant_pos),
                            preconditions={"has_battery"},
                            effects={"in_coverage"},
                            cost=move_cost
                        ))
            if not actions and not self.env.is_at_base(x, y):
                path, move_cost = self.get_path_and_cost((x, y), base_pos)
                if path and state.battery >= move_cost:
                    actions.append(Action(
                        type=ActionType.MOVE,
                        params=((x, y), base_pos),
                        preconditions={"has_battery"},
                        effects={"at_base"},
                        cost=move_cost
                    ))
            return actions

        # Caso general: moverse a POIs pendientes
        uncollected_pois = [p for p in self.env.pois if p.id not in state.collected_pois]
        can_reach_any = False

        for target_poi in uncollected_pois:
            target_pos = (target_poi.x, target_poi.y)
            if (x, y) != target_pos:
                path, move_cost = self.get_path_and_cost((x, y), target_pos)
                dist_to_base = self.get_path_and_cost(target_pos, base_pos)[1]
                safe_battery = move_cost + (dist_to_base if dist_to_base != float('inf') else 0)

                if path and state.battery >= safe_battery:
                    can_reach_any = True
                    actions.append(Action(
                        type=ActionType.MOVE,
                        params=((x, y), target_pos),
                        preconditions={"has_battery"},
                        effects={"at_poi"},
                        cost=move_cost
                    ))

        # Si tenemos muestras analizadas pendientes de transmisión, también es opción ir a la antena
        analyzed_untransmitted = [p for p in state.analyzed_pois if p not in state.transmitted_pois]
        if analyzed_untransmitted:
            for ant_pos in antennas:
                if (x, y) != ant_pos:
                    path, move_cost = self.get_path_and_cost((x, y), ant_pos)
                    if path and state.battery >= move_cost:
                        actions.append(Action(
                            type=ActionType.MOVE,
                            params=((x, y), ant_pos),
                            preconditions={"has_battery"},
                            effects={"in_coverage"},
                            cost=move_cost
                        ))

        # Si la batería no da para ir a un POI y regresar seguro, recargar en Base
        if (not can_reach_any or state.battery < self.max_battery * 0.30) and not self.env.is_at_base(x, y):
            path, move_cost = self.get_path_and_cost((x, y), base_pos)
            if path and state.battery >= move_cost:
                actions.append(Action(
                    type=ActionType.MOVE,
                    params=((x, y), base_pos),
                    preconditions={"has_battery"},
                    effects={"at_base"},
                    cost=move_cost
                ))

        return actions
    
    def apply_action(self, state: PlanningState, action: Action) -> PlanningState:
        """Aplica una acción y retorna el nuevo estado."""
        new_state = PlanningState(
            rover_pos=state.rover_pos,
            battery=state.battery,
            memory=state.memory,
            collected_pois=set(state.collected_pois),
            analyzed_pois=set(state.analyzed_pois),
            transmitted_pois=set(state.transmitted_pois)
        )
        
        if action.type == ActionType.RECHARGE:
            new_state.battery = self.max_battery
        elif action.type == ActionType.MOVE:
            _, to_pos = action.params
            new_state.rover_pos = to_pos
            new_state.battery -= action.cost
        elif action.type == ActionType.COLLECT:
            poi_id, _ = action.params
            new_state.collected_pois.add(poi_id)
            new_state.memory += 1
            new_state.battery -= action.cost
        elif action.type == ActionType.ANALYZE:
            poi_id = action.params[0]
            new_state.analyzed_pois.add(poi_id)
            new_state.battery -= action.cost
        elif action.type == ActionType.TRANSMIT:
            poi_id, _ = action.params
            new_state.transmitted_pois.add(poi_id)
            new_state.memory -= 1
            new_state.battery -= action.cost
        
        return new_state
    
    def is_goal_state(self, state: PlanningState) -> bool:
        """Meta: todos los POIs transmitidos y rover en la base."""
        all_transmitted = len(state.transmitted_pois) == len(self.env.pois)
        at_base = self.env.is_at_base(state.rover_pos[0], state.rover_pos[1])
        return all_transmitted and at_base

    def _heuristic(self, state: PlanningState) -> float:
        """Heurística para guiar A* en el espacio de planes."""
        base_pos = (self.env.base.x, self.env.base.y)
        h = 0.0
        
        pending_collect = [p for p in self.env.pois if p.id not in state.collected_pois]
        h += len(pending_collect) * 1.0
        h += (len(state.collected_pois) - len(state.analyzed_pois)) * 2.0
        h += (len(self.env.pois) - len(state.transmitted_pois)) * 1.0

        if pending_collect:
            min_dist = min(self.get_path_and_cost(state.rover_pos, (p.x, p.y))[1] for p in pending_collect)
            h += min_dist if min_dist != float('inf') else 5.0
        elif not self.env.is_at_base(state.rover_pos[0], state.rover_pos[1]):
            dist_base = self.get_path_and_cost(state.rover_pos, base_pos)[1]
            h += dist_base if dist_base != float('inf') else 5.0

        return h
    
    def forward_search(self, max_depth: int = 200) -> Tuple[List[Action], float]:
        """Búsqueda A* en el espacio de planes de alto nivel."""
        initial_state = PlanningState(
            rover_pos=(self.env.base.x, self.env.base.y),
            battery=self.max_battery,
            memory=0
        )
        
        counter = 0
        
        queue = [(0.0, counter, 0.0, initial_state, [])]
        visited_costs = {}
        
        while queue:
            f_val, _, cost, state, plan = heapq.heappop(queue)
            
            if self.is_goal_state(state):
                return plan, cost
            
            if len(plan) >= max_depth:
                continue
            
            state_key = (
                state.rover_pos,
                round(state.battery, 1),
                state.memory,
                frozenset(state.collected_pois),
                frozenset(state.analyzed_pois),
                frozenset(state.transmitted_pois)
            )
            
            if state_key in visited_costs and visited_costs[state_key] <= cost:
                continue
            visited_costs[state_key] = cost
            
            for action in self.get_applicable_actions(state):
                new_state = self.apply_action(state, action)
                if new_state.battery < 0:
                    continue
                
                new_plan = plan + [action]
                new_cost = cost + action.cost
                h_cost = self._heuristic(new_state)
                
                counter += 1
                heapq.heappush(queue, (new_cost + h_cost, counter, new_cost, new_state, new_plan))
        
        return [], float('inf')
    
    def plan_mission(self) -> Tuple[List[Action], float, Dict]:
        """Ejecuta la planificación y retorna el plan, su costo y métricas."""
        plan, total_cost = self.forward_search(max_depth=200)
        
        stats = {
            "plan_length": len(plan),
            "total_cost": total_cost,
            "actions_by_type": {}
        }
        
        for action in plan:
            action_type = action.type.value
            stats["actions_by_type"][action_type] = stats["actions_by_type"].get(action_type, 0) + 1
        
        return plan, total_cost, stats