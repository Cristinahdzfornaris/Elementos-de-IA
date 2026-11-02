"""
Módulo A: Búsqueda en Espacios de Estados
Implementa algoritmos de búsqueda para navegación en rejilla con costos variables.
"""

import heapq
import time
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

try:
    from .environment import Environment
except ImportError:
    from environment import Environment


@dataclass
class SearchMetrics:
    """Métricas de búsqueda."""
    total_cost: float
    nodes_expanded: int
    cpu_time: float
    path_length: int


def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Distancia Manhattan entre dos puntos."""
    return abs(x1 - x2) + abs(y1 - y2)


def euclidean_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Distancia Euclídea entre dos puntos."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def dijkstra(env: Environment, start: Tuple[int, int], 
             goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], SearchMetrics]:
    """
    Búsqueda de Costo Uniforme (Dijkstra).
    f(n) = g(n)
    
    Args:
        env: Entorno del rover
        start: Posición inicial (x, y)
        goal: Posición objetivo (x, y)
    
    Returns:
        (path, metrics): Camino como lista de posiciones y métricas
    """
    start_time = time.time()
    nodes_expanded = 0
    
    # Estado: (g_cost, position, parent)
    # Usamos priority queue ordenada por g_cost
    open_set = []
    heapq.heappush(open_set, (0.0, start, None))
    
    # Diccionario para rastrear el mejor g_cost para cada posición
    g_costs = {start: 0.0}
    
    # Para reconstruir el camino
    came_from = {}
    
    while open_set:
        current_g, current, parent = heapq.heappop(open_set)
        
        # Si ya encontramos un mejor camino a current, saltar
        if current_g > g_costs.get(current, float('inf')):
            continue
        
        came_from[current] = parent
        nodes_expanded += 1
        
        # Prueba de meta al expandir (según especificación)
        if current == goal:
            end_time = time.time()
            path = reconstruct_path(came_from, start, goal)
            metrics = SearchMetrics(
                total_cost=current_g,
                nodes_expanded=nodes_expanded,
                cpu_time=end_time - start_time,
                path_length=len(path)
            )
            return path, metrics
        
        # Expandir vecinos
        for neighbor in env.get_neighbors(current[0], current[1]):
            edge_cost = env.get_movement_cost(current[0], current[1], 
                                               neighbor[0], neighbor[1])
            tentative_g = current_g + edge_cost
            
            if tentative_g < g_costs.get(neighbor, float('inf')):
                g_costs[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor, current))
    
    # No se encontró camino
    end_time = time.time()
    metrics = SearchMetrics(
        total_cost=float('inf'),
        nodes_expanded=nodes_expanded,
        cpu_time=end_time - start_time,
        path_length=0
    )
    return [], metrics


def a_star(env: Environment, start: Tuple[int, int], goal: Tuple[int, int],
           heuristic: Callable = None, weight: float = 1.0) -> Tuple[List[Tuple[int, int]], SearchMetrics]:
    """
    Algoritmo A* con heurística configurable.
    f(n) = g(n) + w * h(n)
    
    Args:
        env: Entorno del rover
        start: Posición inicial (x, y)
        goal: Posición objetivo (x, y)
        heuristic: Función heurística (default: Manhattan)
        weight: Peso de la heurística (1.0 para A* estándar, >1 para Weighted A*)
    
    Returns:
        (path, metrics): Camino como lista de posiciones y métricas
    """
    if heuristic is None:
        # Heurística 1: Manhattan escalada por costo mínimo (1.0)
        heuristic = lambda x1, y1, x2, y2: manhattan_distance(x1, y1, x2, y2) * 1.0
    
    start_time = time.time()
    nodes_expanded = 0
    
    # Estado: (f_cost, g_cost, position, parent)
    open_set = []
    h_start = heuristic(start[0], start[1], goal[0], goal[1])
    heapq.heappush(open_set, (h_start, 0.0, start, None))
    
    # Diccionarios para rastrear costos
    g_costs = {start: 0.0}
    
    # Para reconstruir el camino
    came_from = {}
    
    while open_set:
        current_f, current_g, current, parent = heapq.heappop(open_set)
        
        # Si ya encontramos un mejor camino a current, saltar
        if current_g > g_costs.get(current, float('inf')):
            continue
        
        came_from[current] = parent
        nodes_expanded += 1
        
        # Prueba de meta al expandir (según especificación)
        if current == goal:
            end_time = time.time()
            path = reconstruct_path(came_from, start, goal)
            metrics = SearchMetrics(
                total_cost=current_g,
                nodes_expanded=nodes_expanded,
                cpu_time=end_time - start_time,
                path_length=len(path)
            )
            return path, metrics
        
        # Expandir vecinos
        for neighbor in env.get_neighbors(current[0], current[1]):
            edge_cost = env.get_movement_cost(current[0], current[1], 
                                               neighbor[0], neighbor[1])
            tentative_g = current_g + edge_cost
            
            if tentative_g < g_costs.get(neighbor, float('inf')):
                g_costs[neighbor] = tentative_g
                h_neighbor = heuristic(neighbor[0], neighbor[1], goal[0], goal[1])
                f_neighbor = tentative_g + weight * h_neighbor
                heapq.heappush(open_set, (f_neighbor, tentative_g, neighbor, current))
    
    # No se encontró camino
    end_time = time.time()
    metrics = SearchMetrics(
        total_cost=float('inf'),
        nodes_expanded=nodes_expanded,
        cpu_time=end_time - start_time,
        path_length=0
    )
    return [], metrics


def weighted_a_star(env: Environment, start: Tuple[int, int], 
                    goal: Tuple[int, int], weight: float = 2.0) -> Tuple[List[Tuple[int, int]], SearchMetrics]:
    """
    Weighted A* (Heurística 2): A* con peso w > 1.
    No admisible pero más rápido.
    
    Args:
        env: Entorno del rover
        start: Posición inicial (x, y)
        goal: Posición objetivo (x, y)
        weight: Peso de la heurística (típicamente 2.0 a 5.0)
    
    Returns:
        (path, metrics): Camino como lista de posiciones y métricas
    """
    return a_star(env, start, goal, weight=weight)


def reconstruct_path(came_from: dict, start: Tuple[int, int], 
                     goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Reconstruye el camino desde start hasta goal usando el diccionario came_from.
    """
    path = []
    current = goal
    
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    
    path.reverse()
    return path


if __name__ == "__main__":
    # Prueba de los algoritmos
    try:
        from .environment import generate_instance
    except ImportError:
        from environment import generate_instance
    
    env = generate_instance(20, 20, num_pois=5, rock_ratio=0.15, seed=42)
    
    start = (env.base.x, env.base.y)
    if env.pois:
        goal = (env.pois[0].x, env.pois[0].y)
    else:
        goal = (env.width - 1, env.height - 1)
    
    print(f"Start: {start}, Goal: {goal}")
    
    # Dijkstra
    path, metrics = dijkstra(env, start, goal)
    print(f"\nDijkstra:")
    print(f"  Path length: {metrics.path_length}")
    print(f"  Total cost: {metrics.total_cost:.2f}")
    print(f"  Nodes expanded: {metrics.nodes_expanded}")
    print(f"  CPU time: {metrics.cpu_time:.4f}s")
    
    # A* con Manhattan
    path, metrics = a_star(env, start, goal)
    print(f"\nA* (Manhattan):")
    print(f"  Path length: {metrics.path_length}")
    print(f"  Total cost: {metrics.total_cost:.2f}")
    print(f"  Nodes expanded: {metrics.nodes_expanded}")
    print(f"  CPU time: {metrics.cpu_time:.4f}s")
    
    # Weighted A*
    path, metrics = weighted_a_star(env, start, goal, weight=2.0)
    print(f"\nWeighted A* (w=2.0):")
    print(f"  Path length: {metrics.path_length}")
    print(f"  Total cost: {metrics.total_cost:.2f}")
    print(f"  Nodes expanded: {metrics.nodes_expanded}")
    print(f"  CPU time: {metrics.cpu_time:.4f}s")
