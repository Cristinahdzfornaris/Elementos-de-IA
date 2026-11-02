"""
Módulo de Entorno y Generador de Instancias
Representa el entorno del rover como una rejilla 2D con costos variables.
"""

import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class CellType(Enum):
    """Tipos de celda con sus costos de movimiento."""
    PLAIN = 1.0      # Plano/despejado
    SAND = 2.0       # Arena/dunas
    ROCKY = 4.0      # Rocoso/piedras
    OBSTACLE = float('inf')  # Intransitable


@dataclass
class POI:
    """Punto de Interés Científico."""
    id: int
    x: int
    y: int
    scientific_value: float
    sample_collected: bool = False
    sample_analyzed: bool = False


@dataclass
class Antenna:
    """Antena de comunicación con cobertura."""
    x: int
    y: int
    coverage_radius: int = 3


@dataclass
class Base:
    """Base con estación de recarga."""
    x: int
    y: int


@dataclass
class RoverState:
    """Estado del rover."""
    x: int
    y: int
    battery: float
    memory: int  # Número de muestras almacenadas
    max_memory: int
    max_battery: float


@dataclass
class SearchMetrics:
    """Métricas de búsqueda."""
    total_cost: float
    nodes_expanded: int
    cpu_time: float
    path_length: int


class Environment:
    """
    Entorno del rover: rejilla 2D con costos variables, base, antenas y POIs.
    """
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid: List[List[CellType]] = []
        self.base: Optional[Base] = None
        self.antennas: List[Antenna] = []
        self.pois: List[POI] = []
        self.rover: Optional[RoverState] = None
        self.max_battery: float = 100.0
        self.max_memory: int = 3
        self.mission_time_limit: float = 1000.0
        
    def get_cell_cost(self, x: int, y: int) -> float:
        """Retorna el costo de movimiento en la celda (x, y)."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return float('inf')
        return self.grid[y][x].value
    
    def is_obstacle(self, x: int, y: int) -> bool:
        """Verifica si la celda es un obstáculo."""
        return self.get_cell_cost(x, y) == float('inf')
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """Verifica si la posición es válida y no es obstáculo."""
        return 0 <= x < self.width and 0 <= y < self.height and not self.is_obstacle(x, y)
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Retorna las posiciones vecinas válidas (4-conectividad).
        Cada vecino es una tupla (nx, ny).
        """
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def get_movement_cost(self, from_x: int, from_y: int, to_x: int, to_y: int) -> float:
        """
        Retorna el costo de moverse de (from_x, from_y) a (to_x, to_y).
        El costo es el costo de la celda destino.
        """
        return self.get_cell_cost(to_x, to_y)
    
    def is_in_coverage(self, x: int, y: int) -> bool:
        """Verifica si la posición está en cobertura de alguna antena."""
        for antenna in self.antennas:
            dist = abs(x - antenna.x) + abs(y - antenna.y)
            if dist <= antenna.coverage_radius:
                return True
        return False
    
    def get_poi_at(self, x: int, y: int) -> Optional[POI]:
        """Retorna el POI en la posición (x, y) si existe."""
        for poi in self.pois:
            if poi.x == x and poi.y == y:
                return poi
        return None
    
    def is_at_base(self, x: int, y: int) -> bool:
        """Verifica si la posición es la base."""
        return self.base is not None and x == self.base.x and y == self.base.y
    
    def __repr__(self):
        return (f"Environment({self.width}x{self.height}, "
                f"{len(self.pois)} POIs, {len(self.antennas)} antennas)")


def generate_instance(width: int, height: int, num_pois: int, 
                      rock_ratio: float = 0.15, seed: int = 42) -> Environment:
    """
    Genera una instancia del entorno de forma determinista dada la semilla.
    
    Args:
        width: Ancho de la rejilla
        height: Alto de la rejilla
        num_pois: Número de puntos de interés científico
        rock_ratio: Proporción de celdas rocosas (0.0 a 1.0)
        seed: Semilla para reproducibilidad
    
    Returns:
        Environment: Instancia generada
    """
    rng = random.Random(seed)
    
    env = Environment(width, height)
    
    # 1. Generar rejilla con tipos de terreno
    # Distribución: 60% plano, 25% arena, rock_ratio% rocoso, resto obstáculos
    total_cells = width * height
    num_rocky = int(total_cells * rock_ratio)
    num_sand = int(total_cells * 0.25)
    num_obstacles = int(total_cells * 0.05)
    num_plain = total_cells - num_rocky - num_sand - num_obstacles
    
    cells = ([CellType.PLAIN] * num_plain + 
             [CellType.SAND] * num_sand + 
             [CellType.ROCKY] * num_rocky + 
             [CellType.OBSTACLE] * num_obstacles)
    
    rng.shuffle(cells)
    
    # Llenar rejilla
    env.grid = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(cells[y * width + x])
        env.grid.append(row)
    
    # 2. Colocar base en posición aleatoria válida
    while True:
        bx, by = rng.randint(0, width - 1), rng.randint(0, height - 1)
        if env.is_valid_position(bx, by):
            env.base = Base(bx, by)
            # Asegurar que la base no sea obstáculo
            env.grid[by][bx] = CellType.PLAIN
            break
    
    # 3. Colocar 1 o 2 antenas en posiciones válidas
    num_antennas = rng.choice([1, 2])
    for _ in range(num_antennas):
        while True:
            ax, ay = rng.randint(0, width - 1), rng.randint(0, height - 1)
            if env.is_valid_position(ax, ay) and not env.is_at_base(ax, ay):
                env.antennas.append(Antenna(ax, ay, coverage_radius=3))
                break
    
    # 4. Colocar POIs en posiciones válidas
    valid_positions = []
    for y in range(height):
        for x in range(width):
            if env.is_valid_position(x, y) and not env.is_at_base(x, y):
                valid_positions.append((x, y))
    
    rng.shuffle(valid_positions)
    
    for i in range(min(num_pois, len(valid_positions))):
        x, y = valid_positions[i]
        # Valor científico aleatorio entre 10 y 100
        scientific_value = rng.uniform(10.0, 100.0)
        env.pois.append(POI(id=i, x=x, y=y, scientific_value=scientific_value))
    
    # 5. Configurar recursos del rover
    env.max_battery = rng.uniform(100.0, 150.0)
    env.max_memory = 3
    env.mission_time_limit = 1000.0
    
    # Inicializar rover en la base
    env.rover = RoverState(
        x=env.base.x,
        y=env.base.y,
        battery=env.max_battery,
        memory=0,
        max_memory=env.max_memory,
        max_battery=env.max_battery
    )
    
    return env


if __name__ == "__main__":
    # Prueba del generador
    env = generate_instance(20, 20, num_pois=5, rock_ratio=0.15, seed=42)
    print(env)
    print(f"Base: ({env.base.x}, {env.base.y})")
    print(f"Antennas: {[(a.x, a.y) for a in env.antennas]}")
    print(f"POIs: {[(p.x, p.y, p.scientific_value) for p in env.pois]}")
    print(f"Rover battery: {env.rover.battery}")
