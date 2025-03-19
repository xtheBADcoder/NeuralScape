import unreal
import json
import google.generativeai as genai
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor

# ---- Configuration ----
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"  # Or your desired Gemini model
GEMINI_MAX_OUTPUT_TOKENS = 1500
GEMINI_TEMPERATURE = 0.7
VALID_WATER_BODY_TYPES = ["WaterBodyLake", "WaterBodyOcean", "WaterBodyRiver", "WaterBodyIsland"]

# Get prompt from passed in parameter
prompt = (my_text_string)  # Parameter from Execute Python Script (Assumed)

if prompt:
    print(f"Text Content: {prompt}")
else:
    print("No text prompt provided.")


def asset_matches_keywords(asset, search_string):
    """
    Returns True if the asset's name matches keywords, balancing precision and recall.
    Prioritizes exact keyword matches, then falls back to substring matching for synonyms and keywords.
    """
    if not asset or not search_string:
        unreal.log_warning(f"asset_matches_keywords - Early exit: Asset or search_string is None/empty. Asset: {asset}, Search String: {search_string}")
        return False

    asset_name = asset.get_name().lower()
    normalized_asset_name = re.sub(r'[_\-]+', ' ', asset_name)
    keywords = [k.strip() for k in search_string.lower().split(",") if k.strip()]
    synonyms = {
        "tree": ["tree", "pine", "oak", "birch", "maple"],
        "grass": ["grass", "turf", "sod"],
        "rock": ["rock", "stone", "boulder"],
        "fireflies": ["fireflies", "lightning bugs", "glowbugs", "bioluminescent particles", "ambient particles"] # Slightly broader synonyms
    }
    final_keywords = []
    for keyword in keywords:
        if keyword in synonyms:
            final_keywords.extend(synonyms[keyword])
        else:
            final_keywords.append(keyword)

    unreal.log_warning(f"asset_matches_keywords - Processing Asset: '{asset_name}', Search String: '{search_string}', Keywords: {keywords}, Final Keywords: {final_keywords}")

    # --- Debugging Logs ---
    unreal.log_warning(f"  - Asset Name (raw): '{asset.get_name()}'") # Raw name for comparison
    unreal.log_warning(f"  - Asset Name (lower): '{asset_name}'")
    unreal.log_warning(f"  - Normalized Asset Name: '{normalized_asset_name}'")
    unreal.log_warning(f"  - Search String: '{search_string}'")
    unreal.log_warning(f"  - Keywords: {keywords}")
    unreal.log_warning(f"  - Final Keywords (with synonyms): {final_keywords}")
    # --- End Debugging Logs ---

    for keyword in keywords:  # 1. Check for exact whole-word keyword matches (highest priority)
        # --- Normalize the keyword here before creating the regex ---
        normalized_keyword = re.sub(r'[_\-]+', ' ', keyword)
        pattern = re.compile(r'\b' + re.escape(normalized_keyword) + r'\b', re.IGNORECASE)
        if pattern.search(normalized_asset_name):
            unreal.log_warning(f"Asset '{asset_name}' - Exact Keyword Match: '{keyword}'")
            return True

    for keyword in final_keywords:  # 2. Check for whole-word synonym matches (medium priority)
        # --- Normalize the synonym keyword here before creating the regex ---
        normalized_keyword = re.sub(r'[_\-]+', ' ', keyword)
        pattern = re.compile(r'\b' + re.escape(normalized_keyword) + r'\b', re.IGNORECASE)
        if pattern.search(normalized_asset_name):
            original_keyword = keywords[final_keywords.index(keyword) % len(keywords)] if keyword in synonyms.get(keywords[0], []) else keywords[0] # Correctly get original keyword
            unreal.log_warning(f"Asset '{asset_name}' - Whole-Word Synonym Match: '{keyword}' (for '{original_keyword}')")
            return True

    for keyword in keywords:  # 3. Fallback: Substring match for keywords (lower priority - broader search)
        # --- Normalize the keyword here for substring match ---
        normalized_keyword = re.sub(r'[_\-]+', ' ', keyword)
        if normalized_keyword in normalized_asset_name:
            unreal.log_warning(f"Asset '{asset_name}' - Substring Keyword Match: '{keyword}' (Fallback)")
            return True

    for keyword in final_keywords: # 4. Fallback: Substring match for synonyms (lowest priority - broadest search)
        # --- Normalize the synonym keyword here for substring match ---
        normalized_keyword = re.sub(r'[_\-]+', ' ', keyword)
        if normalized_keyword in normalized_asset_name:
            original_keyword = keywords[final_keywords.index(keyword) % len(keywords)] if keyword in synonyms.get(keywords[0], []) else keywords[0] # Correctly get original keyword
            unreal.log_warning(f"Asset '{asset_name}' - Substring Synonym Match: '{keyword}' (for '{original_keyword}') - Fallback")
            return True

    unreal.log_warning(f"Asset '{asset_name}' - No match for search: '{search_string}'")
    return False


def get_water_volumes():
    """Gets all water body actors in the level."""
    water_volumes = []
    all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
    for actor in all_actors:
        if isinstance(actor, unreal.WaterBodyActor):
            water_volumes.append(actor)
    return water_volumes


def is_location_in_water(location):
    """Checks if a location is inside a water body."""
    water_volumes = get_water_volumes()
    for water_volume in water_volumes:
        water_body_component = water_volume.get_component_by_class(unreal.WaterBodyComponent)
        if water_body_component and water_body_component.is_point_inside_water_body(location):
            return True
    return False


def get_terrain_slope(location):
    """Calculates the slope (in degrees) at a given location."""
    terrain = unreal.EditorLevelLibrary.get_world().get_terrain()
    if not terrain:
        return 0.0
    normal = terrain.get_normal_at_location(location)
    if normal.z == 0:
        return 90.0
    return math.degrees(math.atan2(math.sqrt(normal.x * normal.x + normal.y * normal.y), normal.z))


def handle_water_bodies(ai_config):
    """
    Spawns water body actors from the configuration.
    """
    water_body_actors = {}
    for water_data in ai_config.get("water_bodies", []):
        try:
            water_type = water_data.get("type", "WaterBodyLake")
            water_location = water_data.get("location")
            water_size = water_data.get("size")  # e.g., [radius, extra_param]

            if water_type not in VALID_WATER_BODY_TYPES:
                unreal.log_error(f"Unsupported water body type: {water_type}. Valid types: {VALID_WATER_BODY_TYPES}")
                continue
            if not water_location or not water_size:
                unreal.log_error("Water body must have a location and size defined.")
                continue

            water_body_class = getattr(unreal, water_type, unreal.WaterBodyLake)
            water_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
                water_body_class, unreal.Vector(water_location[0], water_location[1], water_location[2])
            )
            if not water_actor:
                unreal.log_error(f"Failed to spawn water body actor of type {water_type}")
                continue

            water_body_component = water_actor.get_component_by_class(unreal.WaterBodyComponent)
            if not water_body_component:
                unreal.log_error(f"Failed to get WaterBodyComponent for {water_type}")
                continue

            unreal.log(f"Water body of type {water_type} spawned at {water_location}.")
            water_body_actors[water_type] = water_actor
        except Exception as e:
            unreal.log_error(f"Error creating water body: {e}")
    return water_body_actors


def find_assets_by_type(asset_types):
    """Searches for assets of the specified types."""
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
    assets_map = {}
    unreal.log(f"Starting find_assets_by_type for asset_types: {asset_types}")
    for asset_type in asset_types:
        unreal.log(f"Processing asset_type: {asset_type}")
        class_reference = None
        if asset_type == "StaticMesh":
            class_reference = unreal.StaticMesh
        elif asset_type == "Texture":
            class_reference = unreal.Texture
        elif asset_type == "NiagaraSystem": # Add Niagara System support
            class_reference = unreal.NiagaraSystem
        else:
            unreal.log_warning(f"Unsupported asset type: {asset_type}")
            assets_map[asset_type] = []
            continue

        full_class_path_name = class_reference.static_class().get_class_path_name()
        unreal.log(f"Full class path name for {asset_type}: {full_class_path_name}")
        asset_filter = unreal.ARFilter(class_paths=[full_class_path_name])
        asset_data_list = asset_registry.get_assets(asset_filter)
        unreal.log(f"Found {len(asset_data_list)} asset data entries for {asset_type}")

        if not asset_data_list:
            unreal.log_warning(f"No assets found for {asset_type} using ARFilter with class paths: {asset_filter.class_paths}")
            assets_map[asset_type] = []
            continue

        assets = []
        for asset_data in asset_data_list:
            asset = unreal.AssetRegistryHelpers.get_asset(asset_data)
            if asset:
                assets.append(asset)
        unreal.log(f"Loaded {len(assets)} assets for {asset_type}")
        assets_map[asset_type] = assets
    unreal.log(f"Returning assets_map: {assets_map}")
    return assets_map


def spawn_static_mesh(asset, location, rotation=None, scale=None):
    """Spawns a static mesh actor at the given location."""
    if not asset or not isinstance(asset, unreal.StaticMesh):
        unreal.log_error("Invalid static mesh asset provided.")
        return None

    world = unreal.EditorLevelLibrary.get_editor_world()
    if not world:
        unreal.log_error("Failed to get editor world.")
        return None

    editor_actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if not editor_actor_subsystem:
        unreal.log_error("Failed to get EditorActorSubsystem.")
        return None

    actor_rotation = unreal.Rotator(0, 0, 0)
    if rotation and isinstance(rotation, (list, tuple)) and len(rotation) == 3:
        try:
            actor_rotation = unreal.Rotator(float(rotation[0]), float(rotation[1]), float(rotation[2]))
        except ValueError:
            unreal.log_error(f"Invalid rotation values: {rotation}. Using default rotation.")
    static_mesh_actor = editor_actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor, location, actor_rotation)
    if not static_mesh_actor:
        unreal.log_error("Failed to create Static Mesh Actor.")
        return None

    static_mesh_component = static_mesh_actor.get_component_by_class(unreal.StaticMeshComponent)
    static_mesh_component.set_static_mesh(asset)
    if scale and isinstance(scale, (list, tuple)) and len(scale) == 3:
        static_mesh_actor.set_actor_scale3d(unreal.Vector(scale[0], scale[1], scale[2]))
    static_mesh_actor.set_actor_rotation(actor_rotation, teleport_physics=False)
    static_mesh_component.set_mobility(unreal.ComponentMobility.STATIC)
    return static_mesh_actor


def spawn_niagara_system(asset, location, rotation=None, scale=None):
    """Spawns a Niagara System actor at the given location."""
    if not asset or not isinstance(asset, unreal.NiagaraSystem):
        unreal.log_error("Invalid Niagara System asset provided.")
        return None

    world = unreal.EditorLevelLibrary.get_editor_world()
    if not world:
        unreal.log_error("Failed to get editor world.")
        return None

    editor_actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if not editor_actor_subsystem:
        unreal.log_error("Failed to get EditorActorSubsystem.")
        return None

    actor_rotation = unreal.Rotator(0, 0, 0)
    if rotation and isinstance(rotation, (list, tuple)) and len(rotation) == 3:
        try:
            actor_rotation = unreal.Rotator(float(rotation[0]), float(rotation[1]), float(rotation[2]))
        except ValueError:
            unreal.log_error(f"Invalid rotation values: {rotation}. Using default rotation.")

    niagara_actor = editor_actor_subsystem.spawn_actor_from_class(unreal.NiagaraActor, location, actor_rotation)
    if not niagara_actor:
        unreal.log_error("Failed to create Niagara Actor.")
        return None

    niagara_component = niagara_actor.get_component_by_class(unreal.NiagaraComponent)
    if not niagara_component:
        unreal.log_error("Failed to get NiagaraComponent.")
        niagara_actor.destroy_actor()  # Clean up the actor if component is not found
        return None

    niagara_component.set_asset(asset)
    if scale and isinstance(scale, (list, tuple)) and len(scale) == 3:
        niagara_actor.set_actor_scale3d(unreal.Vector(scale[0], scale[1], scale[2]))
    niagara_actor.set_actor_rotation(actor_rotation, teleport_physics=False)
    return niagara_actor


# --- New Helper Function for Fractal Pattern ---
def generate_fractal_points(start, angle, length, depth, branch_angle, branch_scale):
    """
    Recursively generates points for a simple fractal (binary tree) pattern.
    Returns a list of (x, y, z) tuples.
    """
    points = [start]
    if depth <= 0 or length < 10:
        return points
    end_x = start[0] + length * math.cos(angle)
    end_y = start[1] + length * math.sin(angle)
    end_point = (end_x, end_y, start[2])
    points.append(end_point)
    left_points = generate_fractal_points(end_point, angle + branch_angle, length * branch_scale, depth - 1, branch_angle, branch_scale)
    right_points = generate_fractal_points(end_point, angle - branch_angle, length * branch_scale, depth - 1, branch_angle, branch_scale)
    points.extend(left_points)
    points.extend(right_points)
    return points


def spawn_asset_group(group_config, asset_types_map):
    """
    Spawns a group of assets based on the placement pattern defined in group_config.
    Supported patterns include:
      - circle, scatter, small_village/large_village, three_lane_map, grid, spiral, concentric,
      - hexagonal, zigzag, cluster,
      - star, diamond, rectangle, wave,
      - arc, sector, elliptical, polygon, s_curve, fibonacci_spiral, arc_radial_fill, fractal
      - city, radial_sunburst, jittered_grid, linear_path, circular_shockwave, spiral_wave, density_map_scatter, floating, in_the_middle, triangle  (New patterns added)
    """
    pattern = group_config.get("pattern")
    asset_type = group_config.get("asset_type")
    name_contains = group_config.get("name_contains", "")
    count = group_config.get("count", 1)
    z_offset = group_config.get("z_offset", 0)
    spawned_actors = []

    if asset_type not in asset_types_map or not asset_types_map[asset_type]:
        unreal.log_error(f"No assets found for group with asset_type: {asset_type}")
        return spawned_actors

    candidate_assets = [asset for asset in asset_types_map[asset_type] if asset_matches_keywords(asset, name_contains)]
    if not candidate_assets:
        unreal.log_error(f"No assets found for {asset_type} that match keywords '{name_contains}'. Skipping group spawn.")
        return spawned_actors

    # ---------- Circle Pattern ----------
    if pattern == "circle":
        center = group_config.get("center", [0, 0, 0])
        radius = group_config.get("radius", 1500)
        angle_step = 360 / count
        for i in range(count):
            angle_deg = i * angle_step
            angle_rad = math.radians(angle_deg)
            x_offset = radius * math.cos(angle_rad)
            y_offset = radius * math.sin(angle_rad)
            spawn_location = unreal.Vector(center[0] + x_offset, center[1] + y_offset, center[2] + z_offset)
            rotation = unreal.Rotator(0, (angle_deg + 180) % 360, 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'circle'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Scatter Pattern ----------
    elif pattern == "scatter":
        area = group_config.get("area", [[0, 0, 0], [2000, 2000, 0]])
        min_corner, max_corner = area[0], area[1]
        for i in range(count):
            x = random.uniform(min_corner[0], max_corner[0])
            y = random.uniform(min_corner[1], max_corner[1])
            z = min_corner[2]
            spawn_location = unreal.Vector(x, y, z + z_offset)
            rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'scatter'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Small/Large Village (Grid-Based) ----------
    elif pattern in ["small_village", "large_village"]:
        if pattern == "small_village":
            area = group_config.get("area", [[0, 0, 0], [8000, 8000, 0]])
        else:
            area = group_config.get("area", [[-5000, -5000, 0], [8000, 8000, 0]])
        min_corner, max_corner = area[0], area[1]
        grid_cols = int(math.sqrt(count))
        grid_rows = grid_cols if grid_cols * grid_cols >= count else grid_cols + 1
        cell_width = (max_corner[0] - min_corner[0]) / grid_cols
        cell_height = (max_corner[1] - min_corner[1]) / grid_rows
        spawn_positions = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if len(spawn_positions) >= count:
                    break
                cell_center_x = min_corner[0] + (col + 0.5) * cell_width
                cell_center_y = min_corner[1] + (row + 0.5) * cell_height
                offset_x = random.uniform(-cell_width * 0.1, cell_width * 0.1)
                offset_y = random.uniform(-cell_height * 0.1, cell_height * 0.1)
                spawn_positions.append(unreal.Vector(cell_center_x + offset_x, cell_center_y + offset_y, min_corner[2] + z_offset))
        for pos in spawn_positions:
            rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern '{pattern}'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Three Lane Map (FPS Map) ----------
    elif pattern == "three_lane_map":
        lane_length = group_config.get("lane_length", 10000)
        lane_width = group_config.get("lane_width", 200)
        lane_separation = group_config.get("lane_separation", 400)
        cover_count = group_config.get("cover_count", 20)
        base_center = group_config.get("center", [0, 0, 0])
        z_offset = group_config.get("z_offset", 0)
        center_lane = unreal.Vector(base_center[0], base_center[1], base_center[2] + z_offset)
        left_lane = unreal.Vector(base_center[0] - lane_separation, base_center[1], base_center[2] + z_offset)
        right_lane = unreal.Vector(base_center[0] + lane_separation, base_center[1], base_center[2] + z_offset)
        lane_centers = [left_lane, center_lane, right_lane]
        for lane in lane_centers:
            for i in range(cover_count):
                fraction = i / (cover_count - 1) if cover_count > 1 else 0.5
                y_pos = -lane_length / 2 + fraction * lane_length
                x_offset = random.uniform(-lane_width * 0.3, lane_width * 0.3)
                spawn_location = unreal.Vector(lane.x + x_offset, lane.y + y_pos, lane.z)
                rotation = unreal.Rotator(0, random.uniform(-10, 10), 0)
                selected_asset = random.choice(candidate_assets)
                if asset_type == "StaticMesh":
                    actor = spawn_static_mesh(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
                elif asset_type == "NiagaraSystem": # Spawn Niagara System
                    actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
                else:
                    unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'three_lane_map'.")
                continue
                if actor:
                    spawned_actors.append(actor)

    # ---------- Grid Pattern ----------
    elif pattern == "grid":
        area = group_config.get("area", [[0, 0, 0], [2000, 2000, 0]])
        min_corner, max_corner = area[0], area[1]
        grid_cols = int(math.ceil(math.sqrt(count)))
        grid_rows = int(math.ceil(count / grid_cols))
        cell_width = (max_corner[0] - min_corner[0]) / grid_cols
        cell_height = (max_corner[1] - min_corner[1]) / grid_rows
        spawn_positions = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if len(spawn_positions) >= count:
                    break
                cell_center_x = min_corner[0] + (col + 0.5) * cell_width
                cell_center_y = min_corner[1] + (row + 0.5) * cell_height
                offset_x = random.uniform(-cell_width * 0.1, cell_width * 0.1)
                offset_y = random.uniform(-cell_height * 0.1, cell_height * 0.1)
                spawn_positions.append(unreal.Vector(cell_center_x + offset_x, cell_center_y + offset_y, min_corner[2] + z_offset))
        for pos in spawn_positions:
            rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'grid'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Spiral Pattern ----------
    elif pattern == "spiral":
        center = group_config.get("center", [0, 0, 0])
        spiral_a = group_config.get("spiral_a", 10)  # initial offset
        spiral_b = group_config.get("spiral_b", 20)  # spacing factor
        angle_step = group_config.get("spiral_angle_step", 0.5)  # in radians
        for i in range(count):
            theta = i * angle_step
            r = spiral_a + spiral_b * theta
            x = center[0] + r * math.cos(theta)
            y = center[1] + r * math.sin(theta)
            pos = unreal.Vector(x, y, center[2] + z_offset)
            rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'spiral'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Concentric Pattern ----------
    elif pattern == "concentric":
        center = group_config.get("center", [0, 0, 0])
        ring_count = group_config.get("ring_count", 3)
        max_radius = group_config.get("max_radius", 1500)
        spawned = 0
        for ring in range(1, ring_count + 1):
            radius = (ring / ring_count) * max_radius
            ring_assets = int(math.ceil(count / ring_count))
            angle_step = 360 / ring_assets if ring_assets > 0 else 0
            for i in range(ring_assets):
                if spawned >= count:
                    break
                angle_deg = i * angle_step
                angle_rad = math.radians(angle_deg)
                x = center[0] + radius * math.cos(angle_rad)
                y = center[1] + radius * math.sin(angle_rad)
                pos = unreal.Vector(x, y, center[2] + z_offset)
                rotation = unreal.Rotator(0, (angle_deg + 180) % 360, 0)
                selected_asset = random.choice(candidate_assets)
                if asset_type == "StaticMesh":
                    actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
                elif asset_type == "NiagaraSystem": # Spawn Niagara System
                    actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
                else:
                    unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'concentric'.")
                continue
                if actor:
                    spawned_actors.append(actor)
                    spawned += 1
            if spawned >= count:
                break

    # ---------- Hexagonal Pattern ----------
    elif pattern == "hexagonal":
        area = group_config.get("area", [[0, 0, 0], [2000, 2000, 0]])
        min_corner, max_corner = area[0], area[1]
        cell_width = group_config.get("cell_width", 200)
        cell_height = cell_width * math.sqrt(3) / 2
        area_width = max_corner[0] - min_corner[0]
        area_height = max_corner[1] - min_corner[1]
        cols = int(math.ceil(area_width / cell_width))
        rows = int(math.ceil(area_height / cell_height))
        spawn_positions = []
        for row in range(rows):
            for col in range(cols):
                if len(spawn_positions) >= count:
                    break
                x = min_corner[0] + col * cell_width
                if row % 2 == 1:
                    x += cell_width / 2
                y = min_corner[1] + row * cell_height
                spawn_positions.append(unreal.Vector(x, y, min_corner[2] + z_offset))
        spawn_positions = spawn_positions[:count]
        for pos in spawn_positions:
            rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'hexagonal'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Zigzag Pattern ----------
    elif pattern == "zigzag":
        start_point = group_config.get("start", [0, 0, 0])
        end_point = group_config.get("end", [1000, 0, 0])
        amplitude = group_config.get("zigzag_amplitude", 100)
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        dz = end_point[2] - start_point[2]
        dist = math.sqrt(dx * dx + dy * dy) or 1
        ndx, ndy = dx / dist, dy / dist
        pdx, pdy = -ndy, ndx  # perpendicular vector
        for i in range(count):
            fraction = i / (count - 1) if count > 1 else 0.5
            x = start_point[0] + dx * fraction
            y = start_point[1] + dy * fraction
            z = start_point[2] + dz * fraction
            offset = amplitude * ((-1) ** i)
            x += pdx * offset
            y += pdy * offset
            pos = unreal.Vector(x, y, z + z_offset)
            rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'zigzag'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Cluster Pattern ----------
    elif pattern == "cluster":
        center = group_config.get("center", [0, 0, 0])
        cluster_radius = group_config.get("cluster_radius", 500)
        spawn_positions = []
        attempts = 0
        while len(spawn_positions) < count and attempts < count * 10:
            angle = random.uniform(0, 2 * math.pi)
            r = cluster_radius * math.sqrt(random.uniform(0, 1))
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            pos = unreal.Vector(x, y, center[2] + z_offset)
            spawn_positions.append(pos)
            attempts += 1
        for pos in spawn_positions[:count]:
            rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'cluster'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Star Pattern ----------
    elif pattern == "star":
        center = group_config.get("center", [0, 0, 0])
        star_points = group_config.get("star_points", 5)
        outer_radius = group_config.get("outer_radius", 1500)
        inner_radius = group_config.get("inner_radius", outer_radius / 2)
        num_vertices = 2 * star_points
        for i in range(num_vertices):
            angle = i * (math.pi / star_points)
            r = outer_radius if i % 2 == 0 else inner_radius
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            pos = unreal.Vector(x, y, center[2] + z_offset)
            rotation = unreal.Rotator(0, (math.degrees(angle) + 180) % 360, 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'star'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Diamond Pattern ----------
    elif pattern == "diamond":
        center = group_config.get("center", [0, 0, 0])
        width = group_config.get("width", 1500)
        vertices = [
            (center[0], center[1] + width / 2),   # top
            (center[0] + width / 2, center[1]),     # right
            (center[0], center[1] - width / 2),       # bottom
            (center[0] - width / 2, center[1])      # left
        ]
        if count <= 4:
            for i in range(count):
                x, y = vertices[i]
                pos = unreal.Vector(x, y, center[2] + z_offset)
                rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
                selected_asset = random.choice(candidate_assets)
                if asset_type == "StaticMesh":
                    actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
                elif asset_type == "NiagaraSystem": # Spawn Niagara System
                    actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
                else:
                    unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'diamond'.")
                continue
                if actor:
                    spawned_actors.append(actor)
        else:
            per_edge = count // 4
            remainder = count % 4
            for edge in range(4):
                start = vertices[edge]
                end = vertices[(edge + 1) % 4]
                num_points = per_edge + (1 if edge < remainder else 0)
                for i in range(num_points):
                    t = i / (num_points - 1) if num_points > 1 else 0.5
                    x = start[0] + t * (end[0] - start[0])
                    y = start[1] + t * (end[1] - start[1])
                    pos = unreal.Vector(x, y, center[2] + z_offset)
                    rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
                    selected_asset = random.choice(candidate_assets)
                    if asset_type == "StaticMesh":
                        actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
                    elif asset_type == "NiagaraSystem": # Spawn Niagara System
                        actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
                    else:
                        unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'diamond'.")
                        continue
                    if actor:
                        spawned_actors.append(actor)

    # ---------- Rectangle Pattern ----------
    elif pattern == "rectangle":
        center = group_config.get("center", [0, 0, 0])
        width = group_config.get("width", 2000)
        height = group_config.get("height", 1000)
        half_w = width / 2
        half_h = height / 2
        vertices = [
            (center[0] - half_w, center[1] + half_h),  # top-left
            (center[0] + half_w, center[1] + half_h),  # top-right
            (center[0] + half_w, center[1] - half_h),  # bottom-right
            (center[0] - half_w, center[1] - half_h)   # bottom-left
        ]
        if count <= 4:
            for i in range(count):
                x, y = vertices[i]
                pos = unreal.Vector(x, y, center[2] + z_offset)
                rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
                selected_asset = random.choice(candidate_assets)
                if asset_type == "StaticMesh":
                    actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
                elif asset_type == "NiagaraSystem": # Spawn Niagara System
                    actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
                else:
                    unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'rectangle'.")
                continue
                if actor:
                    spawned_actors.append(actor)
        else:
            per_edge = count // 4
            remainder = count % 4
            for edge in range(4):
                start = vertices[edge]
                end = vertices[(edge + 1) % 4]
                num_points = per_edge + (1 if edge < remainder else 0)
                for i in range(num_points):
                    t = i / (num_points - 1) if num_points > 1 else 0.5
                    x = start[0] + t * (end[0] - start[0])
                    y = start[1] + t * (end[1] - start[1])
                    pos = unreal.Vector(x, y, center[2] + z_offset)
                    rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
                    selected_asset = random.choice(candidate_assets)
                    if asset_type == "StaticMesh":
                        actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
                    elif asset_type == "NiagaraSystem": # Spawn Niagara System
                        actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
                    else:
                        unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'rectangle'.")
                        continue
                    if actor:
                        spawned_actors.append(actor)

    # ---------- Wave Pattern ----------
    elif pattern == "wave":
        start_point = group_config.get("start", [0, 0, 0])
        end_point = group_config.get("end", [1000, 0, 0])
        amplitude = group_config.get("amplitude", 100)
        frequency = group_config.get("frequency", 1)
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        dz = end_point[2] - start_point[2]
        length = math.sqrt(dx * dx + dy * dy) or 1
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.5
            # Base point along the line.
            x_line = start_point[0] + t * dx
            y_line = start_point[1] + t * dy
            z_line = start_point[2] + t * dz
            # Perpendicular direction.
            perp_x = -dy
            perp_y = dx
            norm = math.sqrt(perp_x ** 2 + perp_y ** 2) or 1
            perp_x /= norm
            perp_y /= norm
            offset = amplitude * math.sin(2 * math.pi * frequency * t)
            x = x_line + perp_x * offset
            y = y_line + perp_y * offset
            pos = unreal.Vector(x, y, z_line + z_offset)
            rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'wave'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Arc Pattern ----------
    elif pattern == "arc":
        center = group_config.get("center", [0, 0, 0])
        radius = group_config.get("radius", 1500)
        start_angle = math.radians(group_config.get("start_angle", 0))
        end_angle = math.radians(group_config.get("end_angle", 90))
        angle_step = (end_angle - start_angle) / (count - 1) if count > 1 else 0
        for i in range(count):
            angle = start_angle + i * angle_step
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            pos = unreal.Vector(x, y, center[2] + z_offset)
            rotation = unreal.Rotator(0, math.degrees(angle) + 180, 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'arc'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Sector Pattern (Fill) ----------
    elif pattern == "sector":
        center = group_config.get("center", [0, 0, 0])
        radius = group_config.get("radius", 1500)
        start_angle = math.radians(group_config.get("start_angle", 0))
        end_angle = math.radians(group_config.get("end_angle", 90))
        for i in range(count):
            angle = random.uniform(start_angle, end_angle)
            r = random.uniform(0, radius)
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            pos = unreal.Vector(x, y, center[2] + z_offset)
            rotation = unreal.Rotator(0, math.degrees(angle) + 180, 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'sector'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Elliptical Pattern ----------
    elif pattern == "elliptical":
        center = group_config.get("center", [0, 0, 0])
        radius_x = group_config.get("radius_x", 1500)
        radius_y = group_config.get("radius_y", 1000)
        for i in range(count):
            angle = 2 * math.pi * (i / count)
            x = center[0] + radius_x * math.cos(angle)
            y = center[1] + radius_y * math.sin(angle)
            pos = unreal.Vector(x, y, center[2] + z_offset)
            rotation = unreal.Rotator(0, math.degrees(angle) + 180, 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'elliptical'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Polygon Pattern ----------
    elif pattern == "polygon":
        center = group_config.get("center", [0, 0, 0])
        sides = group_config.get("sides", 6)
        radius = group_config.get("radius", 1500)
        vertices = []
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            vertices.append((x, y))
        if count <= sides:
            for i in range(count):
                x, y = vertices[i]
                pos = unreal.Vector(x, y, center[2] + z_offset)
                rotation = unreal.Rotator(0, math.degrees(2 * math.pi * i / sides) + 180, 0)
                selected_asset = random.choice(candidate_assets)
                if asset_type == "StaticMesh":
                    actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
                elif asset_type == "NiagaraSystem": # Spawn Niagara System
                    actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
                else:
                    unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'polygon'.")
                continue
                if actor:
                    spawned_actors.append(actor)
        else:
            per_edge = count // sides
            remainder = count % sides
            for edge in range(4):
                start = vertices[edge]
                end = vertices[(edge + 1) % sides]
                num_points = per_edge + (1 if edge < remainder else 0)
                for i in range(num_points):
                    t = i / (num_points - 1) if num_points > 1 else 0.5
                    x = start[0] + t * (end[0] - start[0])
                    y = start[1] + t * (end[1] - start[1])
                    pos = unreal.Vector(x, y, center[2] + z_offset)
                    rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
                    selected_asset = random.choice(candidate_assets)
                    if asset_type == "StaticMesh":
                        actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
                    elif asset_type == "NiagaraSystem": # Spawn Niagara System
                        actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
                    else:
                        unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'polygon'.")
                        continue
                    if actor:
                        spawned_actors.append(actor)

    # ---------- S-Curve Pattern (Cubic Bézier) ----------
    elif pattern == "s_curve":
        start = group_config.get("start", [0, 0, 0])
        control1 = group_config.get("control1", [250, 500, 0])
        control2 = group_config.get("control2", [750, -500, 0])
        end = group_config.get("end", [1000, 0, 0])
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.5
            x = ((1-t)**3 * start[0] +
                 3*(1-t)**2 * t * control1[0] +
                 3*(1-t) * t**2 * control2[0] +
                 t**3 * end[0])
            y = ((1-t)**3 * start[1] +
                 3*(1-t)**2 * t * control1[1] +
                 3*(1-t) * t**2 * control2[1] +
                 t**3 * end[1])
            z = ((1-t)**3 * start[2] +
                 3*(1-t)**2 * t * control1[2] +
                 3*(1-t) * t**2 * control2[2] +
                 t**3 * end[2])
            pos = unreal.Vector(x, y, z + z_offset)
            rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 's_curve'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Fibonacci Spiral Pattern ----------
    elif pattern == "fibonacci_spiral":
        center = group_config.get("center", [0, 0, 0])
        a = group_config.get("a", 5)  # scaling factor
        phi = 1.61803398875
        b = math.log(phi) / (math.pi/2)
        angle_step = group_config.get("angle_step", 0.5)
        for i in range(count):
            theta = i * angle_step
            r = a * math.exp(b * theta)
            x = center[0] + r * math.cos(theta)
            y = center[1] + r * math.sin(theta)
            pos = unreal.Vector(x, y, center[2] + z_offset)
            rotation = unreal.Rotator(0, math.degrees(theta) + 180, 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'fibonacci_spiral'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Arc-Radial Fill Pattern ----------
    elif pattern == "arc_radial_fill":
        center = group_config.get("center", [0, 0, 0])
        min_radius = group_config.get("min_radius", 500)
        max_radius = group_config.get("max_radius", 1500)
        start_angle = math.radians(group_config.get("start_angle", 0))
        end_angle = math.radians(group_config.get("end_angle", 90))
        for i in range(count):
            angle = random.uniform(start_angle, end_angle)
            r = random.uniform(min_radius, max_radius)
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            pos = unreal.Vector(x, y, center[2] + z_offset)
            rotation = unreal.Rotator(0, math.degrees(angle) + 180, 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'arc_radial_fill'.")
                continue
            if actor:
                spawned_actors.append(actor)

    # ---------- Fractal Pattern (Simple Binary Branching) ----------
    elif pattern == "fractal":
        start = group_config.get("start", [0, 0, 0])
        angle = math.radians(group_config.get("angle", 90))
        length = group_config.get("length", 300)
        depth = group_config.get("depth", 3)
        branch_angle = math.radians(group_config.get("branch_angle", 30))
        branch_scale = group_config.get("branch_scale", 0.7)
        points = generate_fractal_points(start, angle, length, depth, branch_angle, branch_scale)
        unique_points = []
        for pt in points:
            if pt not in unique_points:
                unique_points.append(pt)
            if len(unique_points) >= count:
                break
        for pt in unique_points[:count]:
            pos = unreal.Vector(pt[0], pt[1], pt[2] + z_offset)
            rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'fractal'.")
                continue
            if actor:
                spawned_actors.append(actor)

    elif pattern == "city":
        area = group_config.get("area", [[0, 0, 0], [10000, 10000, 0]])
        road_width = group_config.get("road_width", 200)
        block_size = group_config.get("block_size", 800)
        buildings_per_block = group_config.get("buildings_per_block", 2)
        min_corner, max_corner = area[0], area[1]
        area_width = max_corner[0] - min_corner[0]
        area_height = max_corner[1] - min_corner[1]
        num_cols = int(area_width // (block_size + road_width))
        num_rows = int(area_height // (block_size + road_width))
        for row in range(num_rows):
            for col in range(num_cols):
                block_origin_x = min_corner[0] + col * (block_size + road_width) + road_width / 2
                block_origin_y = min_corner[1] + row * (block_size + road_width) + road_width / 2
                for i in range(buildings_per_block):
                    x = random.uniform(block_origin_x, block_origin_x + block_size)
                    y = random.uniform(block_origin_y, block_origin_y + block_size)
                    pos = unreal.Vector(x, y, min_corner[2] + z_offset)
                    rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
                    selected_asset = random.choice(candidate_assets)
                    if asset_type == "StaticMesh":
                        actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
                    elif asset_type == "NiagaraSystem": # Spawn Niagara System
                        actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
                    else:
                        unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'city'.")
                        continue
                    if actor:
                        spawned_actors.append(actor)

    # ---------- Radial Sunburst Pattern ----------
    elif pattern == "radial_sunburst":
        center = group_config.get("center", [0, 0, 0])
        ray_count = group_config.get("ray_count", 8)
        num_assets_per_ray = group_config.get("num_assets_per_ray", 5)
        spawned = []
        for i in range(ray_count):
            angle = 2 * math.pi * i / ray_count
            for j in range(num_assets_per_ray):
                r = group_config.get("min_radius", 500) + j * group_config.get("ray_spacing", 200)
                x = center[0] + r * math.cos(angle)
                y = center[1] + r * math.sin(angle)
                pos = unreal.Vector(x, y, center[2] + z_offset)
                rotation = unreal.Rotator(0, math.degrees(angle) + 180, 0)
                selected_asset = random.choice(candidate_assets)
                if asset_type == "StaticMesh":
                    actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
                elif asset_type == "NiagaraSystem": # Spawn Niagara System
                    actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
                else:
                    unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'radial_sunburst'.")
                continue
                if actor:
                    spawned.append(actor)

    # ---------- Jittered Grid (Density-Based Random Scatter) ----------
    elif pattern == "jittered_grid":
        area = group_config.get("area", [[0, 0, 0], [2000, 2000, 0]])
        min_corner, max_corner = area[0], area[1]
        grid_cols = group_config.get("grid_cols", 10)
        grid_rows = group_config.get("grid_rows", 10)
        cell_width = (max_corner[0] - min_corner[0]) / grid_cols
        cell_height = (max_corner[1] - min_corner[1]) / grid_rows
        spawned = []
        jitter = group_config.get("jitter", 0.2)  # fraction of cell dimension
        for row in range(grid_rows):
            for col in range(grid_cols):
                x_center = min_corner[0] + (col + 0.5) * cell_width
                y_center = min_corner[1] + (row + 0.5) * cell_height
                x_offset = random.uniform(-jitter * cell_width, jitter * cell_width)
                y_offset = random.uniform(-jitter * cell_height, jitter * cell_height)
                pos = unreal.Vector(x_center + x_offset, y_center + y_offset, min_corner[2] + z_offset)
                rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
                selected_asset = random.choice(candidate_assets)
                if asset_type == "StaticMesh":
                    actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
                elif asset_type == "NiagaraSystem": # Spawn Niagara System
                    actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
                else:
                    unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'jittered_grid'.")
                continue
                if actor:
                    spawned.append(actor)

    # ---------- Linear Path / Road Pattern ----------
    elif pattern == "linear_path":
        start = group_config.get("start", [0, 0, 0])
        end = group_config.get("end", [1000, 0, 0])
        spawned = []
        lateral_offset = group_config.get("lateral_offset", 50)
        dx = end[0] - start[0]
        dy = end[1] - start_point[1]
        length = math.sqrt(dx*dx + dy*dy) or 1
        # Perpendicular direction:
        perp_x, perp_y = -dy/length, dx/length
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.5
            x = start[0] + t * dx
            y = start[1] + t * dy
            offset = random.uniform(-lateral_offset, lateral_offset)
            pos = unreal.Vector(x + perp_x * offset, y + perp_y * offset, start[2] + t * (end[2]-start[2]) + z_offset)
            rotation = unreal.Rotator(0, math.degrees(math.atan2(dy, dx)), 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'linear_path'.")
                continue
            if actor:
                spawned.append(actor)

    # ---------- Circular Shockwave Pattern ----------
    elif pattern == "circular_shockwave":
        center = group_config.get("center", [0, 0, 0])
        base_radius = group_config.get("base_radius", 500)
        spacing = group_config.get("spacing", 200)
        spawned = []
        for i in range(count):
            radius = base_radius + i * spacing
            angle = random.uniform(0, 2*math.pi)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            pos = unreal.Vector(x, y, center[2] + z_offset)
            rotation = unreal.Rotator(0, math.degrees(angle) + 180, 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'circular_shockwave'.")
                continue
            if actor:
                spawned.append(actor)

    # ---------- Spiral Wave Pattern ----------
    elif pattern == "spiral_wave":
        center = group_config.get("center", [0, 0, 0])
        spiral_a = group_config.get("spiral_a", 10)
        spiral_b = group_config.get("spiral_b", 20)
        wave_amplitude = group_config.get("wave_amplitude", 50)
        wave_frequency = group_config.get("wave_frequency", 1)
        spawned = []
        for i in range(count):
            theta = i * group_config.get("spiral_angle_step", 0.5)
            r = spiral_a + spiral_b * theta + wave_amplitude * math.sin(wave_frequency * theta)
            x = center[0] + r * math.cos(theta)
            y = center[1] + r * math.sin(theta)
            pos = unreal.Vector(x, y, center[2] + z_offset)
            rotation = unreal.Rotator(0, math.degrees(theta) + 180, 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'spiral_wave'.")
                continue
            if actor:
                spawned.append(actor)

    # ---------- Density Map Driven Scatter Pattern ----------
    elif pattern == "density_map_scatter":
        area = group_config.get("area", [[0, 0, 0], [2000, 2000, 0]])
        min_corner, max_corner = area[0], area[1]
        spawned = []
        # Generate extra candidate points and filter by a density threshold.
        for i in range(count * 2):
            x = random.uniform(min_corner[0], max_corner[0])
            y = random.uniform(min_corner[1], max_corner[1])
            # For demonstration, we use a random value as "density" (in practice, use Perlin noise or similar)
            density = random.random()
            if density > group_config.get("density_threshold", 0.5):
                pos = unreal.Vector(x, y, min_corner[2] + z_offset)
                rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
                selected_asset = random.choice(candidate_assets)
                if asset_type == "StaticMesh":
                    actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
                elif asset_type == "NiagaraSystem": # Spawn Niagara System
                    actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
                else:
                    unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'density_map_scatter'.")
                continue
                if actor:
                    spawned.append(actor)
            if len(spawned) >= count:
                break

    # ---------- Floating Pattern (3D Orbital/Floating Placement) ----------
    elif pattern == "floating":
        center = group_config.get("center", [0, 0, 0])
        min_radius = group_config.get("min_radius", 1000)
        max_radius = group_config.get("max_radius", 2000)
        spawned = []
        for i in range(count):
            r = random.uniform(min_radius, max_radius)
            theta = random.uniform(0, math.pi)
            phi = random.uniform(0, 2 * math.pi)
            x = center[0] + r * math.sin(theta) * math.cos(phi)
            y = center[1] + r * math.sin(theta) * math.sin(phi)
            z = center[2] + r * math.cos(theta)
            pos = unreal.Vector(x, y, z + z_offset)
            rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'floating'.")
                continue
            if actor:
                spawned.append(actor)

    # ---------- In The Middle Pattern ----------
    elif pattern == "in_the_middle":
        center = group_config.get("center")
        area = group_config.get("area")

        if center:
            spawn_center = unreal.Vector(center[0], center[1], center[2] + z_offset)
        elif area:
            min_corner, max_corner = area[0], area[1]
            center_x = (min_corner[0] + max_corner[0]) / 2
            center_y = (min_corner[1] + max_corner[1]) / 2
            center_z = (min_corner[2] + max_corner[2]) / 2  # Consider using just min_corner[2] + z_offset if z_offset is relative to the base Z.
            spawn_center = unreal.Vector(center_x, center_y, center_z + z_offset)
        else:
            unreal.log_warning("In 'in_the_middle' pattern, neither 'center' nor 'area' provided. Defaulting to origin [0,0,0].")
            spawn_center = unreal.Vector(0, 0, z_offset)

        for i in range(count):
            spawn_location = spawn_center # Initial location

            if count > 1: # Add slight scatter if count > 1
                scatter_radius = group_config.get("scatter_radius", 50) # Configurable scatter radius
                angle = random.uniform(0, 2 * math.pi)
                radius = scatter_radius * math.sqrt(random.random()) # Uniform disk distribution
                x_offset = radius * math.cos(angle)
                y_offset = radius * math.sin(angle)
                spawn_location = unreal.Vector(spawn_center.x + x_offset, spawn_center.y + y_offset, spawn_center.z)

            rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
            selected_asset = random.choice(candidate_assets)
            if asset_type == "StaticMesh":
                actor = spawn_static_mesh(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
            elif asset_type == "NiagaraSystem": # Spawn Niagara System
                actor = spawn_niagara_system(selected_asset, spawn_location, rotation, scale=[1, 1, 1])
            else:
                unreal.log_error(f"Unsupported asset type '{asset_type}' in group pattern 'in_the_middle'.")
                continue
            if actor:
                spawned_actors.append(actor)


    # ---------- Triangle Pattern ----------
    elif pattern == "triangle":
        vertices_config = group_config.get("vertices")
        if not vertices_config or len(vertices_config) != 3:
            unreal.log_error("Triangle pattern requires 'vertices' with exactly 3 points defined.")
            return spawned_actors

        vertices = [unreal.Vector(v[0], v[1], v[2] + z_offset) for v in vertices_config] # Apply z_offset to vertices
        distribution = group_config.get("distribution", "vertices") # Default to vertices distribution

        if distribution == "vertices":
            spawn_positions = vertices[:min(count, 3)] # Limit to 3 vertices max
            for pos in spawn_positions:
                rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
                selected_asset = random.choice(candidate_assets)
                if asset_type == "StaticMesh":
                    actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
                elif asset_type == "NiagaraSystem": # Spawn Niagara System
                    actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
                else:
                    unreal.log_error(f"Unsupported asset type '{asset_type}' in triangle pattern with 'vertices' distribution.")
                    continue
                if actor:
                    spawned_actors.append(actor)

        elif distribution == "edges":
            num_edges = 3
            points_per_edge = count // num_edges
            remainder = count % num_edges

            for edge_index in range(num_edges):
                start_point = vertices[edge_index]
                end_point = vertices[(edge_index + 1) % num_edges]
                num_points_on_edge = points_per_edge + (1 if edge_index < remainder else 0)

                for i in range(num_points_on_edge):
                    t = i / (num_points_on_edge - 1) if num_points_on_edge > 1 else 0.5 # Avoid division by zero
                    pos = start_point + (end_point - start_point) * t
                    rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
                    selected_asset = random.choice(candidate_assets)
                    if asset_type == "StaticMesh":
                        actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
                    elif asset_type == "NiagaraSystem": # Spawn Niagara System
                        actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
                    else:
                        unreal.log_error(f"Unsupported asset type '{asset_type}' in triangle pattern with 'edges' distribution.")
                        continue
                    if actor:
                        spawned_actors.append(actor)

        elif distribution in ["fill", "scatter"]: # "fill" and "scatter" distributions
            min_x = min(v.x for v in vertices)
            max_x = max(v.x for v in vertices)
            min_y = min(v.y for v in vertices)
            max_y = max(v.y for v in vertices)

            spawn_positions = []
            attempts = 0
            while len(spawn_positions) < count and attempts < count * 10: # Limit attempts to avoid infinite loops
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                point = unreal.Vector(x, y, vertices[0].z) # Z is assumed to be roughly the same for all vertices, using vertices[0].z
                # Check if point is inside the triangle using barycentric coordinates (or simpler point-in-triangle test)
                v0, v1, v2 = vertices[0], vertices[1], vertices[2]
                def is_point_in_triangle(p, a, b, c): # Point-in-triangle test (using barycentric coordinates implicitly)
                    s = (a.y * c.x - a.x * c.y + (c.y - a.y) * p.x + (a.x - c.x) * p.y) < 0;
                    if(s != ((a.y * b.x - a.x * b.y + (b.y - a.y) * p.x + (a.x - b.x) * p.y) < 0)): return False;
                    if(s == ((b.y * c.x - b.x * c.y + (c.y - b.y) * p.x + (b.x - c.x) * p.y) < 0)): return False;
                    return True;

                if is_point_in_triangle(point, v0, v1, v2):
                    spawn_positions.append(point)
                attempts += 1

            for pos in spawn_positions[:count]: # Limit to requested count
                rotation = unreal.Rotator(0, random.uniform(0, 360), 0)
                selected_asset = random.choice(candidate_assets)
                if asset_type == "StaticMesh":
                    actor = spawn_static_mesh(selected_asset, pos, rotation, scale=[1, 1, 1])
                elif asset_type == "NiagaraSystem": # Spawn Niagara System
                    actor = spawn_niagara_system(selected_asset, pos, rotation, scale=[1, 1, 1])
                else:
                    unreal.log_error(f"Unsupported asset type '{asset_type}' in triangle pattern with '{distribution}' distribution.")
                    continue
                if actor:
                    spawned_actors.append(actor)

        else:
            unreal.log_error(f"Unsupported distribution type '{distribution}' for triangle pattern. Valid types: vertices, edges, fill, scatter.")

    else:
        unreal.log_error(f"Unsupported group pattern: {pattern}")

    return spawned_actors


def generate_assets_with_ai(prompt, environment_scale=1.0):
    """Generates assets in the level based on an AI prompt with dynamic placement instructions."""
    if not prompt:
        unreal.log_error("Error: No prompt was specified")
        return

    # ---- Gemini API Setup ----
    gemini_api_key = "YOUR-API-KEY"  # Store securely if possible.
    if not gemini_api_key:
        unreal.log_error("GEMINI_API_KEY is not set. Set it!")
        return

    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        unreal.log_error(f"Error initializing Gemini model: {e}")
        return

    unreal.log(f"Starting asset generation with prompt: {prompt}")

    gemini_prompt = f'''Given the prompt: "{prompt}", generate a JSON configuration for placing assets directly in the level with dynamic spatial arrangements.

The JSON should have the following structure:

{{
    "assets": [
        {{
            "type": "StaticMesh",
            "name": "Lake_01",
            "location": [1000, 950, 0],
            "rotation": [0, 0, 0],
            "scale": [1, 1, 1]
        }},
        {{
            "type": "NiagaraSystem",
            "name": "Fireflies_System_V01",
            "location": [500, 500, 100],
            "rotation": [0, 0, 0],
            "scale": [1, 1, 1]
        }}
    ],
    "groups": [
         {{
            "pattern": "circle",
            "center": [1000, 950, 0],
            "radius": 500,
            "asset_type": "StaticMesh",
            "name_contains": "Tree",
            "count": 12,
            "z_offset": 100,
            "target_water_body": "WaterBodyLake"
         }},
         {{
            "pattern": "scatter",
            "area": [[-500, -500, 0], [500, 500, 0]],
            "asset_type": "StaticMesh",
            "name_contains": "Grass",
            "count": 50,
            "z_offset": 0
         }},
         {{
            "pattern": "scatter",
            "area": [[-500, -500, 100]],
            "asset_type": "NiagaraSystem",
            "name_contains": "Fireflies",
            "count": 30,
            "z_offset": 50
         }},
         {{
            "pattern": "small_village",
            "area": [[0, 0, 0], [8000, 8000, 0]],
            "asset_type": "StaticMesh",
            "name_contains": "House",
            "count": 20,
            "z_offset": 0
         }},
         {{
            "pattern": "in_the_middle",
            "area": [[-1000, -1000, 0], [1000, 1000, 0]],
            "asset_type": "StaticMesh",
            "name_contains": "Rock",
            "count": 3,
            "scatter_radius": 100,
            "z_offset": 0
         }},
         {{
            "pattern": "triangle",
            "asset_type": "StaticMesh",
            "name_contains": "Tree",
            "count": 5,
            "vertices": [[-500, 0, 0], [500, 0, 0], [0, 866, 0]],
            "distribution": "edges",
            "z_offset": 0
         }},
         # ... rest of the group patterns ...
    ],
    "water_bodies": [
        {{
            "type": "WaterBodyLake",
            "location": [1000, 1000, 0],
            "size": [500, 500]
        }},
        # ... water body definitions ...
    ]
}}

Additional requirements:
- Ensure assets are arranged naturally and do not overlap significantly (except for grass).
- For village patterns, distribute houses evenly with space for roads or communal areas.
- For "three_lane_map", generate three parallel lanes with cover placement.
- The generated JSON must be valid and include only the assets and groups relevant to the prompt.
- When searching for Niagara Systems like 'fireflies', ensure to only match systems specifically named 'fireflies' or very close synonyms, avoiding systems with just 'fire' in the name. Prioritize exact keyword matches.
'''

    # Initialize ai_config before using it.
    ai_config = {}
    try:
        gemini_response = model.generate_content(
            gemini_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                temperature=GEMINI_TEMPERATURE
            )
        ).text or ""
        if not gemini_response:
            unreal.log_error("Gemini returned an empty response.")
            return
        gemini_response = gemini_response.strip().removeprefix("```json").strip()
        try:
            last_brace_index = gemini_response.rindex('}')
            gemini_response = gemini_response[:last_brace_index + 1]
        except ValueError:
            unreal.log_error("Invalid JSON format from Gemini: Missing closing curly brace.")
            return
        unreal.log(f"Gemini Response: {gemini_response}")
        print("Gemini Response:", gemini_response)
    except Exception as e:
        unreal.log_error(f"Gemini request error: {e}")
        return

    try:
        ai_config = json.loads(gemini_response)
        if "assets" not in ai_config:
            unreal.log_error(f"Gemini response missing assets: {gemini_response}")
            return
    except json.JSONDecodeError as e:
        unreal.log_error(f"Gemini response is not valid JSON: {e}, Response: {gemini_response}")
        return

    # Collect unique asset types from both assets and groups using a set.
    asset_types_set = set()
    for asset_data in ai_config.get("assets", []):
        if "type" in asset_data:
            asset_types_set.add(asset_data.get("type"))
    for group in ai_config.get("groups", []):
        if "asset_type" in group:
            asset_types_set.add(group.get("asset_type"))
    asset_types_map = find_assets_by_type(list(asset_types_set))
    unreal.log(f"Asset types map: {asset_types_map}")

    water_asset_actors = {}
    for asset_data in ai_config.get("assets", []):
        asset_type = asset_data.get("type")
        asset_name = asset_data.get("name")
        location = asset_data.get("location")
        rotation = asset_data.get("rotation")
        scale = asset_data.get("scale")
        if asset_type in asset_types_map and asset_types_map[asset_type]:
            candidate_assets = [asset for asset in asset_types_map[asset_type] if asset_matches_keywords(asset, asset_name)]
            if not candidate_assets:
                unreal.log_error(f"No matching asset found for '{asset_name}' in {asset_type}. Skipping asset spawn.")
                continue
            found_asset = random.choice(candidate_assets)
            if found_asset: # Check if asset is found (could be StaticMesh or NiagaraSystem)
                if asset_type == "StaticMesh" and isinstance(found_asset, unreal.StaticMesh):
                    spawned_actor = spawn_static_mesh(found_asset, unreal.Vector(location[0], location[1], location[2]), rotation, [scale[0], scale[1], scale[2]])
                elif asset_type == "NiagaraSystem" and isinstance(found_asset, unreal.NiagaraSystem):
                    spawned_actor = spawn_niagara_system(found_asset, unreal.Vector(location[0], location[1], location[2]), rotation, [scale[0], scale[1], scale[2]])
                else:
                    unreal.log_error(f"Invalid asset type or asset mismatch for: {asset_name}, type: {asset_type}")
                    continue

                lower_name = asset_name.lower()
                if "lake" in lower_name:
                    water_asset_actors["lake"] = spawned_actor
                elif "river" in lower_name:
                    water_asset_actors["river"] = spawned_actor
                elif "ocean" in lower_name:
                    water_asset_actors["ocean"] = spawned_actor
                elif "island" in lower_name:
                    water_asset_actors["island"] = spawned_actor
            else:
                unreal.log_error(f"Invalid asset type or asset not found for: {asset_name}")
        else:
            unreal.log_error(f"Asset type {asset_type} not found in asset_types_map.")

    water_body_actors = handle_water_bodies(ai_config)
    water_all = water_asset_actors.copy()
    water_all.update({
        "lake": water_body_actors.get("WaterBodyLake", None),
        "river": water_body_actors.get("WaterBodyRiver", None),
        "ocean": water_body_actors.get("WaterBodyOcean", None),
        "island": water_body_actors.get("WaterBodyIsland", None)
    })

    for group_config in ai_config.get("groups", []):
        target = group_config.get("target_water_body")
        if target:
            key = ""
            lower_target = target.lower()
            if "lake" in lower_target:
                key = "lake"
            elif "river" in lower_target:
                key = "river"
            elif "ocean" in lower_target:
                key = "ocean"
            elif "island" in lower_target:
                key = "island"
            if key and key in water_all and water_all[key]:
                loc = water_all[key].get_actor_location()
                group_config["center"] = [loc.x, loc.y, loc.z]
                unreal.log(f"Updated group center for target '{target}' to: {group_config['center']}")
            else:
                unreal.log_warning(f"Target water body '{target}' not found among spawned water bodies.")
        else:
            if "tree" in group_config.get("name_contains", "").lower() and "lake" in water_all and water_all["lake"]:
                loc = water_all["lake"].get_actor_location()
                group_config["center"] = [loc.x, loc.y, loc.z]
                unreal.log(f"Updated tree group center to lake asset's center: {group_config['center']}")

    for group_config in ai_config.get("groups", []):
        spawn_asset_group(group_config, asset_types_map)

    unreal.log_warning("Advanced asset placement complete. Note: Large numbers of assets or complex placements may impact performance.")


# --- Execution ---
generate_assets_with_ai(prompt, 3.0)