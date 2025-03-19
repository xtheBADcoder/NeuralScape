# NeuralScape
AI driven 3D environment creation for Unreal Engine
# Getting Started:
- Step 1: Enable the the following Plugins in your project - Python Editor Script Plugin, Python Foundation Packages, Editor Scripting Utilities, Geometry Script and Water.
- Step 2: Install Google Gemini(pip Install google-genai) into your Unreal python environment. Watch video if you need help doing so... https://www.youtube.com/watch?v=Qt0AW08REKg&list=WL&index=1&ab_channel=MattLakeTA
- Step 3: Go to https://aistudio.google.com/app/apikey and create and copy your own api key.
- Step 4: Download and open the ScriptAI.py file from the repo. Add your api key into the line gemini_api_key = "YOUR-API-KEY" and then copy the script.
- Step 5: Download and add the AI_PCG.uasset file from the repo into your Unreal project.
- Step 6: Open the AI_PCG Utility Widget in your project and go to the Event Graph. Paste your script into the Execute Python Script node. Compile and save.
- Step 7: Now run the Widget, type a prompt and hit enter to submit it.
# How it works:
- Upon submitting a prompt the script searches through the assets in your project and picks and adds them dynamically into the level based on whatever your prompt was. Make sure your assets are appropriately named and make sure your prompts are as specific as possible for the script to work properly. The script supports adding Static Meshes and Niagara Systems. You can also control the area covered affected by adding an environmental scale to your prompt.
# Whats Possible:
- Basic Shapes & Distributions:
- circle: Assets placed in a circular formation.
- scatter: Randomly distributed assets within a defined area.
- grid: Assets arranged in a regular grid pattern.
- hexagonal: Assets in a hexagonal grid layout.
- cluster: Assets grouped tightly around a central point.
- rectangle: Assets placed along the edges of a rectangle.
- diamond: Assets placed along the edges of a diamond shape.
- polygon: Assets placed along the edges of a polygon (number of sides configurable).
- triangle: Assets placed along the vertices, edges, or filled area of a triangle.
- concentric: Assets arranged in concentric rings.
- jittered_grid: A grid pattern with randomized position jitter for a more natural look.
- density_map_scatter: Scatter pattern influenced by a density map (currently uses a random density for demonstration but designed for real density maps).
- Linear & Curvilinear Patterns:
- linear_path: Assets placed along a straight line, with optional lateral offset.
- zigzag: Assets placed in a zigzag line pattern.
- wave: Assets arranged in a sinusoidal wave pattern along a line.
- arc: Assets placed along an arc segment.
- s_curve: Assets placed along an S-shaped curve (cubic Bézier).
- spiral: Assets arranged in a spiral shape.
- fibonacci_spiral: Assets in a Fibonacci spiral pattern.
- Radial & Outward Patterns:
- sector: Assets scattered within a sector of a circle (pie slice).
- arc_radial_fill: Assets filling a sector of a circle radially.
- radial_sunburst: Assets radiating outwards from a center point like sun rays.
- circular_shockwave: Assets arranged in expanding concentric circles, like a shockwave.
- spiral_wave: A spiral pattern with a wave-like radial displacement.
- Village/City & Thematic Patterns:
- small_village: Grid-based distribution suitable for a small village layout.
- large_village: Grid-based distribution for a larger village or town.
- three_lane_map: Pattern for creating a three-lane FPS map layout with cover elements.
- city: Grid-based pattern suitable for city block generation.
- floating: Assets placed in a 3D spherical distribution, simulating floating or orbital placement.
- in_the_middle: Places assets at the center of a defined area or a specific center point, with optional scatter.
- Fractal Pattern:
- fractal: Simple binary tree fractal pattern for tree-like distributions or abstract shapes.

# Environment Types Potentially Creatable:
- Based on the patterns and the asset types the script handles (Static Meshes, Niagara Systems, and Water Bodies), you could potentially create a wide range of environment types. Here are some examples, keeping in mind that the specific assets available in your Unreal project will heavily influence the final look:
Natural Landscapes:
- Forests/Woods: Using scatter, cluster, grid, or even fractal patterns for trees. You can control density with pattern parameters.
- Grasslands/Fields: scatter for grass assets over a large area.
- Rocky Terrains: scatter or cluster for rocks and boulders.
- Water Bodies: Lakes, oceans, rivers, and islands using the water_bodies configuration and potentially surrounding them with assets (e.g., trees around a lake using circle pattern targeted at the lake's location).
- Beaches/Coastlines: Combining water bodies (ocean) with scatter patterns for sand, rocks, and beach vegetation.
- Mountainous Regions: Using terrain sculpting alongside asset placement for rocks, trees, and potentially Niagara systems for mist or snow effects.
- Urban/Civilized Environments:
- Villages/Towns: small_village, large_village, and potentially combinations of grid and linear_path patterns for houses, roads, and other village elements.
- Cities: city pattern to generate city block layouts, populated with buildings.
- Road Networks: linear_path pattern to create roads and pathways, potentially combined with scatter for roadside details (trees, rocks, etc.).
- Parks/Gardens: grid, circle, scatter, or concentric patterns for trees, bushes, flowers, and park furniture.
- Thematic/Stylized Environments:
- Magical/Fantasy Landscapes: Using Niagara systems for particle effects (fireflies, magical glows), combined with stylized static meshes for trees, rocks, and structures. floating pattern could be used for -floating islands or magical objects.
- Sci-Fi/Futuristic Environments: Using metallic or futuristic static meshes, combined with Niagara systems for energy effects, lights, and holographic elements. radial_sunburst, circular_shockwave, or - - --spiral_wave patterns could create interesting sci-fi structures or energy fields.
- Abstract/Artistic Scenes: The various patterns themselves can be used to create abstract art installations or stylized environments, even without representing realistic landscapes.
- Gameplay-Oriented Environments:
- FPS Maps: three_lane_map pattern provides a starting point for FPS level layouts, which can be further customized.
- Arena/Combat Zones: circle, grid, rectangle, diamond patterns could be used to create structured arena layouts with cover elements.
