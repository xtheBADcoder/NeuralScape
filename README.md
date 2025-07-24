# NeuralScape
AI driven 3D environment creation for Unreal Engine
# Getting Started:
- Step 1: Enable the the following Plugins in your project - Python Editor Script Plugin, Python Foundation Packages, Editor Scripting Utilities, Geometry Script and Water.
- Step 2: Install Google Gemini (pip Install google-genai) into your Unreal python environment. Watch video if you need help doing so... https://www.youtube.com/watch?v=Qt0AW08REKg&list=WL&index=1&ab_channel=MattLakeTA
- Step 3: Go to https://aistudio.google.com/app/apikey and create and copy your own api key.
- Step 4: Open the event graph within both Editor Utility Widgets and add your api key into the line gemini_api_key = "" within the python script. Compile and save.
- Step 5: Now run the Widget, type a prompt and hit Generate.
# How it works:
- Precision Mode
Leverage the power of AI to generate detailed and structured scenes from a simple text prompt. This tool spawns a wide array of assets—including Static Meshes, Niagara VFX, and Blueprints—using a vast library of precise placement patterns. Go beyond simple spawning by procedurally configuring PCG graphs, water bodies, fog, and post-process settings for a fully realized environment, all within a user-defined area. Can also control parameter overrides for PCG Graphs,Blueprints,Niagara Systems, post-processing and fog.
- Crazy Fun Mode
Unleash creative chaos! This mode discovers all available assets in your project and empowers the AI to act as a wild environment artist. Feed it a prompt, and watch as it intelligently selects from your meshes, Blueprints, and VFX to build surprising scenes. The AI will even attempt to creatively configure parameters on your assets to match the theme, leading to unique, fun, and unpredictable results.
- Best Practice
Use PCG volumes for the bounding boxes(user defined areas).
# Whats Possible:
- scatter: Randomly places assets throughout the defined area.
- grid: Arranges assets in a grid formation with optional jitter.
- jittered_grid: Arranges assets in a grid, then applies a significant random offset within each cell.
- staggered_grid: Creates a grid where every other row or column is offset, like a brick pattern.
- hexagonal	Arranges: assets in a hexagonal grid pattern.
- cluster	Randomly places: assets within a specified radius of a central point.
- multiple_clusters	Creates - several smaller, randomly placed clusters within the defined area.
- density_map_scatter	Scatters: assets with a higher probability of spawning near the center of the area.
- gradient_density_scatter: Scatters assets based on a linear falloff from a defined density_center.
- checkerboard: Places assets on alternating squares of a grid.
- poisson_disc_scatter	Simplified: A fallback that currently behaves like the scatter pattern.
- voronoi	Simplified: Creates several random seed points and scatters assets around them.
2D Shapes & Outlines	
- circle: Arranges assets in a perfect circle.
- ring / annulus: Randomly places assets in the space between an inner and outer radius.
- rectangle: Places assets along the outline of a rectangle.
- rounded_rectangle: Places assets along the outline of a rectangle with rounded corners.
- cross: Arranges assets in a cross shape.
- l_shape: Arranges assets in an "L" shape.
- diamond: Places assets along the outline of a diamond shape.
- polygon: Places assets along the outline of a regular polygon with a definable number of sides.
- triangle: Places assets on the vertices, edges, or filled area of a triangle defined by 3 points.
- star: Arranges assets in a star shape with a definable number of points.
- chevron / arrow: Arranges assets in a V-shape, like a chevron or arrowhead.
- crescent: Randomly places assets within a crescent moon shape.
- heart: Arranges assets in the shape of a heart.
- gear: Arranges assets in the shape of a gear, including teeth and an inner circle.
- infinity_symbol: Arranges assets along the path of a lemniscate (infinity symbol).
- teardrop_petal: Arranges assets in a teardrop or petal shape.
2D Paths & Curves	
- linear_path:  Places assets randomly along a straight line between two points.
- arc: Arranges assets along a circular arc between a start and end angle.
- s_curve: Places assets along a cubic Bézier curve, creating a smooth "S" shape.
- wave: laces assets along a sine wave path.
- zigzag: Places assets in a zigzag pattern between two points.
- spiral: Arranges assets in an Archimedean spiral.
- fibonacci_spiral: Arranges assets in a logarithmic spiral (Golden Spiral).
- spiral_wave: Creates a spiral path with an added sine wave for a wobbly effect.
- riverbed: Simulates a meandering river path with varying width.
Thematic & Organic	
- small_village / large_village: Simulates a village layout by placing assets in slightly randomized grid cells.
- city: Creates a simplified city block layout with buildings placed inside blocks.
- three_lane_map: Places assets along three parallel lanes, simulating a MOBA-style map layout.
- fractal: Generates a simple fractal tree structure and places assets at its points.
- radial_sunburst: Places assets in rays extending from a central point.
- circular_shockwave: Arranges assets in expanding, slightly randomized concentric rings.
Water-Based (Requires target_water_body)	
- water_edge_scatter: Scatters assets in a band along the edges of a specified water body's spline.
- water_spline_linear: Places assets linearly along the spline of a specified water body, with a lateral offset.
3D Patterns	
- floating: Randomly places assets in 3D space, creating a floating cloud effect.
- grid_3d: Arranges assets in a three-dimensional grid.
- sphere_surface: Evenly distributes assets on the surface of a sphere.
- cube_surface_edges: Places assets randomly on the surface or along the edges of a 3D cube.
- torus: Places assets on the surface of a 3D torus (donut shape).
- cylinder_surface: Places assets on the surface (sides, top, or bottom) of a cylinder.
- cone_surface: Places assets on the surface or circular base of a cone.
- pyramid_surface

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
# Demo
- https://www.youtube.com/watch?v=EMtDle6EDjM&t=107s&ab_channel=BrandonDavis
Check out the rest of Channel for more.
