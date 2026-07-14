import sys
import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
from gi.repository import Gtk, Gdk, GLib
import moderngl
import numpy as np
from pyrr import Matrix44
import trimesh
from PIL import Image


import os
import sys
import numpy as np
import trimesh
import moderngl

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk


"""
1. shader: 
shader_program() state: "tri"|"bus"|"station" return program()
shader_program_tri(): return ver, frag
shader_program_bus(): return ver, frag
shader_program_station(): return ver, frag

set_shader_program(program): self.program

on_realize(): self.program=... <- default shader


2. pass mesh data: 
set_new_model_data()->call->shader_program()

each model -> click -> on_model_select()->set_new_model_data()



"""




def load_tri_mesh_data(filepath):
    """Loads mesh data using trimesh into bytes for ModernGL."""
    mesh = trimesh.load(filepath, force='mesh')
    
    # trimesh automatically handles normal generation
    if not hasattr(mesh, 'vertex_normals'):
        mesh.generate_normals()

    vertices = mesh.vertices.astype('f4')
    normals = mesh.vertex_normals.astype('f4')
    
    vertex_data = np.hstack([vertices, normals])
    indices = mesh.faces.astype('i4')

    return vertex_data.tobytes(), indices.tobytes()


class ModernGLWidgetTriangle(Gtk.GLArea):
    def __init__(self, obj_file_path):
        super().__init__()
        self.obj_file_path = obj_file_path
        
        # Initialize attributes
        self.ctx = None
        self.program = None
        self.vao = None
        self.rotation = 0.5
        self.speed = 2.0  # Configurable rotational speed constant
        self.zoom = 5.0

        # 1. Add a Y position variable (defaults to 0.0)
        self.pos_y = 0.0

        # Set widget layout properties to fill the container space
        self.set_hexpand(True)
        self.set_vexpand(True)

        # Request depth buffer and configure ES 3.0 target
        self.set_has_depth_buffer(True)
        self.set_required_version(3, 0) # Targets OpenGL ES 3.0 (300 es)

        # Signal Bindings
        self.connect('realize', self.on_realize)
        self.connect('render', self.on_render)
        self.connect('resize', self.on_resize)

        # Start animation frame ticking
        #GLib.timeout_add(16, self.on_tick)

    def on_realize(self, area):
        area.make_current()
        if area.get_error() is not None:
            print("GLArea initialization error:", area.get_error())
            return

        # Initialize ModernGL context inside active GTK Context
        self.ctx = moderngl.get_context()
        self.ctx.enable(self.ctx.DEPTH_TEST)

        # ES 3.0 compliant Shaders with explicit high precision floats
        vertex_shader = """
        #version 300 es
        precision highp float;

        in vec3 in_position;
        in vec3 in_normal;

        uniform mat4 m_proj;
        uniform mat4 m_view;
        uniform mat4 m_model;

        out vec3 v_normal;

        void main() {
            v_normal = mat3(m_model) * in_normal;
            gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
        }
        """

        fragment_shader = """
        #version 300 es
        precision highp float;

        in vec3 v_normal;
        out vec4 fragColor;

        void main() {
            vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
            float diff = max(dot(normalize(v_normal), light_dir), 0.2);
            vec3 color = vec3(0.2, 0.6, 1.0) * diff;
            fragColor = vec4(color, 1.0);
        }
        """

        try:
            self.program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
        except Exception as e:
            print("GLSL compilation failed:\n", e)
            return

        # Load obj buffers
        try:
            v_bytes, i_bytes = load_tri_mesh_data(self.obj_file_path)
            vbo = self.ctx.buffer(v_bytes)
            ibo = self.ctx.buffer(i_bytes)

            self.vao = self.ctx.vertex_array(
                self.program,
                [(vbo, '3f 3f', 'in_position', 'in_normal')],
                index_buffer=ibo
            )
        except Exception as e:
            print("Failed loading 3D Mesh Assets:\n", e)

    def on_resize(self, area, width, height):
        if not self.ctx or not self.program:
            return

        aspect = width / max(height, 1)
        proj = Matrix44.perspective_projection(45.0, aspect, 0.1, 100.0)
        self.program['m_proj'].write(proj.astype('f4').tobytes())

    def on_render(self, area, context):
        if not self.ctx or not self.program or not self.vao:
            return False

        # Make sure ModernGL renders to GTK's framebuffer
        fbo = self.ctx.detect_framebuffer()
        fbo.use()

        self.ctx.clear(0.15, 0.15, 0.18, 1.0)

        view = Matrix44.look_at(
            eye=(0.0, 3.0, self.zoom),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0)
        )
        #model = Matrix44.from_eulers((0.0, self.rotation, 0.0))

        # 1. Base rotation to flip Z-up to Y-up (-90 degrees in radians)
        base_correction = Matrix44.from_x_rotation(-np.pi / 100)
        # 2. Continuous rotation around the Y-axis (Up axis)
        spin = Matrix44.from_y_rotation(self.rotation)
        # 3. Create a translation matrix using our self.pos_y value
        translation = Matrix44.from_translation((0.0, self.pos_y, 0.0))
        # 3. Combine them: Apply base correction first, then spin it
        model = translation* spin * base_correction

        self.program['m_view'].write(view.astype('f4').tobytes())
        self.program['m_model'].write(model.astype('f4').tobytes())

        self.vao.render()
        return True

    def on_tick(self):
        self.rotation += 0.02
        self.queue_draw()
        return True


# bus

def load_bus_mesh_data(filepath):
    """
    Loads any 3D model (like bus.obj) using Trimesh.
    Centering and scaling are normalized so it fits perfectly on screen.
    """
    # Force loading as a single combined mesh (handles multi-part OBJ files)
    mesh = trimesh.load(filepath, force='mesh')
    
    # --- AUTO-CENTERING ---
    # Shift the geometry so its center of mass sits exactly at (0, 0, 0)
    mesh.vertices -= mesh.center_mass

    # --- AUTO-SCALING ---
    # Scale the model so its largest dimension is exactly 2.0 units wide/tall.
    # This prevents the camera from clipping inside the model.
    max_bound = np.max(np.abs(mesh.vertices))
    if max_bound > 0:
        mesh.vertices /= max_bound
        mesh.vertices *= 1.2  # Fine-tuned size factor for a comfortable perspective view

    # Ensure normal vectors exist for 3D depth shading
    if not hasattr(mesh, 'vertex_normals') or len(mesh.vertex_normals) == 0:
        mesh.generate_normals()

    vertices = mesh.vertices.astype('f4')
    normals = mesh.vertex_normals.astype('f4')
    
    # Pack positions (x,y,z) and normals (nx,ny,nz) side-by-side
    vertex_data = np.hstack([vertices, normals])
    indices = mesh.faces.astype('i4')

    return vertex_data.tobytes(), indices.tobytes()


class ModernGLWidgetBus(Gtk.GLArea):
    def __init__(self, obj_file_path):
        super().__init__()
        self.obj_file_path = obj_file_path
        
        self.ctx = None
        self.program = None
        self.vao = None
        self.rotation = 0.0
        self.speed = 2.0  # Custom rotation speed
        self.zoom = 4.5

        # 1. Add a Y position variable (defaults to 0.0)
        self.pos_y = 0.0

        # Setup GTK widget scaling behaviors
        self.set_hexpand(True)
        self.set_vexpand(True)

        self.set_has_depth_buffer(True)
        self.set_required_version(3, 0) # GLES 3.0 compatible

        self.connect('realize', self.on_realize)
        self.connect('render', self.on_render)
        self.connect('resize', self.on_resize)



    def on_realize(self, area):
        area.make_current()
        if area.get_error() is not None:
            print("GLArea initialization error:", area.get_error())
            return

        self.ctx = moderngl.get_context()
        self.ctx.enable(self.ctx.DEPTH_TEST)

        # ES 3.0 Shaders (Precision highp is mandatory)
        vertex_shader = """
        #version 300 es
        precision highp float;

        in vec3 in_position;
        in vec3 in_normal;

        uniform mat4 m_proj;
        uniform mat4 m_view;
        uniform mat4 m_model;

        out vec3 v_normal;

        void main() {
            v_normal = mat3(m_model) * in_normal;
            gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
        }
        """

        fragment_shader = """
        #version 300 es
        precision highp float;

        in vec3 v_normal;
        out vec4 fragColor;

        void main() {
            // Smooth directional lighting
            vec3 light_dir = normalize(vec3(1.5, 2.5, 2.0));
            float diff = max(dot(normalize(v_normal), light_dir), 0.1);
            
            // Slate Gray base color for the bus body
            vec3 base_color = vec3(0.65, 0.7, 0.75);
            vec3 diffuse = base_color * diff;
            
            // Ambient base color so unlit crevices don't lose definition
            vec3 ambient = vec3(0.12, 0.12, 0.15);
            
            fragColor = vec4(diffuse + ambient, 1.0);
        }
        """

        try:
            self.program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
        except Exception as e:
            print("Shader compilation failed:\n", e)
            return

        # Read and assign normalized geometry buffers
        try:
            v_bytes, i_bytes = load_bus_mesh_data(self.obj_file_path)
            vbo = self.ctx.buffer(v_bytes)
            ibo = self.ctx.buffer(i_bytes)

            self.vao = self.ctx.vertex_array(
                self.program,
                [(vbo, '3f 3f', 'in_position', 'in_normal')],
                index_buffer=ibo
            )
            print(f"Successfully loaded and normalized '{self.obj_file_path}'")
        except Exception as e:
            print(f"Error loading {self.obj_file_path}:", e)

    def on_resize(self, area, width, height):
        if not self.ctx or not self.program:
            return

        aspect = width / max(height, 1)
        proj = Matrix44.perspective_projection(45.0, aspect, 0.1, 100.0)
        self.program['m_proj'].write(proj.astype('f4').tobytes())

    def on_render(self, area, context):
        if not self.ctx or not self.program or not self.vao:
            return False

        fbo = self.ctx.detect_framebuffer()
        fbo.use()

        # Render dark aesthetic background
        self.ctx.clear(0.1, 0.11, 0.13, 1.0)

        # Camera placement
        view = Matrix44.look_at(
            eye=(0.0, 1.5, self.zoom),      # Lowered camera slightly
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0)
        )
        
        # 1. Base rotation to flip Z-up to Y-up (-90 degrees in radians)
        base_correction = Matrix44.from_x_rotation(-np.pi / 100)
        # 2. Continuous rotation around the Y-axis (Up axis)
        spin = Matrix44.from_y_rotation(self.rotation)
        # 3. Create a translation matrix using our self.pos_y value
        translation = Matrix44.from_translation((0.0, self.pos_y, 0.0))
        # 3. Combine them: Apply base correction first, then spin it
        model = translation* spin * base_correction
        
        # Tilt model forward slightly and rotate around the Y axis
        #model = Matrix44.from_eulers((0.15, self.rotation, 0.0))

        self.program['m_view'].write(view.astype('f4').tobytes())
        self.program['m_model'].write(model.astype('f4').tobytes())

        self.vao.render()
        return True



# station 

def load_station_mesh_data(filepath):
    mesh = trimesh.load(filepath, force='mesh')
    
    mesh.vertices -= mesh.center_mass
    max_bound = np.max(np.abs(mesh.vertices))
    if max_bound > 0:
        mesh.vertices /= max_bound
        mesh.vertices *= 1.2

    if not hasattr(mesh, 'vertex_normals') or len(mesh.vertex_normals) == 0:
        mesh.generate_normals()

    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uvs = mesh.visual.uv.astype('f4')
    else:
        uvs = np.zeros((len(mesh.vertices), 2), dtype='f4')

    vertices = mesh.vertices.astype('f4')
    normals = mesh.vertex_normals.astype('f4')
    
    vertex_data = np.hstack([vertices, normals, uvs])
    indices = mesh.faces.astype('i4')

    texture_image = None
    if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
        texture_image = mesh.visual.material.image
    
    if texture_image is None:
        texture_image = Image.new('RGB', (2, 2), (255, 255, 255))

    return vertex_data.tobytes(), indices.tobytes(), texture_image


class ModernGLWidgetStation(Gtk.GLArea):
    def __init__(self, obj_file_path):
        super().__init__()
        self.obj_file_path = obj_file_path
        
        self.ctx = None
        self.program = None
        self.vao = None
        self.texture = None  # To hold the GPU texture object
        self.rotation = 0.0
        self.speed = 2.0  
        self.pos_y = 0.0  
        self.rotation_y = 100
        # 1. Zoom variable represents the camera's Z distance from the model
        self.zoom = 4.5
        

        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_has_depth_buffer(True)
        self.set_required_version(3, 0)

        self.connect('realize', self.on_realize)
        self.connect('render', self.on_render)
        self.connect('resize', self.on_resize)

    def on_realize(self, area):
        area.make_current()
        if area.get_error() is not None:
            print("GLArea initialization error:", area.get_error())
            return

        self.ctx = moderngl.get_context()
        self.ctx.enable(self.ctx.DEPTH_TEST)

        # ES 3.0 Shaders handling lighting & UV Coordinates
        vertex_shader = """
        #version 300 es
        precision highp float;

        in vec3 in_position;
        in vec3 in_normal;
        in vec2 in_texcoord; // Received from "2f" layout

        uniform mat4 m_proj;
        uniform mat4 m_view;
        uniform mat4 m_model;

        out vec3 v_normal;
        out vec2 v_texcoord;

        void main() {
            v_normal = mat3(m_model) * in_normal;
            v_texcoord = in_texcoord;
            gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
        }
        """

        fragment_shader = """
        #version 300 es
        precision highp float;

        in vec3 v_normal;
        in vec2 v_texcoord;

        uniform sampler2D u_texture; // Target Texture unit

        out vec4 fragColor;

        void main() {
            // Smooth directional lighting
            vec3 light_dir = normalize(vec3(1.5, 2.5, 2.0));
            float diff = max(dot(normalize(v_normal), light_dir), 0.25);
            
            // Sample color from stationBus.mtl image
            vec3 tex_color = texture(u_texture, v_texcoord).rgb;
            
            vec3 diffuse = tex_color * diff;
            vec3 ambient = tex_color * 0.15;
            
            fragColor = vec4(diffuse + ambient, 1.0);
        }
        """

        try:
            self.program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
        except Exception as e:
            print("Shader compilation failed:\n", e)
            return

        # Load mesh, normals, and UVs
        try:
            v_bytes, i_bytes, tex_img = load_station_mesh_data(self.obj_file_path)
            
            # Flip PIL image vertically to match OpenGL texture coordinate logic
            tex_img = tex_img.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Convert image to raw bytes
            img_data = tex_img.convert('RGBA').tobytes()
            
            # 1. Create and write Texture object on GPU
            self.texture = self.ctx.texture(tex_img.size, 4, data=img_data)
            self.texture.build_mipmaps()

            vbo = self.ctx.buffer(v_bytes)
            ibo = self.ctx.buffer(i_bytes)

            # 2. Map Layout: 3 floats (pos), 3 floats (normals), 2 floats (UV texcoords)
            self.vao = self.ctx.vertex_array(
                self.program,
                [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord')],
                index_buffer=ibo
            )
            print(f"Successfully loaded '{self.obj_file_path}' with materials!")
        except Exception as e:
            print(f"Error loading assets:", e)

    def on_resize(self, area, width, height):
        if not self.ctx or not self.program:
            return

        aspect = width / max(height, 1)
        proj = Matrix44.perspective_projection(45.0, aspect, 0.1, 100.0)
        self.program['m_proj'].write(proj.astype('f4').tobytes())

    def on_render(self, area, context):
        if not self.ctx or not self.program or not self.vao or not self.texture:
            return False

        fbo = self.ctx.detect_framebuffer()
        fbo.use()

        self.ctx.clear(0.1, 0.11, 0.13, 1.0)

        # Bind texture unit 0 to the shader
        self.texture.use(location=0)
        self.program['u_texture'] = 0

        view = Matrix44.look_at(
            eye=(0.0, 1.5, self.zoom),      
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0)
        )
        
        # Apply orientation fix & transforms
        base_correction = Matrix44.from_x_rotation(-np.pi / self.rotation_y)
        yaw_correction = Matrix44.from_z_rotation(np.pi / 100) # Rotate 180 degrees
        # Combine them into a single starting orientation
        initial_setup = yaw_correction * base_correction
        spin = Matrix44.from_y_rotation(self.rotation)
        translation = Matrix44.from_translation((0.0, self.pos_y, 0.0))
        
        model = translation * spin * initial_setup

        self.program['m_view'].write(view.astype('f4').tobytes())
        self.program['m_model'].write(model.astype('f4').tobytes())

        self.vao.render()
        return True




def on_tick(canvas):
    # Diese Schleife signalisiert GTK im Hintergrund, den Canvas permanent neu zu zeichnen
    canvas.rotation += (canvas.speed / 100.0)
    canvas.queue_render()
    return True # True hält den Timer am Leben


class GTK4App(Gtk.Application):
    def __init__(self):
        super().__init__(application_id='com.example.ModernGLGTK4')
        self.state = "triangle"

    def do_activate(self):
        win = Gtk.ApplicationWindow(application=self)
        win.set_title("GTK4 + ModernGL 3D Bus Loader")
        win.set_default_size(950, 650)

        # 1. Instance canvas pointing to your bus file
        gl_tri = ModernGLWidgetTriangle('tri_3d.obj') 
        #gl_tri.set_visible(False)

        gl_bus = ModernGLWidgetBus('bus.obj')
        gl_bus.set_visible(False)

        gl_station = ModernGLWidgetStation('stationBus.obj')
        gl_station.set_visible(False)



        #
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_size_request(800, -1)
        b_lbl = Gtk.Label(label="test...")
        box.append(b_lbl)

        # 2. Layout construction
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        main_box.set_margin_top(12)
        main_box.set_margin_bottom(12)
        main_box.set_margin_start(12)
        main_box.set_margin_end(12)
        #main_box.append(gl_widget)
        main_box.append(gl_tri)
        main_box.append(gl_bus)
        main_box.append(gl_station)
        #main_box.append(box)

        # 3. Sidebar Panel
        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        sidebar.set_size_request(220, -1)
        sidebar.set_halign(Gtk.Align.END)



        #
        def on_click(state, button):
            print(f"clicked: {state}")
            b_lbl.set_label(f"clicked: {state}")
            self.state = state
            if state == "triangle":
                gl_tri.set_visible(True)
                gl_bus.set_visible(False)
                gl_station.set_visible(False)

            elif state == "bus":
                gl_bus.set_visible(True)
                gl_tri.set_visible(False)
                gl_station.set_visible(False)
                
            elif state == "station":
                gl_station.set_visible(True)
                gl_tri.set_visible(False)
                gl_bus.set_visible(False)
            else:
                gl_tri.set_visible(True)
                gl_bus.set_visible(False)
                gl_station.set_visible(False)

                
                

        tri_btn = Gtk.Button(label="Triangle")
        sidebar.append(tri_btn)
        tri_btn.connect("clicked", lambda button : on_click("triangle", button))

        bus_btn = Gtk.Button(label="Bus") 
        sidebar.append(bus_btn)
        bus_btn.connect("clicked", lambda button : on_click("bus", button))

        station_btn = Gtk.Button(label="Station") 
        sidebar.append(station_btn)
        station_btn.connect("clicked", lambda button : on_click("station", button))



        #
        slider_label = Gtk.Label(label="Rotation Speed:")
        slider_label.set_halign(Gtk.Align.START)
        slider_footer = Gtk.Label(label=f"Rotation: 0")
        slider_footer.set_halign(Gtk.Align.START)
        slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 10.0, 0.2)
        #slider.set_value(gl_widget.speed)
        #slider.connect("value-changed", on_slider_changed, gl_widget)

        def cb(sl):
            if not self.state is "triangle":
                setattr(gl_bus, 'speed', sl.get_value())
                setattr(gl_station, 'speed', sl.get_value())

        
        slider.connect("value-changed", lambda sl: [
            slider_footer.set_label(f"Rotation: {sl.get_value(): .1f}"),
            cb(sl)
        ])


        sidebar.append(slider_label)
        sidebar.append(slider)
        sidebar.append(slider_footer)

        #gl_widget.pos_y = 5.5

        # --- SLIDER 2: Height (Y Position) ---
        height_slider_label = Gtk.Label(label="Height (Y Position):")
        height_slider_label.set_halign(Gtk.Align.START)
        height_slider_footer = Gtk.Label(label=f"Height: 0")
        height_slider_footer.set_halign(Gtk.Align.START)
        # Allows shifting the bus up or down by up to 2.0 units
        height_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, -2.0, 2.0, 0.1)
        #height_slider.set_value(gl_widget.pos_y)
        
        # Connect the height slider to update the canvas' pos_y directly
        #height_slider.connect("value-changed", lambda s: setattr(gl_widget, 'pos_y', s.get_value()))
        def cb_height(sl):
            if self.state:
                setattr(gl_tri, 'pos_y', sl.get_value())
                setattr(gl_bus, 'pos_y', sl.get_value())
                setattr(gl_station, 'pos_y', sl.get_value())
                height_slider_label.set_label(f"Height: {sl.get_value(): .1f}")

        height_slider.connect("value-changed", cb_height)
        sidebar.append(height_slider_label)
        sidebar.append(height_slider)
        sidebar.append(height_slider_footer)


         # --- SLIDER 3: Zoom (Camera Distance) ---
        zoom_label = Gtk.Label(label="Zoom (Distance):")
        zoom_label.set_halign(Gtk.Align.START)

        zoom_footer_label = Gtk.Label()

        # 1.5 is very close up, 10.0 is far away
        zoom_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1.5, 10.0, 0.1)
        #zoom_slider.set_value(gl_widget.zoom)
        def cb_zoom(sl):
            if self.state:
                setattr(gl_tri, 'zoom', sl.get_value())
                setattr(gl_bus, 'zoom', sl.get_value())
                setattr(gl_station, 'zoom', sl.get_value())
                zoom_footer_label.set_label(f"zoom: {sl.get_value(): .1f}")

        zoom_slider.connect("value-changed", cb_zoom)
        
        #zoom_footer_label = Gtk.Label(label=f"zoom: {gl_widget.zoom:.1f}")
        zoom_footer_label.set_halign(Gtk.Align.START)
        sidebar.append(zoom_label)
        sidebar.append(zoom_slider)
        sidebar.append(zoom_footer_label)





        main_box.append(sidebar)

        win.set_child(main_box)
        win.present()
        
        # Start GLib frame ticks
        GLib.timeout_add(16, on_tick, gl_tri)
        GLib.timeout_add(16, on_tick, gl_bus)
        GLib.timeout_add(16, on_tick, gl_station)

if __name__ == '__main__':
  app = GTK4App()
  app.run(sys.argv)