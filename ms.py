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

VERTEX_SHADER = """
 #version 300 es
precision highp float;

in vec3 in_position;
in vec3 in_normal;
in vec4 in_color;

out vec4 v_color;
out vec3 v_normal;

void main() {
    gl_Position = vec4(in_position, 1.0);
    v_color = in_color;
    v_normal = in_normal;
}
"""

FRAGMENT_SHADER = """
#version 300 es
precision highp float;

in vec4 v_color;
in vec3 v_normal;
out vec4 fragColor;

void main() {
    // Light from top-front-right
    vec3 light_dir = normalize(vec3(0.5, 1.0, 0.5));
    float diffuse = max(dot(normalize(v_normal), light_dir), 0.0);
    
    vec3 ambient = 0.3 * v_color.rgb;
    vec3 active_diffuse = 0.7 * v_color.rgb * diffuse;
    
    fragColor = vec4(ambient + active_diffuse, v_color.a);
}
"""

def shader_program(state = "tri"):
    if state == "tri":
        return shader_program_tri()
    elif state == "bus":
        return shader_program_bus()
    else:
        return shader_program_station()


def shader_program_tri(): 
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

    return vertex_shader, fragment_shader

def shader_program_bus(): 
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
    
    return vertex_shader, fragment_shader

def shader_program_station(): 
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

    return vertex_shader, fragment_shader


#

class ModernGLWidget(Gtk.GLArea):

    def __init__(self):
        super().__init__()
        self.program = None
        self.ctx = None
        self.vbo = None
        self.ibo = None
        self.vao = None
        # Set widget layout properties to fill the container space
        self.set_hexpand(True)
        self.set_vexpand(True)

        # Request depth buffer and configure ES 3.0 target
        self.set_has_depth_buffer(True)
        self.set_required_version(3, 0) # Targets OpenGL ES 3.0 (300 es)

        

        # Signal Bindings
        self.connect('realize', self.on_realize)
        self.connect('render', self.on_render)
        #self.connect('resize', self.on_resize)
        #self.connect("unrealize", self.on_unrealize)
    

   

    
    def on_realize(self, widget):
        self.make_current()
        if self.get_error() is not None:
            print("GLArea initialization error:", self.get_error())
            return

        # Initialize ModernGL context inside active GTK Context
        self.ctx = moderngl.get_context()
        self.ctx.enable(self.ctx.DEPTH_TEST)

        
        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )
        #
        # Load obj buffers
        # Initial fallback display
        initial_v = np.array([
            [-0.5, -0.5, 0.0,  0.0, 0.0, 1.0,  1.0, 0.0, 0.0, 1.0],
            [ 0.5, -0.5, 0.0,  0.0, 0.0, 1.0,  0.0, 1.0, 0.0, 1.0],
            [ 0.0,  0.5, 0.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0, 1.0]
        ], dtype='f4').tobytes()
        initial_i = np.array([0, 1, 2], dtype='i4').tobytes()

        self.set_new_model_data(initial_v, initial_i)

    def on_render(self, area, context):
        if not self.ctx or not self.vao:
            return True
        self.ctx.clear(0.12, 0.12, 0.14, 1.0, depth=1.0)
        self.vao.render(moderngl.TRIANGLES)
        return True

    def on_unrealize(self, area):
        self.make_current()
        if self.vao: self.vao.release()
        if self.vbo: self.vbo.release()
        if self.ibo: self.ibo.release()
        if self.program: self.program.release()
        if self.ctx: self.ctx.release()

    

    def set_new_model_data(self, v_bytes, i_bytes):
        print("set_new_model_data", v_bytes, i_bytes)
        # Load obj buffers
        self.make_current()
        
        #if self.vao: self.vao.release()
        #if self.vbo: self.vbo.release()
        #if self.ibo: self.ibo.release()

        try:

            self.vbo = self.ctx.buffer(v_bytes)
            self.ibo = self.ctx.buffer(i_bytes)

            #self.vbo.write(v_bytes)
            #self.ibo.write(i_bytes)

            self.vao = self.ctx.vertex_array(
                self.program,
                [(self.vbo, '3f 3f 4f', 'in_position', 'in_normal', 'in_color')],
                index_buffer=self.ibo
            )
            print(f"success write vao")
        except Exception as e:
            print(f"Error: {e}")



        """self.vao = self.ctx.vertex_array(
                self.program,
                [(self.vbo, '3f 3f', 'in_position', 'in_normal')],
                index_buffer=self.ibo
            )"""



#


class MeshViewerApp(Gtk.ApplicationWindow):
    def __init__(self):
        super().__init__(application_id='com.example.ModernGLGTK4')
        self.state = "triangle"
        self.gl_tri = None
        self.gl_bus = None
        self.gl_station = None

        self.triangle_file = "tri_3d.obj"


        self.raw_triangle = self._load_or_generate_triangle()
        #self.raw_bus = self._load_or_generate_bus()
        self.gl_widget = None


    def _load_or_generate_triangle(self):
        if os.path.exists(self.triangle_file):
            return trimesh.load(self.triangle_file, force='mesh')
        # Default procedural triangle fallback
        mesh = trimesh.Trimesh(
            vertices=[[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
            faces=[[0, 1, 2]]
        )
        mesh.visual.vertex_colors = [220, 50, 50, 255] # Red
        return mesh
    
    def get_mesh_bytes(self, mesh):
        # trimesh automatically handles normal generation
        if not hasattr(mesh, 'vertex_normals'):
            mesh.generate_normals()

        vertices = mesh.vertices.astype('f4')
        normals = mesh.vertex_normals.astype('f4')
        
        vertex_data = np.hstack([vertices, normals])
        indices = mesh.faces.astype('i4')

        return vertex_data.tobytes(), indices.tobytes() 


    def on_model_select(self, state):
            print(f"clicked: {state}")
            #b_lbl.set_label(f"clicked: {state}")
            self.state = state
            if state == "triangle":
                #self.gl_tri.set_visible(True)
                #self.gl_bus.set_visible(False)
                #self.gl_station.set_visible(False)
                #
                if self.raw_triangle:
                    print("raw tri")
                mesh = self.raw_triangle.copy()
                print(f"tri mesh len: {mesh}")
                mesh.vertices -= mesh.center_mass
                max_bound = np.max(np.abs(mesh.vertices))
                """if max_bound > 0:
                    mesh.vertices /= max_bound
                    mesh.vertices *= 1.0"""
                v_bytes, i_bytes = self.get_mesh_bytes(mesh)

                
                
                #self.gl_widget.set_new_model_data(v_bytes, i_bytes)

                

            elif state == "bus":
                pass
                #self.gl_bus.set_visible(True)
                #self.gl_tri.set_visible(False)
                #self.gl_station.set_visible(False)
                
                
            elif state == "station":
                pass
                #self.gl_station.set_visible(True)
                #self.gl_tri.set_visible(False)
                #self.gl_bus.set_visible(False)
                
            else:
                pass
                #self.gl_tri.set_visible(True)
                #self.gl_bus.set_visible(False)
                #self.gl_station.set_visible(False)

            #
            
            self.gl_widget.set_new_model_data(v_bytes, i_bytes)
            self.gl_widget.queue_draw()
                

    def do_activate(self):
        self.raw_triangle = self._load_or_generate_triangle()
        #
        win = Gtk.ApplicationWindow(application=self)
        win.set_title("GTK4 + ModernGL 3D Bus Loader")
        win.set_default_size(950, 650)
        self.gl_widget = ModernGLWidget()
        self.gl_widget.set_visible(True)
        

        # 1. Instance canvas pointing to your bus file
        #self.gl_tri = ModernGLWidgetTriangle('tri_3d.obj') 
        #gl_tri.set_visible(False)

        #self.gl_bus = ModernGLWidgetBus('bus.obj')
        #self.gl_bus.set_visible(False)

        #self.gl_station = ModernGLWidgetStation('stationBus.obj')
        #self.gl_station.set_visible(False)



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
        #
        main_box.append(self.gl_widget)
        

        # 3. Sidebar Panel
        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        sidebar.set_size_request(220, -1)
        sidebar.set_halign(Gtk.Align.END)



        #
        tri_btn = Gtk.Button(label="Triangle")
        sidebar.append(tri_btn)
        tri_btn.connect("clicked", lambda button : self.on_model_select("triangle"))

        bus_btn = Gtk.Button(label="Bus") 
        sidebar.append(bus_btn)
        #bus_btn.connect("clicked", lambda button : self.on_model_select("bus", button))

        station_btn = Gtk.Button(label="Station") 
        sidebar.append(station_btn)
        #station_btn.connect("clicked", lambda button : self.on_model_select("station", button))
        #
        slider_label = Gtk.Label(label="Rotation Speed:")
        slider_label.set_halign(Gtk.Align.START)
        slider_footer = Gtk.Label(label=f"Rotation: 0")
        slider_footer.set_halign(Gtk.Align.START)
        slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 10.0, 0.2)
        #slider.set_value(gl_widget.speed)
        #slider.connect("value-changed", on_slider_changed, gl_widget)

        


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
        

        #height_slider.connect("value-changed", cb_height)
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
       

        #zoom_slider.connect("value-changed", cb_zoom)
        
        #zoom_footer_label = Gtk.Label(label=f"zoom: {gl_widget.zoom:.1f}")
        zoom_footer_label.set_halign(Gtk.Align.START)
        sidebar.append(zoom_label)
        sidebar.append(zoom_slider)
        sidebar.append(zoom_footer_label)

        #
        main_box.append(sidebar)

        #
        win.set_child(main_box)
        win.present()
        
        # Start GLib frame ticks
        GLib.timeout_add(16, on_tick, self.gl_widget)
        #GLib.timeout_add(16, on_tick, self.gl_tri)
        #GLib.timeout_add(16, on_tick, self.gl_bus)
        #GLib.timeout_add(16, on_tick, self.gl_station)


def on_tick(canvas):
    # Diese Schleife signalisiert GTK im Hintergrund, den Canvas permanent neu zu zeichnen
    #canvas.rotation += (canvas.speed / 100.0)
    canvas.queue_render()
    return True # True hält den Timer am Leben



def load_mesh_data(filepath):
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



#

class GLWidget(Gtk.GLArea):
    def __init__(self):
        super().__init__()

        self.area = None
       
        
        # Initialize attributes
        self.ctx = None
        self.program = None
        self.vao = None
        self.vbo = None
        self.ibo = None
        self.texture = None


        self.rotation = 0.0
        self.speed = 2.0  # Configurable rotational speed constant

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
        self.area = area
        if area.get_error() is not None:
            print("GLArea initialization error:", area.get_error())
            return

        # Initialize ModernGL context inside active GTK Context
        self.ctx = moderngl.get_context()
        self.ctx.enable(self.ctx.DEPTH_TEST)


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
            ver, frag = shader_program_tri()
            self.program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
        except Exception as e:
            print("GLSL compilation failed:\n", e)
            return

        # Load obj buffers
        try:
            v_bytes, i_bytes = load_mesh_data("tri_3d.obj")
            initial_v = np.array([
                [-0.5, -0.5, 0.0,  0.0, 0.0, 1.0,  1.0, 0.0, 0.0, 1.0],
                [ 0.5, -0.5, 0.0,  0.0, 0.0, 1.0,  0.0, 1.0, 0.0, 1.0],
                [ 0.0,  0.5, 0.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0, 1.0]
            ], dtype='f4').tobytes()
            initial_i = np.array([0, 1, 2], dtype='i4').tobytes()

            #self.vbo = self.ctx.buffer(initial_v)
            #self.ibo = self.ctx.buffer(initial_i)
            self.vbo = self.ctx.buffer(v_bytes)
            self.ibo = self.ctx.buffer(i_bytes)

            self.vao = self.ctx.vertex_array(
                self.program,
                [(self.vbo, '3f 3f', 'in_position', 'in_normal')],
                index_buffer=self.ibo
            )
            """self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f 3f 4f', 'in_position', 'in_normal', 'in_color')],
            index_buffer=self.ibo
            )"""
        except Exception as e:
            print("Failed loading 3D Mesh Assets:\n", e)


    def update_matrices(self):
        """Ensures projection and view matrices are updated for the active shader"""
        if not self.program:
            return
            
        # Write View Matrix if it exists in current shader
        if 'm_view' in self.program:
            view = Matrix44.look_at(
                eye=(0.0, 1.5, 4.5),      
                target=(0.0, 0.0, 0.0),
                up=(0.0, 1.0, 0.0)
            )
            self.program['m_view'].write(view.astype('f4').tobytes())

        # Write Projection Matrix if it exists in current shader
        if 'm_proj' in self.program:
            aspect = self.get_width() / max(self.get_height(), 1)
            proj = Matrix44.perspective_projection(45.0, aspect, 0.1, 100.0)
            self.program['m_proj'].write(proj.astype('f4').tobytes())


    def set_new_model_data(self, v_bytes, i_bytes, f, *attributes, station_tex=None):
        #
        self.make_current()
        #self.area.make_current()

       
       

        if self.vao: self.vao.release()
        if self.vbo: self.vbo.release()
        if self.ibo: self.ibo.release()

        if self.texture: 
            self.texture.release()
            self.texture = None

        self.vbo = self.ctx.buffer(v_bytes)
        self.ibo = self.ctx.buffer(i_bytes)

        
        """if in_texture and station_tex:
                self.tex_img = station_tex.transpose(Image.FLIP_TOP_BOTTOM)
                # Convert image to raw bytes
                self.img_data = station_tex.convert('RGBA').tobytes()
                self.texture = self.ctx.texture(self.tex_img.size, 4, data=self.img_data)
                self.texture.build_mipmaps()
                self.vao = self.ctx.vertex_array(
                    self.program,
                    [(self.vbo, f, pos, nor, in_texture)],
                    self.ibo
                )

                self.ctx.clear(0.1, 0.11, 0.13, 1.0)

                # Bind texture unit 0 to the shader
                if self.texture and self.program['u_texture']:

                    self.texture.use(location=0)
                    self.program['u_texture'] = 0

                    


        else:
             self.vao = self.ctx.vertex_array(
                        self.program,
                        [(self.vbo, f, pos, nor)],
                        self.ibo
             )"""

        if station_tex is not None:
            # Correctly flip AND convert the transposed image
            flipped_img = station_tex.transpose(Image.FLIP_TOP_BOTTOM).convert('RGBA')
            img_data = flipped_img.tobytes()
            
            self.texture = self.ctx.texture(flipped_img.size, 4, data=img_data)
            self.texture.build_mipmaps()     
        


        # 4. Bind VAO dynamically matching the format string and attributes passed
        # E.g., for station: '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord'
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, f, *attributes)],
            index_buffer=self.ibo
        )

        # 5. Set up the camera perspective / view matrices safely
        self.update_matrices()
        
        # 6. Request GTK redraw
        self.queue_draw()
       

        """aspect = self.width / max(self.height, 1)
        proj = Matrix44.perspective_projection(45.0, aspect, 0.1, 100.0)
        if self.program['m_proj']:
           self.program['m_proj'].write(proj.astype('f4').tobytes())"""

        

        #self.vao.render()
        #self.queue_draw()

    def set_shader(self, program):
            ver, frag = program
            #print(ver, frag)
            self.program = self.ctx.program(
                vertex_shader=ver,
                fragment_shader=frag
            )
    
    def set_vao(self, f, pos, nor):
         self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, f, pos, nor)],
            self.ibo
        )



    def on_resize(self, area, width, height):
        if not self.ctx or not self.program:
            return
        
        self.width = width
        self.height = height

        aspect = self.width / max(self.height, 1)
        proj = Matrix44.perspective_projection(45.0, aspect, 0.1, 100.0)
        if self.program['m_proj']:
           self.program['m_proj'].write(proj.astype('f4').tobytes())

    def on_render(self, area, context):
        if not self.ctx or not self.program or not self.vao:
            return False

        # Make sure ModernGL renders to GTK's framebuffer
        fbo = self.ctx.detect_framebuffer()
        fbo.use()

        self.ctx.clear(0.15, 0.15, 0.18, 1.0)

        view = Matrix44.look_at(
            eye=(0.0, 3.0, 5.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0)
        )
        model = Matrix44.from_eulers((0.0, self.rotation, 0.0))

        self.program['m_view'].write(view.astype('f4').tobytes())
        self.program['m_model'].write(model.astype('f4').tobytes())

        if self.texture:
            self.texture.use(location=0)
            if 'u_texture' in self.program:
                self.program['u_texture'] = 0

        self.vao.render()

        #self.ctx.clear(0.12, 0.12, 0.14, 1.0, depth=1.0)
        #self.vao.render(moderngl.TRIANGLES)
        return True

    def on_tick(self):
        self.rotation += 0.02
        self.queue_draw()
        return True

    
    

    






class GTK4App(Gtk.Application):
    def __init__(self):
        super().__init__(application_id='com.example.ModernGLGTK4')
        self.state = "triangle"
        self.gl_tri = None
        self.gl_bus = None
        self.gl_station = None

        self.triangle_file = "tri_3d.obj"

         
        v_, i_ = load_mesh_data("tri_3d.obj")
        self.v_ = v_
        self.i_ = i_

        bus_v, bus_i = load_bus_mesh_data("bus.obj")
        self.bus_v = bus_v
        self.bus_i = bus_i

        station_v, station_i, station_tex = load_station_mesh_data("stationBus.obj")
        self.station_v = station_v
        self.station_i = station_i
        self.station_tex = station_tex

        
            
        # Flip PIL image vertically to match OpenGL texture coordinate logic
        self.tex_img = station_tex.transpose(Image.FLIP_TOP_BOTTOM)
        # Convert image to raw bytes
        self.img_data = station_tex.convert('RGBA').tobytes()
        # 1. Create and write Texture object on GPU
        #self.texture = self.ctx.texture(station_tex.size, 4, data=self.img_data)
        #self.texture.build_mipmaps()


    
                

    def do_activate(self):
        #self.raw_triangle = self._load_or_generate_triangle()
        #
        win = Gtk.ApplicationWindow(application=self)
        win.set_title("GTK4 + ModernGL")
        win.set_default_size(950, 650)

        gl = GLWidget() 

        
        



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
        #
        main_box.append(gl)
        

        # 3. Sidebar Panel
        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        sidebar.set_size_request(220, -1)
        sidebar.set_halign(Gtk.Align.END)



        #
        tri_btn = Gtk.Button(label="Triangle")
        sidebar.append(tri_btn)
        #tri_btn.connect("clicked", lambda button : self.on_model_select("triangle"))
        def cb_click(state):
            if state == "tri":
             #v_bytes, i_bytes = load_mesh_data("tri_3d.obj")
             gl.set_shader(shader_program_tri())
             #gl.set_vao('3f 3f', 'in_position', 'in_normal')
             #gl.vbo.write(self.v_)
             #gl.ibo.write(self.i_)
             gl.set_new_model_data(self.v_, self.i_ , '3f 3f', 'in_position', 'in_normal')
             
             #gl.queue_draw()
            elif state == "bus":
                gl.set_shader(shader_program_bus())
                gl.set_new_model_data(self.bus_v, self.bus_i , '3f 3f', 'in_position', 'in_normal')

            elif state == "station":
                gl.set_shader(shader_program_station())
                gl.set_new_model_data(self.station_v, self.station_i , '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord', station_tex=self.station_tex)


            else:
                print("error")
            
            #gl.queue_draw()


        
        tri_btn.connect("clicked", lambda button : cb_click("tri"))

        bus_btn = Gtk.Button(label="Bus") 
        sidebar.append(bus_btn)
        bus_btn.connect("clicked", lambda button : cb_click("bus"))

        station_btn = Gtk.Button(label="Station") 
        sidebar.append(station_btn)
        station_btn.connect("clicked", lambda button : cb_click("station"))
        

        #
        main_box.append(sidebar)

        #
        win.set_child(main_box)
        win.present()
        
        # Start GLib frame ticks
        #GLib.timeout_add(16, on_tick, self.gl_widget)
        #GLib.timeout_add(16, on_tick, self.gl_tri)
        #GLib.timeout_add(16, on_tick, self.gl_bus)
        #GLib.timeout_add(16, on_tick, self.gl_station)

if __name__ == '__main__':
  app = GTK4App()
  app.run(sys.argv)