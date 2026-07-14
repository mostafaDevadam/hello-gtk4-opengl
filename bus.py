import sys
import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
from gi.repository import Gtk, Gdk, GLib
import moderngl
import numpy as np
from pyrr import Matrix44
import trimesh

def load_mesh_data(filepath):
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


class ModernGLWidget(Gtk.GLArea):
    def __init__(self, obj_file_path):
        super().__init__()
        self.obj_file_path = obj_file_path
        
        self.ctx = None
        self.program = None
        self.vao = None
        self.rotation = 0.0
        self.speed = 2.0  # Custom rotation speed

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
            v_bytes, i_bytes = load_mesh_data(self.obj_file_path)
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
            eye=(0.0, 1.5, 4.5),      # Lowered camera slightly
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0)
        )
        
        # 1. Base rotation to flip Z-up to Y-up (-90 degrees in radians)
        base_correction = Matrix44.from_x_rotation(-np.pi / 2)
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


# --- Widget Event Binding Handlers ---

def on_tick(canvas):
    # Calculate angular velocity delta based on sidebar slider value
    canvas.rotation += (canvas.speed / 120.0)
    canvas.queue_draw()
    return True


def on_slider_changed(slider, canvas):
    canvas.speed = slider.get_value()


class GTK4App(Gtk.Application):
    def __init__(self):
        super().__init__(application_id='com.example.ModernGLGTK4')

    def do_activate(self):
        win = Gtk.ApplicationWindow(application=self)
        win.set_title("GTK4 + ModernGL 3D Bus Loader")
        win.set_default_size(950, 650)

        # 1. Instance canvas pointing to your bus file
        gl_widget = ModernGLWidget('bus.obj') 

        # 2. Layout construction
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        main_box.set_margin_top(12)
        main_box.set_margin_bottom(12)
        main_box.set_margin_start(12)
        main_box.set_margin_end(12)
        main_box.append(gl_widget)

        # 3. Sidebar Panel
        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        sidebar.set_size_request(220, -1)

        slider_label = Gtk.Label(label="Rotation Speed:")
        slider_label.set_halign(Gtk.Align.START)
        
        slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 10.0, 0.2)
        slider.set_value(gl_widget.speed)
        slider.connect("value-changed", on_slider_changed, gl_widget)

        sidebar.append(slider_label)
        sidebar.append(slider)

        #gl_widget.pos_y = 5.5

        # --- SLIDER 2: Height (Y Position) ---
        height_label = Gtk.Label(label="Height (Y Position):")
        height_label.set_halign(Gtk.Align.START)

        # Allows shifting the bus up or down by up to 2.0 units
        height_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, -2.0, 2.0, 0.1)
        height_slider.set_value(gl_widget.pos_y)
        
        # Connect the height slider to update the canvas' pos_y directly
        height_slider.connect("value-changed", lambda s: setattr(gl_widget, 'pos_y', s.get_value()))

        sidebar.append(height_label)
        sidebar.append(height_slider)





        main_box.append(sidebar)

        win.set_child(main_box)
        win.present()
        
        # Start GLib frame ticks
        GLib.timeout_add(16, on_tick, gl_widget)

if __name__ == '__main__':
    app = GTK4App()
    app.run(sys.argv)