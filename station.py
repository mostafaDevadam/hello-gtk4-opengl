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

def load_mesh_data(filepath):
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


class ModernGLWidget(Gtk.GLArea):
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
            v_bytes, i_bytes, tex_img = load_mesh_data(self.obj_file_path)
            
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


# --- Widgets Tick Loops ---

def on_tick(canvas):
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
        win.set_title("GTK4 + ModernGL Textured Bus Loader")
        win.set_default_size(950, 650)

        # Point this to your textured obj file
        gl_widget = ModernGLWidget('stationBus.obj') 

        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        main_box.set_margin_top(12)
        main_box.set_margin_bottom(12)
        main_box.set_margin_start(12)
        main_box.set_margin_end(12)
        main_box.append(gl_widget)

        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        sidebar.set_size_request(220, -1)

        # rotation_slider

        rotation_slider_label = Gtk.Label(label="Rotation Speed:")
        rotation_slider_label.set_halign(Gtk.Align.START)

        rotation_slider_footer_label = Gtk.Label()
        
        rotation_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 10.0, 0.2)
        rotation_slider.set_value(gl_widget.speed)
        #rotation_slider.connect("value-changed", on_slider_changed, gl_widget)
        rotation_slider.connect("value-changed", lambda sl: [
                  setattr(gl_widget, 'speed', sl.get_value()),
                  rotation_slider_footer_label.set_label(f"rotation: {sl.get_value(): .1f}")
                ])

        sidebar.append(rotation_slider_label)
        sidebar.append(rotation_slider)
        sidebar.append(rotation_slider_footer_label)

        height_label = Gtk.Label(label="Height (Y Position):")
        height_label.set_halign(Gtk.Align.START)

        height_footer_lbl = Gtk.Label()

        height_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, -2.0, 2.0, 0.1)
        height_slider.set_value(gl_widget.pos_y)
        height_slider.connect("value-changed", lambda s: [
            setattr(gl_widget, 'pos_y', s.get_value()),
            print(f"height y: {s.get_value()}"),
            height_footer_lbl.set_label(f"y: {s.get_value(): .1f}")
            ])
        

        sidebar.append(height_label)
        sidebar.append(height_slider)
        sidebar.append(height_footer_lbl)

        # --- SLIDER 3: Zoom (Camera Distance) ---
        zoom_label = Gtk.Label(label="Zoom (Distance):")
        zoom_label.set_halign(Gtk.Align.START)

        zoom_footer_label = Gtk.Label()

        # 1.5 is very close up, 10.0 is far away
        zoom_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1.5, 10.0, 0.1)
        zoom_slider.set_value(gl_widget.zoom)
        zoom_slider.connect("value-changed", lambda sl: [
                  setattr(gl_widget, 'zoom', sl.get_value()),
                  zoom_footer_label.set_label(f"zoom: {sl.get_value(): .1f}")
                ])
        
        #zoom_footer_label = Gtk.Label(label=f"zoom: {gl_widget.zoom:.1f}")
        zoom_footer_label.set_halign(Gtk.Align.START)
        sidebar.append(zoom_label)
        sidebar.append(zoom_slider)
        sidebar.append(zoom_footer_label)


        #rotation y

        rotation_y_label = Gtk.Label(label="Rotation (Y Position):")
        rotation_y_label.set_halign(Gtk.Align.START)

        rotation_y_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, -2.0, 2.0, 0.1)
        rotation_y_slider.set_value(gl_widget.rotation_y)
        rotation_y_slider.connect("value-changed", lambda s: setattr(gl_widget, 'rotation_y', s.get_value()))

        #sidebar.append(rotation_y_label)
        #sidebar.append(rotation_y_slider)

        main_box.append(sidebar)
        win.set_child(main_box)
        win.present()
        
        GLib.timeout_add(16, on_tick, gl_widget)

if __name__ == '__main__':
    app = GTK4App()
    app.run(sys.argv)