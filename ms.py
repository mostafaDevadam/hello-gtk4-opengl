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

def shader_program_combined():
        vertex_shader = """
        #version 300 es
        precision highp float;

        in vec3 in_position;
        in vec3 in_normal;
        in vec4 in_color;

        uniform mat4 m_proj;
        uniform mat4 m_view;
        uniform mat4 m_model;

        out vec3 v_normal;
        out vec4 v_color;

        void main() {
            v_normal = mat3(m_model) * in_normal;
            v_color = in_color;
            gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
        }
        """

        fragment_shader = """
        #version 300 es
        precision highp float;

        in vec3 v_normal;
        in vec4 v_color;

        out vec4 fragColor;

        void main() {
            vec3 light_dir = normalize(vec3(1.5, 3.5, 2.5));
            float diff = max(dot(normalize(v_normal), light_dir), 0.3);
            
            vec3 diffuse = v_color.rgb * diff;
            vec3 ambient = v_color.rgb * 0.22;
            
            fragColor = vec4(diffuse + ambient, v_color.a);
        }
        """

        return vertex_shader, fragment_shader

def on_tick(canvas):
    # Diese Schleife signalisiert GTK im Hintergrund, den Canvas permanent neu zu zeichnen
    canvas.rotation += (canvas.speed / 100.0)
    canvas.queue_render()
    return True # True hält den Timer am Leben



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


def load_combined_mesh_data(bus_path, station_path, cb=None):
    bus_mesh = trimesh.load(bus_path, force='mesh')
    station_mesh = trimesh.load(station_path, force='mesh')
    
    # Offset the bus slightly so it parks alongside the station
    #bus_mesh.vertices += [0.0, 0.0, 0.7]  
    # 1. Scale the bus up (adjust 1.8 to make it even larger/smaller if needed)
    bus_mesh.apply_scale(1.8)

    

    if cb is not None:
        cb(bus_mesh) # This will now safely shift the bus on its Z axis!
    else:
        bus_mesh.vertices += [0.0, 0.0, 7.0]

    # Merge geometries together
    scene = trimesh.Scene([bus_mesh, station_mesh])
    combined_mesh = scene.to_mesh()

    # Global scale normalization
    combined_mesh.vertices -= combined_mesh.center_mass
    max_bound = np.max(np.abs(combined_mesh.vertices))
    if max_bound > 0:
        combined_mesh.vertices /= max_bound
        combined_mesh.vertices *= 1.4

    if not hasattr(combined_mesh, 'vertex_normals') or len(combined_mesh.vertex_normals) == 0:
        combined_mesh.generate_normals()

    color_visuals = combined_mesh.visual.to_color()
    colors = color_visuals.vertex_colors.astype('f4') / 255.0

    vertices = combined_mesh.vertices.astype('f4')
    normals = combined_mesh.vertex_normals.astype('f4')
    
    vertex_data = np.hstack([vertices, normals, colors])
    indices = combined_mesh.faces.astype('i4')

    return vertex_data.tobytes(), indices.tobytes()


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
        self.pos_y = 0.0  
        self.zoom = 4.5  

        self.bus_z_margin = 7.0

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
            v_bytes, i_bytes = load_tri_mesh_data("tri_3d.obj")
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
            eye=(0.0, 1.2, self.zoom),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0)
        )
        #model = Matrix44.from_eulers((0.0, self.rotation, 0.0))

        base_correction = Matrix44.from_x_rotation(-np.pi / 100)
        spin = Matrix44.from_y_rotation(self.rotation)
        translation = Matrix44.from_translation((0.0, self.pos_y, 0.0))
        
        model = translation * spin * base_correction


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

    def update_bus_mesh_z_margin(self, x):
                x.vertices += [0.0, 0.0, self.bus_z_margin]

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
        self.bus_file = "bus.obj"
        self.station_file = "stationBus.obj"

        self.tri_mesh = self.get_mesh(self.triangle_file) 
        self.tri_mesh_vi = self.get_mesh_data(self.tri_mesh)

        self.bus_mesh = self.get_mesh("bus.obj") 
        self.bus_mesh_vi = self.get_bus_mesh_data(self.bus_mesh)

        self.station_mesh = self.get_mesh("stationBus.obj") 
        self.station_mesh_vi = self.get_station_mesh_data(self.station_mesh)


        #v_, i_ = load_combined_mesh_data("bus.obj", "stationBus.obj", None)
        #self.v_ = v_
        #self.i_ = i_ 
        self.combined_mesh_vi = self.get_combined_mesh_data(self.get_mesh("bus.obj"), self.get_mesh("stationBus.obj"), None)
        

        

        
            
        


    
    def get_mesh(self, file_name):
        if not file_name:
            print("file_name is required!")
            return
        
        mesh = trimesh.load(file_name, force='mesh')
        return mesh
        

    def get_mesh_data(self, mesh):
        if not mesh:
            print("mesh is required!")
            return
        
        mesh.vertices -= mesh.center_mass
        
        if not hasattr(mesh, 'vertex_normals'):
               mesh.generate_normals()

        vertices = mesh.vertices.astype('f4')
        normals = mesh.vertex_normals.astype('f4')
        
        vertex_data = np.hstack([vertices, normals])
        indices = mesh.faces.astype('i4')

        #self.tri_mesh_vi = vertex_data.tobytes(), indices.tobytes()

        return vertex_data.tobytes(), indices.tobytes()
    
    def get_bus_mesh_data(self, mesh):
            if not mesh:
                print("mesh is required!")
                return
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
    
    def get_station_mesh_data(self, mesh):
            if not mesh:
                print("mesh is required!")
                return
    
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

            #texture_image = self.get_station_texture(mesh)

            return vertex_data.tobytes(), indices.tobytes(), self.get_station_texture(mesh)
    
    def get_station_texture(self, mesh):
                texture_image = None
                if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
                    texture_image = mesh.visual.material.image
                
                if texture_image is None:
                    texture_image = Image.new('RGB', (2, 2), (255, 255, 255))
                
                return texture_image
            
    def get_combined_mesh_data(self, bus_mesh, station_mesh, cb=None):
        if not bus_mesh or not station_mesh:
            print("bus_mesh and station_mesh are required!")
            return 
       
        bus_mesh.apply_scale(1.8)

        #bus_mesh.vertices += [0.0, 0.0, 7.0]

        if cb is not None:
            cb(bus_mesh) # This will now safely shift the bus on its Z axis!
        else:
            bus_mesh.vertices += [0.0, 0.0, 2.0]

        # Merge geometries together
        scene = trimesh.Scene([bus_mesh, station_mesh])
        combined_mesh = scene.to_mesh()

        # Global scale normalization
        combined_mesh.vertices -= combined_mesh.center_mass
        max_bound = np.max(np.abs(combined_mesh.vertices))
        if max_bound > 0:
            combined_mesh.vertices /= max_bound
            combined_mesh.vertices *= 1.4

        if not hasattr(combined_mesh, 'vertex_normals') or len(combined_mesh.vertex_normals) == 0:
            combined_mesh.generate_normals()

        color_visuals = combined_mesh.visual.to_color()
        colors = color_visuals.vertex_colors.astype('f4') / 255.0

        vertices = combined_mesh.vertices.astype('f4')
        normals = combined_mesh.vertex_normals.astype('f4')
        
        vertex_data = np.hstack([vertices, normals, colors])
        indices = combined_mesh.faces.astype('i4')

        return vertex_data.tobytes(), indices.tobytes()


        
                

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
        def cb_click(state = "combined"):
            if state == "tri":
             v_, i_ = self.tri_mesh_vi
             gl.set_shader(shader_program_tri())
             gl.set_new_model_data(v_, i_ , '3f 3f', 'in_position', 'in_normal')
            elif state == "bus":
                bus_v, bus_i = self.bus_mesh_vi
                gl.set_shader(shader_program_bus())
                gl.set_new_model_data(bus_v, bus_i , '3f 3f', 'in_position', 'in_normal')

            elif state == "station":
                station_v, station_i, station_tex = self.station_mesh_vi
                gl.set_shader(shader_program_station())
                gl.set_new_model_data(station_v, station_i , '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord', station_tex=station_tex)
            elif state == "combined":
                v_, i_ = self.combined_mesh_vi
                gl.set_shader(shader_program_combined())
                gl.set_new_model_data(v_, i_, '3f 3f 4f', 'in_position', 'in_normal', 'in_color')
            
            else:
                print("error")
            
            


        
        tri_btn.connect("clicked", lambda button : cb_click("tri"))

        bus_btn = Gtk.Button(label="Bus") 
        sidebar.append(bus_btn)
        bus_btn.connect("clicked", lambda button : cb_click("bus"))

        station_btn = Gtk.Button(label="Station") 
        sidebar.append(station_btn)
        station_btn.connect("clicked", lambda button : cb_click("station"))

        combined_btn = Gtk.Button(label="Combined") 
        sidebar.append(combined_btn)
        combined_btn.connect("clicked", lambda button : cb_click("combined"))
        

        #
        main_box.append(sidebar)
        #
        # Zoom Control
        zoom_label = Gtk.Label(label="Zoom Window:")
        zoom_label.set_halign(Gtk.Align.START)
        zoom_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1.5, 10.0, 0.1)
        zoom_slider.set_value(gl.zoom)
        zoom_footer = Gtk.Label(label=f"zoom: {gl.zoom:.1f}")
        zoom_footer.set_halign(Gtk.Align.START)

        def update_zoom(sl):
            val = sl.get_value()
            gl.zoom = val
            zoom_footer.set_label(f"zoom: {val:.1f}")

        zoom_slider.connect("value-changed", update_zoom)
        sidebar.append(zoom_label)
        sidebar.append(zoom_slider)
        sidebar.append(zoom_footer)

         # Speed Control
        speed_slider_label = Gtk.Label(label="Rotation Speed:")
        speed_slider_label.set_halign(Gtk.Align.START)
        speed_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 10.0, 0.2)
        speed_slider.set_value(gl.speed)
        speed_footer = Gtk.Label(label=f"speed: {gl.speed:.1f}")
        speed_footer.set_halign(Gtk.Align.START)
        def on_speed(sl):
            val = sl.get_value()
            gl.speed = val
            speed_footer.set_label(f"speed: {val:.1f}")

        speed_slider.connect("value-changed", on_speed)
        sidebar.append(speed_slider_label)
        sidebar.append(speed_slider)
        sidebar.append(speed_footer)

        
        # Height Control
        height_label = Gtk.Label(label="Height (Y Position):")
        height_label.set_halign(Gtk.Align.START)
        height_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, -2.0, 2.0, 0.1)
        height_slider.set_value(gl.pos_y)
        height_slider.connect("value-changed", lambda sl: setattr(gl, 'pos_y', sl.get_value()))
        sidebar.append(height_label)
        sidebar.append(height_slider)

        #
        # bus margin z
        bus_margin_z_label = Gtk.Label(label="Bus Z Margin Window:")
        bus_margin_z_label.set_halign(Gtk.Align.START)
        bus_margin_z_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1.5, 10.0, 0.1)
        bus_margin_z_slider.set_value(gl.bus_z_margin)
        
        bus_margin_z_footer = Gtk.Label(label=f"Bus Z Margin: {gl.bus_z_margin:.1f}")
        bus_margin_z_footer.set_halign(Gtk.Align.START)

        def update_bus_z_margin(sl):
            if not sl:
                print("not changed")
                return
            
            val = sl.get_value()
            gl.bus_z_margin = val
            bus_margin_z_footer.set_label(f"Bus Z Margin: {val:.1f}")
           
            
            # Re-run the mesh pipeline using the vertex callback
            """v_bytes, i_bytes = load_combined_mesh_data(
                self.bus_file, 
                self.station_file, 
                lambda x: gl.update_bus_mesh_z_margin(x)
            )"""


            v_bytes, i_bytes = self.get_combined_mesh_data(self.get_mesh("bus.obj"), self.get_mesh("stationBus.obj"), 
                                        lambda x: gl.update_bus_mesh_z_margin(x))

            gl.set_new_model_data(v_bytes, i_bytes, '3f 3f 4f', 'in_position', 'in_normal', 'in_color')


            # 2. ModernGL built-in data overwrite (orphan=True clears the old memory slot safely)
            gl.vbo.write(v_bytes)
            gl.ibo.write(i_bytes)
            
            # Push the updated buffers back into your ModernGL widget context
            #gl_widget.update_mesh_buffers(v_bytes, i_bytes) # Or however your widget uploads bytes to GPU
            #gl.queue_draw()


        bus_margin_z_slider.connect("value-changed", update_bus_z_margin)
        sidebar.append(bus_margin_z_label)
        sidebar.append(bus_margin_z_slider)
        sidebar.append(bus_margin_z_footer)

        #
        win.set_child(main_box)
        win.present()
        
        # Start GLib frame ticks
        GLib.timeout_add(16, on_tick, gl)

if __name__ == '__main__':
  app = GTK4App()
  app.run(sys.argv)