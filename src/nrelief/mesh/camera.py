import numpy as np

ROTATE_SPEED = 0.005  # radians per pixel dragged
ZOOM_STEP = 0.9  # distance multiplier per scroll step
MAX_PITCH = np.pi / 2 - 0.01  # keeps the camera from flipping over the poles


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """View matrix: right-handed, camera looks down -Z."""
    eye, target, up = (np.asarray(v, dtype=np.float32) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    view = np.identity(4, dtype=np.float32)
    view[0, :3], view[1, :3], view[2, :3] = s, u, -f
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Projection matrix: right-handed, clip-space z in [0, 1]."""
    t = 1.0 / np.tan(fov_y / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = t / aspect
    proj[1, 1] = t
    proj[2, 2] = far / (near - far)
    proj[2, 3] = near * far / (near - far)
    proj[3, 2] = -1.0
    return proj


class OrbitCamera:
    """Camera that orbits around `target` on a sphere (yaw, pitch, distance)."""

    def __init__(self, target, radius: float, fov_y: float = np.radians(50.0)):
        self.target = np.asarray(target, dtype=np.float32)
        self.radius = radius  # radius of the mesh bounding sphere
        self.fov_y = fov_y
        self.yaw = 0.6
        self.pitch = 0.3
        # Start far enough for the whole bounding sphere to fit in the view.
        self.dist = 1.2 * radius / np.sin(fov_y / 2.0)

    def rotate(self, dx: float, dy: float):
        self.yaw += dx * ROTATE_SPEED
        self.pitch = float(np.clip(self.pitch + dy * ROTATE_SPEED, -MAX_PITCH, MAX_PITCH))

    def zoom(self, steps: float):
        self.dist = float(
            np.clip(self.dist * ZOOM_STEP**steps, 0.2 * self.radius, 50.0 * self.radius)
        )

    @property
    def eye(self) -> np.ndarray:
        # Spherical -> Cartesian, offset by the orbit target.
        offset = self.dist * np.array(
            [
                np.cos(self.pitch) * np.sin(self.yaw),
                np.sin(self.pitch),
                np.cos(self.pitch) * np.cos(self.yaw),
            ],
            dtype=np.float32,
        )
        return self.target + offset

    def view_proj(self, aspect: float) -> np.ndarray:
        view = look_at(self.eye, self.target)
        proj = perspective(
            self.fov_y, aspect, near=0.01 * self.dist, far=self.dist + 4.0 * self.radius
        )
        return proj @ view
