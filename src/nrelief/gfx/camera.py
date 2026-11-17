import numpy as np
import slangpy as spy
from slangpy import math as sm

ROTATE_SPEED = 0.005  # radians per pixel dragged
ZOOM_STEP = 0.9  # distance multiplier per scroll step
MAX_PITCH = np.pi / 2 - 0.01  # keeps the camera from flipping over the poles


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

    def view(self) -> spy.float4x4:
        """View matrix: right-handed, camera looks down -Z."""
        return sm.matrix_from_look_at(
            spy.float3(*self.eye.tolist()), spy.float3(*self.target.tolist()), spy.float3(0, 1, 0)
        )

    def proj(self, aspect: float) -> spy.float4x4:
        """Projection matrix: right-handed, clip-space z in [0, 1]."""
        return sm.perspective(
            self.fov_y, aspect, z_near=0.01 * self.dist, z_far=self.dist + 4.0 * self.radius
        )

    def view_proj(self, aspect: float) -> spy.float4x4:
        return sm.mul(self.proj(aspect), self.view())
