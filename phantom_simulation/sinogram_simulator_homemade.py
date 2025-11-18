"""
This script aims to provide a homemade custom sinogram simulator for PET data.
The geometry for a 3D cylindrical scanner is considered.
It includes functions to generate LORs, compute system matrix elements using Siddon's algorithm,
and simulate sinograms from a given activity distribution.
"""

import numpy as np
import tqdm

# ---------- Helper: define image grid ----------
# image_origin: (x0, y0, z0) of voxel (0,0,0) center
# voxel_size: (vx, vy, vz)
# nx, ny, nz: grid sizes
class ImageGrid:
    def __init__(self, nx, ny, nz, voxel_size, origin=(0,0,0)):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.vx, self.vy, self.vz = voxel_size
        self.x0, self.y0, self.z0 = origin
    def voxel_center(self, i,j,k):
        x = self.x0 + (i + 0.5)*self.vx
        y = self.y0 + (j + 0.5)*self.vy
        z = self.z0 + (k + 0.5)*self.vz
        return np.array([x,y,z])
    def bounds(self):
        x_min = self.x0
        x_max = self.x0 + self.nx*self.vx
        y_min = self.y0
        y_max = self.y0 + self.ny*self.vy
        z_min = self.z0
        z_max = self.z0 + self.nz*self.vz
        return (x_min,x_max,y_min,y_max,z_min,z_max)

class SinogramSimulatorHomemade:

    def __init__(self, voxel_size, n_rings, n_detectors_per_ring, ring_spacing, scanner_radius):
        """
        Initialize the sinogram simulator with scanner geometry parameters.
        Args:
            n_rings (int): Number of detector rings.
            n_detectors_per_ring (int): Number of detectors per ring.
            ring_spacing (float): Spacing between rings in mm.
            scanner_radius (float): Radius of the scanner in mm.
        """
        self.vx, self.vy, self.vz = voxel_size
        self.n_rings = n_rings
        self.n_detectors_per_ring = n_detectors_per_ring
        self.ring_spacing = ring_spacing
        self.scanner_radius = scanner_radius
    
    def get_crystal_map(self):
        """
        Generate a map of crystal positions in the scanner.
        Returns:
            crystal_map (np.ndarray): Array of shape (n_crystals, 3) with (x, y, z) positions.
        """
        # NOTE : The crystals are equally spaced in the scanner geometry.

        n_crystals = self.n_rings * self.n_detectors_per_ring

        crystal_map = np.zeros((n_crystals, 3))
        for ring in range(self.n_rings):
            z = - (self.n_rings - 1) * self.ring_spacing / 2 + ring * self.ring_spacing
            for det in range(self.n_detectors_per_ring):
                angle = 2 * np.pi * det / self.n_detectors_per_ring
                x = self.scanner_radius * np.cos(angle)
                y = self.scanner_radius * np.sin(angle)
                crystal_index = ring * self.n_detectors_per_ring + det
                crystal_map[crystal_index] = [x, y, z]

        return crystal_map

    def get_grid(self):
        """
        Create an image grid for the simulation.
        Args:
            voxel_size (tuple): Size of each voxel in mm (vx, vy, vz).
            grid_size (tuple): Number of voxels in each dimension (nx, ny, nz).
        Returns:
            grid (ImageGrid): ImageGrid object defining the voxel grid.
        """
        self.grid = ImageGrid(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            voxel_size=(self.vx, self.vy, self.vz),
            origin=(
                - (self.nx * self.vx) / 2,
                - (self.ny * self.vy) / 2,
                - (self.nz * self.vz) / 2
            )
        )
        return self.grid

    def get_list_LORs(self):
        """
        Generate a list of Lines of Response (LORs) based on the crystal map.
        Returns:
            lors (np.ndarray): Array of shape (n_LORs, 2, 3) with start and end points of each LOR.
        """
        crystal_map = self.get_crystal_map()
        n_crystals = crystal_map.shape[0]
        lors = []

        for i in range(n_crystals):
            for j in range(i + 1, n_crystals):
                start_point = crystal_map[i]
                end_point = crystal_map[j]
                lors.append((start_point, end_point))

        return np.array(lors)

    # def siddon_ray_intersections(self, p0, p1, grid: ImageGrid):
    #     """
    #     Compute the intersection lengths of a ray with the voxels in the image grid using Siddon's algorithm.
    #     Args:
    #         p0 (np.ndarray): 3D coordinates of the ray start point.
    #         p1 (np.ndarray): 3D coordinates of the ray end point.
    #         grid (ImageGrid): ImageGrid object defining the voxel grid.
    #     Returns:
    #         intersections (list of tuples): Each tuple contains (voxel_index, length).
    #     """
    #     # p0, p1: endpoints (3,)
    #     # Implementation adapted for clarity; it's not the fastest.
    #     x_min,x_max,y_min,y_max,z_min,z_max = grid.bounds()
    #     # Parametric line p(t) = p0 + t*(p1-p0), t in [0,1]
    #     d = p1 - p0
    #     # quickly check if line intersects bounding box
    #     # compute entry/exit t for box
    #     tmin = 0.0; tmax = 1.0
    #     for dim in range(3):
    #         if abs(d[dim]) < 1e-12:
    #             # parallel to planes normal to this axis
    #             coord = p0[dim]
    #             if coord < [x_min,y_min,z_min][dim] or coord > [x_max,y_max,z_max][dim]:
    #                 return [], []
    #         else:
    #             t1 = ([x_min,y_min,z_min][dim] - p0[dim]) / d[dim]
    #             t2 = ([x_max,y_max,z_max][dim] - p0[dim]) / d[dim]
    #             ta = min(t1,t2); tb = max(t1,t2)
    #             if tb < tmin or ta > tmax:
    #                 return [], []
    #             tmin = max(tmin, ta); tmax = min(tmax, tb)
    #     # restrict to [tmin,tmax]
    #     if tmax <= tmin:
    #         return [], []
    #     # sample a bunch of parametric points at voxel boundaries using Siddon logic
    #     # compute intersections with each set of planes and sort them
    #     planes_t = [tmin, tmax]
    #     # For each axis, add plane intersections
    #     # build list of voxel indices by stepping through t intervals
    #     # For simplicity (but slower), sample a fine subdivision within [tmin,tmax] and test which voxel center lies there.
    #     # Choose Nsteps proportional to max number of voxels along path:
    #     path_len = np.linalg.norm((p1 - p0) * (tmax - tmin))
    #     max_voxel_dim = min(grid.vx, grid.vy, grid.vz)
    #     Nsteps = max( int(np.ceil(path_len / (max_voxel_dim*0.25))) , 5)
    #     ts = np.linspace(tmin, tmax, Nsteps)
    #     vox_dict = {}
    #     for i in range(len(ts)-1):
    #         tm = 0.5*(ts[i]+ts[i+1])
    #         p = p0 + tm*d
    #         # compute voxel indices
    #         ix = int((p[0] - grid.x0) / grid.vx)
    #         iy = int((p[1] - grid.y0) / grid.vy)
    #         iz = int((p[2] - grid.z0) / grid.vz)
    #         if ix < 0 or ix >= grid.nx or iy < 0 or iy >= grid.ny or iz < 0 or iz >= grid.nz:
    #             continue
    #         key = (ix,iy,iz)
    #         vox_dict[key] = vox_dict.get(key, 0.0) + (ts[i+1]-ts[i])*np.linalg.norm(d)
    #     # convert to lists
    #     voxels = []
    #     lengths = []
    #     for (ix,iy,iz), plen in vox_dict.items():
    #         flat = (iz*grid.ny + iy)*grid.nx + ix
    #         voxels.append(flat)
    #         lengths.append(plen)
    #     return voxels, lengths


    def siddon_ray_trace(self, p0, p1, vol_shape, voxel_size, origin):
        """
        Compute Siddon traversal for a ray from p0 to p1 through a volume.

        Parameters
        ----------
        p0, p1 : array-like, shape (3,)
            Ray start and end in world coordinates (same units as voxel_size).
        vol_shape : tuple (nx,ny,nz)
            Number of voxels in each axis.
        voxel_size : tuple (sx,sy,sz)
            Voxel size in world units.
        origin : tuple (ox,oy,oz)
            World coordinates of voxel (0,0,0) corner (not center). IMPORTANT: choose consistent origin.

        Returns
        -------
        voxels : list of tuples (ix,iy,iz,length)
            Each visited voxel index and the length of the ray segment inside it.
        """
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        vol_shape = tuple(vol_shape)
        voxel_size = np.asarray(voxel_size, dtype=float)
        origin = np.asarray(origin, dtype=float)

        # parametric ray: p(t) = p0 + t*(p1-p0), t in [0,1]
        d = p1 - p0
        if np.allclose(d, 0.0):
            return []

        t0, t1 = 0.0, 1.0

        # Precompute for each axis
        voxels = []
        nx, ny, nz = vol_shape

        # compute initial voxel indices for p0 + epsilon along ray
        # compute intersections with slab boundaries to clip t range
        for axis in range(3):
            vmin = origin[axis]
            vmax = origin[axis] + voxel_size[axis] * vol_shape[axis]
            if d[axis] == 0.0:
                # if outside slab, no intersection
                if p0[axis] <= vmin or p0[axis] >= vmax:
                    # If exactly on boundary it might be included; we treat closed-open [vmin, vmax)
                    pass
                continue
            ta = (vmin - p0[axis]) / d[axis]
            tb = (vmax - p0[axis]) / d[axis]
            ta_min = min(ta, tb)
            ta_max = max(ta, tb)
            if ta_max < t0 or ta_min > t1:
                return []  # no intersection
            t0 = max(t0, ta_min)
            t1 = min(t1, ta_max)
            if t0 > t1:
                return []

        # if there's no intersection after clipping
        if t1 <= t0:
            return []

        # compute entry and exit points
        p_entry = p0 + d * t0
        p_exit = p0 + d * t1
        length_total = np.linalg.norm(p_exit - p_entry)
        if length_total == 0:
            return []

        # Convert a 3D coordinate to voxel index (ix,iy,iz) 0-based
        def coord_to_index(coord):
            rel = (coord - origin) / voxel_size
            # floor to get index; clamp to valid range (we assume ray lies inside now)
            idx = np.floor(rel).astype(int)
            return idx

        # initial voxel
        current = coord_to_index(p_entry)
        # clamp safety
        current = np.clip(current, [0,0,0], np.array(vol_shape)-1)

        # prepare stepping
        sign = np.sign(d)
        # replace zeros with +1 to avoid divide-by-zero later
        # but mark axes where d==0 separately
        big = 1e30
        delta_t = np.empty(3, dtype=float)
        next_t = np.empty(3, dtype=float)
        for i in range(3):
            if d[i] == 0.0:
                delta_t[i] = big
                next_t[i] = big
            else:
                if sign[i] > 0:
                    # next boundary in + direction: (i+1)*voxel_size + origin
                    bound = origin[i] + (current[i] + 1) * voxel_size[i]
                    next_t[i] = (bound - p0[i]) / d[i]
                    delta_t[i] = voxel_size[i] / abs(d[i])
                else:
                    # stepping negative: next boundary is current * voxel_size + origin
                    bound = origin[i] + (current[i]) * voxel_size[i]
                    next_t[i] = (bound - p0[i]) / d[i]
                    delta_t[i] = voxel_size[i] / abs(d[i])

        # Walk
        t = t0
        # While t < t1: we step to the nearest next_t
        # track a small epsilon to avoid infinite loops
        eps = 1e-12
        while t < t1 - eps:
            # find axis with smallest next_t
            k = int(np.argmin(next_t))
            t_next = min(next_t[k], t1)
            seg_length = max(0.0, (t_next - t) * np.linalg.norm(d))
            voxels.append((int(current[0]), int(current[1]), int(current[2]), seg_length))

            # advance
            t = t_next
            if t >= t1 - eps:
                break
            # move along axis k
            current[k] += int(sign[k]) if sign[k] != 0 else 0
            # If we've left the volume, break
            if current[0] < 0 or current[0] >= nx or current[1] < 0 or current[1] >= ny or current[2] < 0 or current[2] >= nz:
                break
            next_t[k] += delta_t[k]

        return voxels

    def compute_system_matrix(self, lors, grid: ImageGrid):
        """
        Compute the system matrix elements for the given LORs and image grid, using Siddon's algorithm.
        Args:
            lors (np.ndarray): Array of shape (n_LORs, 2, 3) with start and end points of each LOR.
            grid (ImageGrid): ImageGrid object defining the voxel grid.
        Returns:
            system_matrix (list of tuples): Each tuple contains (LOR_index, voxel_index, length).
        """
        # TODO

    def forward_project(self, activity_distribution, attenuation_map=None):
        """
        Simulate a sinogram from the given activity distribution and optional attenuation map.
        The forward projection is done using Siddon's algorithm.
        Args:
            activity_distribution (np.ndarray): 3D array representing the activity distribution.
            attenuation_map (np.ndarray, optional): 3D array representing the attenuation map.
        Returns:
            sinogram (np.ndarray): Simulated sinogram data.
        """

        self.nx, self.ny, self.nz = activity_distribution.shape

        # Get grid
        self.get_grid()
        # Get LORs
        self.lors = self.get_list_LORs()
        n_lors = self.lors.shape[0]

        # Initialize sinogram
        sinogram = np.zeros(n_lors)
        for i, lor in enumerate(tqdm.tqdm(self.lors)):
            p0, p1 = lor
            voxels = self.siddon_ray_trace(
                p0, p1,
                vol_shape=(self.nx, self.ny, self.nz),
                voxel_size=(self.vx, self.vy, self.vz),
                origin=(
                    - (self.nx * self.vx) / 2,
                    - (self.ny * self.vy) / 2,
                    - (self.nz * self.vz) / 2
                )
            )
            if len(voxels) == 0:
                continue


            for (ix, iy, iz, l) in voxels:
                voxel_index = (iz * self.ny + iy) * self.nx + ix
                sinogram[i] += activity_distribution[ix, iy, iz] * l
            if attenuation_map is not None:
                mu_sum = 0.0
                for (ix, iy, iz, l) in voxels:
                    mu_sum += attenuation_map[ix, iy, iz] * l
                sinogram[i] *= np.exp(-mu_sum)

            
            
            # sinogram[i] = sum(activity_distribution.flatten()[v] * l for v, l in zip(voxel_indices, lengths))
            # if attenuation_map is not None:
            #     mu_sum = sum(attenuation_map.flatten()[v] * l for v, l in zip(voxel_indices, lengths))
            #     sinogram[i] *= np.exp(-mu_sum)

        return sinogram

    def build_sinogram(self, activity_distribution, attenuation_map=None, n_angles=180, n_radial_bins=128, max_radius=100):
        """
        Wrapper function to build the sinogram from activity distribution and optional attenuation map.
        Args:
            activity_distribution (np.ndarray): 3D array representing the activity distribution.
            attenuation_map (np.ndarray, optional): 3D array representing the attenuation map.
        Returns:
            sinogram (np.ndarray): Simulated sinogram data.
        """
        flattened_sinogram = self.forward_project(activity_distribution, attenuation_map)
        sino = np.zeros((n_angles, n_radial_bins), dtype=np.float32)

        for lor, c in zip(self.lors, flattened_sinogram):
            p0, p1 = lor
            v = p1 - p0
            theta = np.arctan2(v[1], v[0])
            theta = (theta + np.pi) % np.pi

            angle_id = int(np.floor(theta / np.pi * n_angles))

            m = 0.5*(p0 + p1)
            s = m[0]*np.cos(theta) + m[1]*np.sin(theta)

            s_norm = (s + max_radius) / (2*max_radius)
            radial_id = int(np.floor(s_norm * n_radial_bins))

            if 0 <= radial_id < n_radial_bins:
                sino[angle_id, radial_id] += c

        return sino

        

if __name__ == "__main__":

    simulator = SinogramSimulatorHomemade(
        voxel_size=(2.0, 2.0, 2.0),
        n_rings=1,
        n_detectors_per_ring=32,
        ring_spacing=4,
        scanner_radius=200.0
    )

    # crystal_map = simulator.get_crystal_map()
    # print("Crystal Map:\n", crystal_map)

    # # give visualization of the crystal map
    # try:
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(crystal_map[:, 0], crystal_map[:, 1], crystal_map[:, 2])
    #     ax.set_xlabel('X (mm)')
    #     ax.set_ylabel('Y (mm)')
    #     ax.set_zlabel('Z (mm)')
    #     ax.set_title('Crystal Map Visualization')
    #     plt.show()
    # except ImportError:
    #     print("matplotlib not installed, skipping visualization.")

    # simulate a circle
    activity_distribution = np.where(
        ( (np.linspace(-100,100,100).reshape(-1,1) **2 + np.linspace(-100,100,100).reshape(1,-1) **2)
          < (32**2) ),1.0, 0.0
    ).reshape(100,100,1)
    sinogram = simulator.build_sinogram(
        activity_distribution=activity_distribution,
        attenuation_map=None
    )
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(sinogram, aspect='auto', cmap='gray')
        plt.show()
    except ImportError:
        print("matplotlib not installed, skipping sinogram plot.")