"""
This script aims to provide a homemade custom sinogram simulator for PET data.
The geometry for a 3D cylindrical scanner is considered.
It includes functions to generate LORs, compute system matrix elements using Siddon's algorithm,
and simulate sinograms from a given activity distribution.
"""

import numpy as np
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

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

    def _process_lor_batch(self, lor_indices, activity_distribution, attenuation_map, 
                           vol_shape, voxel_size, origin, scatter_fraction, random_fraction, seed,
                           attenuation_scale):
        """
        Process a batch of LORs and return their sinogram values.
        This method is designed to be called in parallel processes.
        
        Args:
            lor_indices (list): List of LOR indices to process.
            activity_distribution (np.ndarray): 3D activity distribution.
            attenuation_map (np.ndarray or None): 3D attenuation map.
            vol_shape (tuple): Volume shape (nx, ny, nz).
            voxel_size (tuple): Voxel size (vx, vy, vz).
            origin (tuple): Origin coordinates.
            scatter_fraction (float): Fraction of scatter events (0 to 1).
            random_fraction (float): Fraction of random events relative to prompt events (0 to 1).
            seed (int or None): Random seed for reproducibility.
            attenuation_scale (float): Scaling factor for attenuation map values.
        
        Returns:
            np.ndarray: Sinogram values for this batch.
        """
        batch_size = len(lor_indices)
        batch_result = np.zeros(batch_size)
        
        # Set random seed for this batch if provided
        if seed is not None:
            # Use a unique seed for each batch based on first LOR index
            np.random.seed(seed + lor_indices[0])
        
        for batch_idx, lor_idx in enumerate(lor_indices):
            p0, p1 = self.lors[lor_idx]
            voxels = self.siddon_ray_trace(
                p0, p1,
                vol_shape=vol_shape,
                voxel_size=voxel_size,
                origin=origin
            )
            
            # if len(voxels) == 0:
            #     continue
            
            # Compute true coincidence contribution
            true_counts = 0.0
            for (ix, iy, iz, l) in voxels:
                true_counts += activity_distribution[ix, iy, iz] * l

            # Convert from kBq/Ml to counts
            true_counts *= self.vx * self.vy * self.vz / 1e3  # Convert voxel volume from mm^3 to mL
            
            # Apply attenuation if provided
            if attenuation_map is not None:
                mu_sum = 0.0
                for (ix, iy, iz, l) in voxels:
                    # l is in mm, attenuation should be in mm^-1
                    # Apply scaling factor to convert attenuation map to correct units
                    # Typical value: 0.096 cm^-1 = 0.0096 mm^-1 for water at 511 keV
                    mu_sum += attenuation_map[ix, iy, iz] * attenuation_scale * l
                # Attenuation factor: exp(-mu*l) where mu is in mm^-1 and l is in mm
                attenuation_factor = np.exp(-mu_sum)
                true_counts *= attenuation_factor
            
            # Add scatter component (modeled as fraction of true counts)
            if scatter_fraction > 0:
                scatter_counts = true_counts * scatter_fraction / (1 - scatter_fraction)
            else:
                scatter_counts = 0.0
            
            # Compute prompt counts (true + scatter)
            prompt_counts = true_counts + scatter_counts
            
            # Add random coincidences (as fraction of prompt events)
            if random_fraction > 0:
                random_counts = prompt_counts * random_fraction / (1 - random_fraction)
            else:
                random_counts = 0.0
            
            # Total expected counts
            batch_result[batch_idx] = prompt_counts + random_counts
        
        return batch_result

    def forward_project(self, activity_distribution, attenuation_map=None, 
                       batch_size=1000, n_processes=None, use_multiprocessing=True,
                       add_noise=False, scatter_fraction=0.0, random_fraction=0.0, seed=None,
                       attenuation_scale=0.01):
        """
        Simulate a sinogram from the given activity distribution and optional attenuation map.
        The forward projection is done using Siddon's algorithm with batch processing and multiprocessing.
        
        Args:
            activity_distribution (np.ndarray): 3D array representing the activity distribution.
            attenuation_map (np.ndarray, optional): 3D array representing the attenuation map.
            batch_size (int): Number of LORs to process in each batch. Default: 1000.
            n_processes (int, optional): Number of parallel processes. If None, uses all available CPUs.
            use_multiprocessing (bool): Whether to use multiprocessing. Default: True.
            add_noise (bool): Whether to add Poisson noise to the sinogram. Default: False.
            scatter_fraction (float): Fraction of scatter events relative to total events (0 to 1). 
                                     For example, 0.3 means 30% scatter (scatter/total). Default: 0.0.
            random_fraction (float): Fraction of random events relative to prompt events (0 to 1).
                                    For example, 0.2 means 20% randoms (randoms/prompt). Default: 0.0.
            seed (int, optional): Random seed for reproducibility. Default: None.
            attenuation_scale (float): Scaling factor for attenuation map values to convert to mm^-1.
                                      Default: 0.01 (assumes attenuation map is in relative units).
                                      For water at 511 keV: 0.096 cm^-1 = 0.0096 mm^-1.
        
        Returns:
            sinogram (np.ndarray): Simulated sinogram data.
        """
        self.nx, self.ny, self.nz = activity_distribution.shape

        # Get grid
        self.get_grid()
        # Get LORs
        self.lors = self.get_list_LORs()
        n_lors = self.lors.shape[0]

        # Prepare parameters
        vol_shape = (self.nx, self.ny, self.nz)
        voxel_size = (self.vx, self.vy, self.vz)
        origin = (
            - (self.nx * self.vx) / 2,
            - (self.ny * self.vy) / 2,
            - (self.nz * self.vz) / 2
        )

        # Initialize sinogram
        sinogram = np.zeros(n_lors)
        
        # Create batches of LOR indices
        lor_indices = list(range(n_lors))
        batches = [lor_indices[i:i + batch_size] for i in range(0, n_lors, batch_size)]
        
        if use_multiprocessing and n_lors > batch_size:
            # Use multiprocessing
            if n_processes is None:
                n_processes = cpu_count()
            
            print(f"Processing {n_lors} LORs in {len(batches)} batches using {n_processes} processes...")
            if scatter_fraction > 0 or random_fraction > 0:
                print(f"  Scatter fraction: {scatter_fraction:.2%}, Random fraction: {random_fraction:.2%}")
            if attenuation_map is not None:
                print(f"  Attenuation scaling: {attenuation_scale} (effective mu range: [{attenuation_map.min()*attenuation_scale:.6f}, {attenuation_map.max()*attenuation_scale:.6f}] mm^-1)")
            
            # Create a partial function with fixed parameters
            process_func = partial(
                self._process_lor_batch,
                activity_distribution=activity_distribution,
                attenuation_map=attenuation_map,
                vol_shape=vol_shape,
                voxel_size=voxel_size,
                origin=origin,
                scatter_fraction=scatter_fraction,
                random_fraction=random_fraction,
                seed=seed,
                attenuation_scale=attenuation_scale
            )
            
            # Process batches in parallel
            with Pool(processes=n_processes) as pool:
                batch_results = list(tqdm.tqdm(
                    pool.imap(process_func, batches),
                    total=len(batches),
                    desc="Forward projection"
                ))
            
            # Combine results
            idx = 0
            for batch_result in batch_results:
                batch_len = len(batch_result)
                sinogram[idx:idx + batch_len] = batch_result
                idx += batch_len
        else:
            # Sequential processing with batches (useful for small problems or debugging)
            print(f"Processing {n_lors} LORs in {len(batches)} batches sequentially...")
            if scatter_fraction > 0 or random_fraction > 0:
                print(f"  Scatter fraction: {scatter_fraction:.2%}, Random fraction: {random_fraction:.2%}")
            if attenuation_map is not None:
                print(f"  Attenuation scaling: {attenuation_scale} (effective mu range: [{attenuation_map.min()*attenuation_scale:.6f}, {attenuation_map.max()*attenuation_scale:.6f}] mm^-1)")
            for batch in tqdm.tqdm(batches, desc="Forward projection"):
                batch_result = self._process_lor_batch(
                    batch, activity_distribution, attenuation_map,
                    vol_shape, voxel_size, origin,
                    scatter_fraction, random_fraction, seed,
                    attenuation_scale
                )
                for local_idx, lor_idx in enumerate(batch):
                    sinogram[lor_idx] = batch_result[local_idx]
        
        # Add Poisson noise if requested
        if add_noise:
            if seed is not None:
                np.random.seed(seed)
            # Only add noise to non-zero values to avoid issues
            mask = sinogram > 0
            sinogram[mask] = np.random.poisson(sinogram[mask])
            print(f"Applied Poisson noise to sinogram")

        return sinogram

    def build_sinogram(self, activity_distribution, attenuation_map=None,
                    n_angles=180, n_radial_bins=128, max_radius=200,
                    batch_size=1000, n_processes=None, use_multiprocessing=True,
                    add_noise=False, scatter_fraction=0.0, random_fraction=0.0, seed=None,
                    attenuation_scale=0.01, max_ring_diff=0):
        """
        Build a 2D sinogram from the forward projection data.
        
        Args:
            activity_distribution (np.ndarray): 3D activity distribution.
            attenuation_map (np.ndarray, optional): 3D attenuation map.
            n_angles (int): Number of angular bins (0 to π). Default: 180.
            n_radial_bins (int): Number of radial bins. Default: 128.
            max_radius (float): Maximum radial distance in mm. Default: 200.
            batch_size (int): Batch size for processing. Default: 1000.
            n_processes (int, optional): Number of processes for multiprocessing.
            use_multiprocessing (bool): Enable multiprocessing. Default: True.
            add_noise (bool): Add Poisson noise. Default: False.
            scatter_fraction (float): Scatter fraction. Default: 0.0.
            random_fraction (float): Random fraction. Default: 0.0.
            seed (int, optional): Random seed.
            attenuation_scale (float): Attenuation scaling factor. Default: 0.01.
            max_ring_diff (int): Maximum ring difference to include (0 for central plane only). Default: 0.
        
        Returns:
            np.ndarray: 2D sinogram of shape (n_angles, n_radial_bins).
        """

        flattened = self.forward_project(
            activity_distribution, attenuation_map=attenuation_map, 
            batch_size=batch_size, n_processes=n_processes, use_multiprocessing=use_multiprocessing,
            add_noise=add_noise, scatter_fraction=scatter_fraction, random_fraction=random_fraction, seed=seed,
            attenuation_scale=attenuation_scale
        )

        sinogram = np.zeros((n_angles, n_radial_bins), dtype=np.float32)
        
        # Crystal to ring mapping
        n_crystals = self.n_rings * self.n_detectors_per_ring
        ring_of = np.repeat(np.arange(self.n_rings), self.n_detectors_per_ring)

        # Loop over all LORs
        idx = 0
        n_included = 0
        n_excluded = 0
        
        for i in range(n_crystals):
            for j in range(i + 1, n_crystals):
                # Get ring indices
                ring_i = ring_of[i]
                ring_j = ring_of[j]
                ring_diff = abs(ring_j - ring_i)
                
                # Filter by ring difference
                if ring_diff > max_ring_diff:
                    idx += 1
                    n_excluded += 1
                    continue
                
                count = flattened[idx]
                
                # Skip if no counts
                if count == 0:
                    idx += 1
                    continue
                
                # Get LOR endpoints
                p0, p1 = self.lors[idx]
                
                # Extract 2D coordinates (x,y plane only)
                p0_xy = np.array([p0[0], p0[1]], dtype=np.float64)
                p1_xy = np.array([p1[0], p1[1]], dtype=np.float64)
                
                # Line direction vector
                d = p1_xy - p0_xy
                d_norm = np.linalg.norm(d)
                
                if d_norm < 1e-10:
                    idx += 1
                    continue
                
                d = d / d_norm  # normalize
                
                # Normal vector (perpendicular to line, rotated 90° CCW)
                n = np.array([-d[1], d[0]], dtype=np.float64)
                
                # Perpendicular distance from origin to line: s = p0 · n
                s = np.dot(p0_xy, n)
                
                # Angle of normal vector
                theta = np.arctan2(n[1], n[0])
                
                # Fold to [0, π) and adjust sign of s accordingly
                if theta < 0:
                    theta += np.pi
                    s = -s
                if theta >= np.pi:
                    theta -= np.pi
                    s = -s
                
                # Convert to bin indices
                angle_id = int((theta / np.pi) * n_angles)
                angle_id = np.clip(angle_id, 0, n_angles - 1)
                
                # Map s to radial bins: s ranges from -max_radius to +max_radius
                radial_id = int(((s + max_radius) / (2.0 * max_radius)) * n_radial_bins)
                
                # Accumulate counts in sinogram
                if 0 <= radial_id < n_radial_bins:
                    sinogram[angle_id, radial_id] += count
                    n_included += 1
                
                idx += 1
        
        print(f"Sinogram built: {n_included} LORs included, {n_excluded} LORs excluded (ring diff > {max_ring_diff})")
        print(f"Sinogram shape: {sinogram.shape}, non-zero bins: {np.count_nonzero(sinogram)}")
        print(f"Sinogram range: [{sinogram.min():.2f}, {sinogram.max():.2f}], mean: {sinogram.mean():.2f}")
        
        return sinogram



if __name__ == "__main__":

    simulator = SinogramSimulatorHomemade(
        voxel_size=(2.0, 2.0, 2.0),
        n_rings=1,
        n_detectors_per_ring=512,
        ring_spacing=4,
        scanner_radius=300.0
    )
    
    crystal_map = simulator.get_crystal_map()

    with open('/workspace/brain_web_phantom/gt_web_after_scaling.img', 'rb') as f:
        image = np.frombuffer(f.read(), dtype=np.float32).reshape((160, 160, 1))


    with open('/workspace/brain_web_phantom/attenuat_brain_phantom.img', 'rb') as f:
        attenuation_map = np.frombuffer(f.read(), dtype=np.float32).reshape((160, 160, 1))
    
    
    # Estimate appropriate attenuation scale
    # Water at 511 keV: 0.096 cm^-1 = 0.0096 mm^-1
    # If attenuation map is normalized (0-1 range), scale should be around 0.0096
    # If it's in HU or other units, adjust accordingly


    # Test with standard geometric sinogram - central plane only
    print("\n=== Building 2D sinogram for central plane (ring diff = 0) ===")
    sinogram = simulator.build_sinogram(
        activity_distribution=image,
        attenuation_map=attenuation_map,
        n_angles=256,
        n_radial_bins=256,
        max_radius=image.shape[0] // 2,
        batch_size=256,
        n_processes=None,
        use_multiprocessing=True,
        add_noise=False,
        scatter_fraction=0.30,
        random_fraction=0.20,
        seed=42,
        attenuation_scale=0.1,  # it is given in cm^-1 instead of mm^-1
        max_ring_diff=0  # Only include central plane (ring difference = 0)
    )

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Sinogram
        im0 = axes[0].imshow(sinogram, aspect='auto', cmap='gray', 
                            interpolation='nearest')
        axes[0].set_title('2D Sinogram (Central Plane)')
        axes[0].set_xlabel('Radial bin')
        axes[0].set_ylabel('Angle bin')
        plt.colorbar(im0, ax=axes[0])
        
        # Original image for comparison
        im1 = axes[1].imshow(image[:, :, 0], cmap='gray')
        axes[1].set_title('Original Activity Image')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not installed, skipping sinogram plot.")