import os

class SinogramSimulator:

    def __init__(self, binsimu, dout='./', seed=42, scanner_name='mmr2d', random_deficiencies=10, save_castor=True, verbose=0):
        """
        param binsium: absolute path to the binaries used in this class.
        param dout: destinaion path to store generated sinograms.
        param scanner_name: Scanner name to generate sinogram from.
        param random_deficiencies: Random deficiencies of crystal sensitivities, e.g. give 10 means 10% around 1 randomly shot.
        param save_castor: Whether to save castor file for reconstruction later on.
        """
        self.binsimu = binsimu
        self.seed = seed
        self.scanner_name = scanner_name
        self.random_deficiencies = random_deficiencies
        self.save_castor = save_castor
        self.verbose = verbose

    def create_crystal_map(self):

        self.cmap_out = os.path.join(self.dout, 'cmap')
        cmd = f"cd {self.dout} && {self.binsimu}/create_crystal_map.exe -m {self.scanner_name} -r {self.random_deficiencies} -o cmap -w -v {self.verbose}"
        os.system(cmd)

    def simulate(self, img_path, img_att_path,
                scatter_component=0.35,
                random_component=0.40,
                gaussian_PSF=4.,
                nb_count=3000000
    ):
        """
        param img_path: pre-generated object to simulate sinograms from.
        param img_att_path: attenuation map of object
        param scatter_component: include a scatter component from the given scatter fraction (scat/net_true)
        param random_component: include a random component from the given random fraction (rand/prompt)
        param gaussian_PSF: give the FWHM in mm of the 3D gaussian PSF (can give 2 values separated by a comma for transaxial and axial FWHM)
        nb_count: Total number of prompts
        """
        if not img_path.endswith('.hdr'):
            img_path += '.hdr'
        if not img_att_path.endswith('.hdr'):
            img_att_path += '.hdr'

        cmd = \
            f"cd {self.dout} && {self.binsimu}/simulator.exe -m {self.scanner_name} -c {os.path.join(self.dout, 'cmap', 'cmap.ecm')}" \
            f" -i {img_path} -a {img_att_path} -s {scatter_component} -r {random_component} -p {gaussian_PSF}" \
            f" -v {self.verbose} -P {nb_count} -o simu"
        os.system(cmd)

    def create_castor_data(self):


        cmd = \
            f"cd {self.dout} && {self.binsimu}/create_castor_data.exe -m {self.scanner_name} -o castor_data" \
            f" -p simu/simu_pt.s.hdr" \
            f" -r simu/simu_rd.s.hdr" \
            f" -s simu/simu_sc.s.hdr" \
            f" -n simu/simu_nm.s.hdr" \
            f" -A simu/simu_at.s" \
            f" -c cmap/cmap.ecm -castor -v {self.verbose}"
        os.system(cmd)

    def run(self, img_path, img_att_path, dest_path):

        #
        if not os.path.join(dest_path):
            os.makedirs(dest_path)
        self.dout = dest_path

        # Create crystal map
        if not hasattr(self, 'cmap_out') or (hasattr(self, 'cmap_out') and not os.path.exists(self.cmap_out)):
            self.create_crystal_map()

        # Simulation
        self.simulate(img_path, img_att_path)

        # Save castor data
        if self.save_castor:
            self.create_castor_data()

if __name__ == '__main__':

    # Create object first
    from object_simulator import Phantom2DPetGenerator
    generator = Phantom2DPetGenerator(shape=(256,256), voxel_size=(2,2,2))
    obj_path, att_path = generator.run('/workspace/development/data/data1/object')

    ss = SinogramSimulator(binsimu='/workspace/development/simulator/bin')
    ss.run(img_path=obj_path, img_att_path=att_path, dest_path='/workspace/development/data/data1')