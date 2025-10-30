import pytest
import numpy as np
import shutil
import os

from phantom_simulation.object_simulator import Phantom2DPetGenerator
from phantom_simulation.sinogram_simulator import SinogramSimulator

class TestObjectSimulatorSeed:

    def test_object_simulator_seed(self):

        gen1 = Phantom2DPetGenerator()
        gen2 = Phantom2DPetGenerator()

        # Run
        gen1.set_seed(42)
        obj1, att1 = gen1.run(dest_path='./tmp_object')
        with open(obj1 + '.img', 'rb') as f:
            obj1 = np.fromfile(f, dtype=np.float32).reshape((256,256))
        with open(att1 + '.img', 'rb') as f:
            att1 = np.fromfile(f, dtype=np.float32).reshape((256,256))

        gen2.set_seed(42)
        obj2, att2 = gen2.run(dest_path='./tmp_object')
        with open(obj2 + '.img', 'rb') as f:
            obj2 = np.fromfile(f, dtype=np.float32).reshape((256,256))
        with open(att2 + '.img', 'rb') as f:
            att2 = np.fromfile(f, dtype=np.float32).reshape((256,256))

        # Clean it out
        shutil.rmtree('./tmp_object')

        assert np.all(np.equal(obj1, obj2)), "Different objects with same seed"
        assert np.all(np.equal(att1, att2)), "Different attenuation maps with same seed"

    # NOTE : the following test is disabled because the castor software seems not to be seedable/repeatable
    
    # def test_sinogram_simulator_seed(self):

    #     obj_path, att_path = Phantom2DPetGenerator().run('./tmp_sino/object')

    #     binsimu = os.path.join(os.getenv("WORKSPACE"), "simulator", "bin")
    #     gen1 = SinogramSimulator(binsimu=binsimu)
    #     gen2 = SinogramSimulator(binsimu=binsimu)

    #     gen1.set_seed(42)
    #     gen1.run(img_path=obj_path, img_att_path=att_path, dest_path='./tmp_sino1')
    #     #
    #     with open('./tmp_sino1/simu/simu_nfpt.s', 'rb') as f:
    #         sino1 = np.fromfile(f, dtype=np.float32)

    #     gen2.set_seed(42)
    #     gen2.run(img_path=obj_path, img_att_path=att_path, dest_path='./tmp_sino2')
    #     #
    #     with open('./tmp_sino2/simu/simu_nfpt.s', 'rb') as f:
    #         sino2 = np.fromfile(f, dtype=np.float32)

    #     shutil.rmtree('./tmp_sino')

    #     assert np.all(np.equal(sino1, sino2))